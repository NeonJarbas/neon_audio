# NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2021 Neongecko.com Inc.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
#    and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
#    and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import hashlib
import os
import os.path
import pathlib
import pickle
import re
from os.path import exists

from mycroft.tts.tts import TTS, TTSValidator, TTSFactory as \
    MycroftTTSFactory, EMPTY_PLAYBACK_QUEUE_TUPLE
from mycroft.util import (
    check_for_signal, create_signal
)
from mycroft.configuration import Configuration
from mycroft.util.log import LOG
from mycroft_bus_client import Message
from neon_utils.language_utils import DetectorFactory, TranslatorFactory, \
    get_neon_lang_config
from ovos_plugin_manager.tts import load_tts_plugin
from inspect import signature


class WrappedTTS(TTS):
    def __new__(cls, base_engine, *args, **kwargs):
        base_engine.begin_audio = cls.begin_audio
        base_engine.end_audio = cls.end_audio
        base_engine.execute = cls.execute
        base_engine.__neon_preprocess_sentence = cls.__neon_preprocess_sentence
        return cls._init_neon(base_engine, *args, **kwargs)

    @staticmethod
    def _init_neon(base_engine, *args, **kwargs):
        """ called after the __init__ method to inject neon-core properties
        into the selected TTS engine """
        base_engine = base_engine(*args, **kwargs)
        base_engine.language_config = Configuration.get()["language"]
        base_engine.lang_detector = DetectorFactory.create()
        base_engine.translator = TranslatorFactory.create()
        base_engine.lang = base_engine.lang or \
                           base_engine.language_config.get("user", "en-us")
        base_engine.keys = {}
        return base_engine

    def begin_audio(self, ident=None):
        """Helper function for child classes to call in execute()"""
        # Create signals informing start of speech
        self.bus.emit(Message("recognizer_loop:audio_output_start",
                              context={"ident": ident}))

    def end_audio(self, listen=False, ident=None):
        """Helper function for child classes to call in execute().

        Sends the recognizer_loop:audio_output_end message (indicating
        that speaking is done for the moment) as well as trigger listening
        if it has been requested. It also checks if cache directory needs
        cleaning to free up disk space.

        Arguments:
            listen (bool): indication if listening trigger should be sent.
            ident (str): Identifier of the input utterance associated with the response
        """

        self.bus.emit(Message("recognizer_loop:audio_output_end",
                              context={"ident": ident}))
        if listen:
            self.bus.emit(Message('mycroft.mic.listen'))

        self.cache.curate()
        # This check will clear the "signal"
        check_for_signal("isSpeaking")

    def __neon_preprocess_sentence(self, sentence):
        # multi lang support
        # NOTE this is kinda optional because skills will translate
        # However speak messages might be sent directly to bus
        # this is here to cover that use case

        # # check for user specified language
        # if message and hasattr(message, "user_data"):
        #     user_lang = message.user_data.get("lang") or self.language_config["user"]
        # else:
        #     user_lang = self.language_config["user"]
        #
        # detected_lang = self.lang_detector.detect(sentence)
        # LOG.debug("Detected language: {lang}".format(lang=detected_lang))
        # if detected_lang != user_lang.split("-")[0]:
        #     sentence = self.translator.translate(sentence, user_lang)
        return sentence

    def execute(self, sentence, ident=None, listen=False, message=None):
        """Convert sentence to speech, preprocessing out unsupported ssml

            The method caches results if possible using the hash of the
            sentence.

            Arguments:
                sentence:   Sentence to be spoken
                ident:      Id reference to current interaction
                listen:     True if listen should be triggered at the end
                            of the utterance.
                message:    Message associated with request
        """
        sentence = self.validate_ssml(sentence)
        sentence = self.__neon_preprocess_sentence(sentence)
        create_signal("isSpeaking")
        try:
            # check the signature to either pass a message or not
            if len(signature(self._execute).parameters) == 5:
                return self._execute(sentence, ident, listen, message)
            else:
                return self._execute(sentence, ident, listen)
        except Exception:
            # If an error occurs end the audio sequence through an empty entry
            self.queue.put(EMPTY_PLAYBACK_QUEUE_TUPLE)
            # Re-raise to allow the Exception to be handled externally as well.
            raise


class KlatWrappedTTS(WrappedTTS):
    def __new__(cls, base_engine, *args, **kwargs):
        base_engine.begin_audio = cls.begin_audio
        base_engine.end_audio = cls.end_audio
        base_engine.execute = cls.execute
        base_engine._execute = cls._execute
        base_engine.__neon_preprocess_sentence = cls.__neon_preprocess_sentence
        return cls._init_neon(base_engine, *args, **kwargs)

    @staticmethod
    def _init_neon(base_engine, *args, **kwargs):
        base_engine = WrappedTTS._init_neon(base_engine, *args, **kwargs)
        # TODO subclass and replace base_engine.cache instead
        base_engine.cache_dir = base_engine.cache.temporary_cache_dir
        base_engine.translation_cache = os.path.join(base_engine.cache_dir,
                                                     'lang_dict.txt')
        if not pathlib.Path(base_engine.translation_cache).exists():
            base_engine.cached_translations = {}
            os.makedirs(os.path.dirname(base_engine.translation_cache),
                        exist_ok=True)
            open(base_engine.translation_cache, 'wb+').close()
        else:
            with open(base_engine.translation_cache,
                      'rb') as cached_utterances:
                try:
                    base_engine.cached_translations = pickle.load(
                        cached_utterances)
                except EOFError:
                    base_engine.cached_translations = {}
                    LOG.info("Cache file exists, but it's empty so far")
        return base_engine

    def _execute(self, sentence: str, ident: str, listen: bool,
                 message: Message):
        def _get_requested_tts_languages(msg) -> list:
            """
            Builds a list of the requested TTS for a given spoken response
            :param msg: Message associated with request
            :return: List of TTS dict data
            """
            profiles = msg.context.get("nick_profiles")
            tts_name = "Neon"

            tts_reqs = []
            # Get all of our language parameters
            try:
                # If speaker data is present, use it
                if msg.data.get("speaker"):
                    speaker = msg.data.get("speaker")
                    tts_reqs.append({"speaker": speaker["name"],
                                     "language": speaker["language"],
                                     "gender": speaker["gender"],
                                     "voice": speaker.get("voice")
                                     })
                    LOG.debug(f">>> speaker={speaker}")

                # If multiple profiles attached to message, get TTS for all of them
                elif profiles:
                    LOG.info(f"Got profiles: {profiles}")
                    for nickname in profiles:
                        chat_user = profiles.get(nickname, None)
                        user_lang = chat_user.get("speech", chat_user)
                        language = user_lang.get('tts_language', 'en-us')
                        gender = user_lang.get('tts_gender', 'female')
                        LOG.debug(f"{nickname} requesting {gender} {language}")
                        data = {"speaker": tts_name,
                                "language": language,
                                "gender": gender,
                                "voice": None
                                }
                        if data not in tts_reqs:
                            tts_reqs.append(data)

                # General non-server response, use yml configuration
                else:
                    user_config = get_neon_user_config()["speech"]

                    tts_reqs.append({"speaker": tts_name,
                                     "language": user_config["tts_language"],
                                     "gender": user_config["tts_gender"],
                                     "voice": user_config["neon_voice"]
                                     })

                    if user_config["secondary_tts_language"] and \
                            user_config["secondary_tts_language"] != \
                            user_config["tts_language"]:
                        tts_reqs.append({"speaker": tts_name,
                                         "language": user_config[
                                             "secondary_tts_language"],
                                         "gender": user_config[
                                             "secondary_tts_gender"],
                                         "voice": user_config[
                                             "secondary_neon_voice"]
                                         })
            except Exception as x:
                LOG.error(x)

            # TODO: Associate voice with cache here somehow? (would be a per-TTS engine set) DM
            LOG.debug(f"Got {len(tts_reqs)} TTS Voice Requests")
            return tts_reqs

        def _update_pickle():
            with open(self.translation_cache, 'wb+') as cached_utterances:
                pickle.dump(self.cached_translations, cached_utterances)

        if self.phonetic_spelling:
            for word in re.findall(r"[\w']+", sentence):
                if word.lower() in self.spellings:
                    sentence = sentence.replace(word,
                                                self.spellings[word.lower()])

        chunks = self._preprocess_sentence(sentence)
        # Apply the listen flag to the last chunk, set the rest to False
        chunks = [(chunks[i], listen if i == len(chunks) - 1 else False)
                  for i in range(len(chunks))]

        for sentence, l in chunks:
            key = str(hashlib.md5(
                sentence.encode('utf-8', 'ignore')).hexdigest())
            phonemes = None
            response_audio_files = []
            tts_requested = _get_requested_tts_languages(message)
            LOG.debug(f"tts_requested={tts_requested}")

            # Go through all the audio we need and see if it is in the cache
            responses = {}
            for request in tts_requested:
                file = os.path.join(self.cache_dir, "tts", self.tts_name,
                                    request["language"], request["gender"],
                                    key + '.' + self.audio_ext)
                lang = request["language"]
                translated_sentence = None
                try:
                    # Handle any missing cache directories
                    if not exists(os.path.dirname(file)):
                        os.makedirs(os.path.dirname(file), exist_ok=True)

                    # Get cached text response
                    if os.path.exists(file):
                        LOG.debug(f">>>{lang}{key} in cache<<<")
                        phonemes = self.load_phonemes(key)
                        LOG.debug(phonemes)

                        # Get cached translation (remove audio if no corresponding translation)
                        if f"{lang}{key}" in self.cached_translations:
                            translated_sentence = self.cached_translations[
                                f"{lang}{key}"]
                        else:
                            LOG.error("cache error! Removing audio file")
                            os.remove(file)

                    # If no file cached or cache error was encountered, get tts
                    if not translated_sentence:
                        LOG.debug(f"{lang}{key} not cached")
                        if not lang.split("-", 1)[
                                   0] == "en":  # TODO: Internal lang DM
                            try:
                                translated_sentence = self.translator.translate(
                                    sentence, lang, "en")
                                # request["translated"] = True
                                LOG.info(translated_sentence)
                            except Exception as e:
                                LOG.error(e)
                        else:
                            translated_sentence = sentence
                        file, phonemes = self.get_tts(translated_sentence,
                                                      file, request)
                        # Update cache for next time
                        self.cached_translations[
                            f"{lang}{key}"] = translated_sentence
                        LOG.debug(">>>Cache Updated!<<<")
                        _update_pickle()
                except Exception as e:
                    # Remove audio file if any exception occurs, this forces re-translation/cache next time
                    LOG.error(e)
                    if os.path.exists(file):
                        os.remove(file)

                if not responses.get(lang):
                    responses[lang] = {"sentence": translated_sentence}
                if os.path.isfile(
                        file):  # Based on <speak> tags, this may not exist
                    responses[lang][request["gender"]] = file
                    response_audio_files.append(file)

            # Server execution - send mycroft's speech (wav file) over to the chat_server
            if message.context.get("klat_data"):
                LOG.debug(f"responses={responses}")
                self.bus.emit(
                    message.forward("klat.response", {"responses": responses,
                                                      "speaker": message.data.get(
                                                          "speaker")}))
                # self.bus.wait_for_response
            # API Call
            elif message.msg_type in ["neon.get_tts"]:
                return responses
            # Non-server execution
            else:
                if response_audio_files:
                    vis = self.viseme(phonemes) if phonemes else phonemes
                    for response in response_audio_files:
                        self.queue.put((self.audio_ext, str(response), vis,
                                        ident, listen))
                else:
                    check_for_signal("isSpeaking")


class TTSFactory(MycroftTTSFactory):

    @staticmethod
    def create(config=None):
        """Factory method to create a TTS engine based on configuration.

        The configuration file ``mycroft.conf`` contains a ``tts`` section with
        the name of a TTS module to be read by this method.

        "tts": {
            "module": <engine_name>
        }
        """
        config = config or Configuration.get()
        lang = config.get("language", {}).get("user") or \
               config.get("lang", "en-us")
        tts_module = config.get('tts', {}).get('module', 'mimic')
        tts_config = config.get('tts', {}).get(tts_module, {})
        tts_lang = tts_config.get('lang', lang)
        try:
            if tts_module in TTSFactory.CLASSES:
                clazz = TTSFactory.CLASSES[tts_module]
            else:
                clazz = load_tts_plugin(tts_module)
                LOG.info('Loaded plugin {}'.format(tts_module))
            if clazz is None:
                raise ValueError('TTS module not found')

            # TODO if klat use KlatWrappedTTS
            tts = WrappedTTS(clazz, tts_lang, tts_config)
            tts.validator.validate()
        except Exception as e:
            LOG.error(e)
            # Fallback to mimic if an error occurs while loading.
            if tts_module != 'mimic':
                LOG.exception('The selected TTS backend couldn\'t be loaded. '
                              'Falling back to Mimic')
                clazz = TTSFactory.CLASSES.get('mimic')
                # TODO if klat use KlatWrappedTTS
                tts = WrappedTTS(clazz, tts_lang, tts_config)
                tts.validator.validate()
            else:
                LOG.exception('The TTS could not be loaded.')
                raise

        return tts
