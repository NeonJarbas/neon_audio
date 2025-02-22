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

import time
from threading import Lock, Thread, Event
from typing import Optional

from mycroft_bus_client import Message, MessageBusClient

from neon_audio.tts import TTSFactory, TTS
from mycroft.configuration import Configuration
from mycroft.tts.remote_tts import RemoteTTSTimeoutException
from mycroft.tts.mimic_tts import Mimic
from mycroft.metrics import report_timing, Stopwatch
from mycroft.util.log import LOG
from mycroft.util.signal import check_for_signal

bus: Optional[MessageBusClient] = None  # Mycroft messagebus connection
config = None
tts: Optional[TTS] = None
mimic_fallback_obj: Optional[TTS] = None
tts_hash = None
lock = Lock()
speak_muted = False

_last_stop_signal = 0


def handle_unmute_tts(event):
    """ enable tts execution """
    global speak_muted
    speak_muted = False
    bus.emit(Message("mycroft.tts.mute_status", {"muted": speak_muted}))


def handle_mute_tts(event):
    """ disable tts execution """
    global speak_muted
    speak_muted = True
    bus.emit(Message("mycroft.tts.mute_status", {"muted": speak_muted}))


def handle_mute_status(event):
    """ emit tts mute status to bus """
    bus.emit(Message("mycroft.tts.mute_status", {"muted": speak_muted}))


def handle_get_tts(message):
    """
    Handle a request to get TTS only
    :param message: Message associated with request
    """
    global tts
    text = message.data.get("text")
    ident = message.context.get("ident") or "neon.get_tts.response"
    if not message.data.get("speaker"):
        LOG.warning(f"No speaker data with request, core defaults will be used.")
    if text:
        if not isinstance(text, str):
            bus.emit(message.reply(ident, data={"error": f"text is not a str: {text}"}))
            return
        try:
            responses = tts.execute(text, message=message)
            # TODO: Consider including audio bytes here in case path is inaccessible DM
            # responses = {lang: {sentence: text, male: Optional[path], female: Optional[path}}
            bus.emit(message.reply(ident, data=responses))
        except Exception as e:
            LOG.error(e)
            bus.emit(message.reply(ident, data={"error": repr(e)}))
    else:
        bus.emit(message.reply(ident, data={"error": "No text provided."}))


def handle_speak(message):
    """Handle "speak" message

    Parse sentences and invoke text to speech service.
    """
    # Configuration.set_config_update_handlers(bus)
    global _last_stop_signal

    message.context = message.context or {}

    # if the message is targeted and audio is not the target don't
    # don't synthezise speech
    # if 'audio' not in event.context.get('destination', ['audio']):
    #     return

    # Get conversation ID
    message.context['ident'] = message.context.get("ident", "unknown")

    with lock:
        stopwatch = Stopwatch()
        stopwatch.start()
        utterance = message.data['utterance']
        mute_and_speak(utterance, message)
        stopwatch.stop()
    report_timing(message.context['ident'], 'speech', stopwatch,
                  {'utterance': utterance, 'tts': tts.__class__.__name__})


def mute_and_speak(utterance, message):
    """Mute mic and start speaking the utterance using selected tts backend.

    Arguments:
        utterance:  The sentence to be spoken
        message:    Message associated with request
    """
    global tts_hash, speak_muted, tts
    LOG.info("Speak: " + utterance)
    if speak_muted:
        LOG.warning("Tried to speak, but TTS is muted!")
        return

    listen = message.data.get('expect_response', False)

    # update TTS object if configuration has changed
    if tts_hash != hash(str(config.get('tts', ''))):
        # Stop tts playback thread
        tts.playback.stop()
        tts.playback.join()
        # Create new tts instance
        tts = TTSFactory.create(config)
        tts.init(bus)
        tts_hash = hash(str(config.get('tts', '')))

    try:
        tts.execute(utterance, message.context['ident'],
                    listen, message)
    except RemoteTTSTimeoutException as e:
        LOG.error(e)
        mimic_fallback_tts(utterance, message.context['ident'], message)
    except Exception as e:
        LOG.error('TTS execution failed ({})'.format(repr(e)))


def mimic_fallback_tts(utterance, ident, event=None):
    global mimic_fallback_obj
    # TODO: This could also be Mozilla TTS DM
    # fallback if connection is lost
    config = Configuration.get()
    tts_config = config.get('tts', {}).get("mimic", {})
    lang = config.get("lang", "en-us")
    if not mimic_fallback_obj:
        mimic_fallback_obj = Mimic(lang, tts_config)
    mimic_tts = mimic_fallback_obj
    LOG.debug("Mimic fallback, utterance : " + str(utterance))
    mimic_tts.init(bus)
    mimic_tts.execute(utterance, ident)


def handle_stop(event):
    """Handle stop message.

    Shutdown any speech.
    """
    global _last_stop_signal
    if check_for_signal("isSpeaking", -1):
        _last_stop_signal = time.time()
        tts.playback.clear()  # Clear here to get instant stop
        bus.emit(Message("mycroft.stop.handled", {"by": "TTS"}))


def init(messagebus, conf=None):
    """Start speech related handlers.

    Arguments:
        messagebus: Connection to the Mycroft messagebus
        conf: configuration override
    """

    global bus
    global tts
    global tts_hash
    global config

    bus = messagebus
    # Configuration.set_config_update_handlers(bus)
    # config = Configuration.get()
    config = conf or {}
    bus.on('mycroft.stop', handle_stop)
    bus.on('mycroft.audio.speech.stop', handle_stop)
    bus.on('speak', handle_speak)
    bus.on('mycroft.tts.mute', handle_mute_tts)
    bus.on('mycroft.tts.unmute', handle_unmute_tts)
    bus.on('mycroft.tts.mute_status.request', handle_mute_status)

    # API Methods
    bus.on("neon.get_tts", handle_get_tts)

    tts = TTSFactory.create(config)
    tts.init(bus)
    tts_hash = hash(str(config.get('tts', '')))


def shutdown():
    """Shutdown the audio service cleanly.

    Stop any playing audio and make sure threads are joined correctly.
    """
    if tts:
        tts.playback.stop()
        tts.playback.join()
    if mimic_fallback_obj:
        mimic_fallback_obj.playback.stop()
        mimic_fallback_obj.playback.join()
