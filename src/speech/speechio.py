from io import BytesIO
from typing import Callable, ClassVar, Mapping, Optional, Protocol, Sequence, cast
from enum import Enum
import os
import re
import asyncio
import hashlib
import wave
import time
import threading
import pyaudio
import json
import time
from typing_extensions import Self

from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, AudioInfo
from viam.resource.base import ResourceBase
from viam.components.audio_in import AudioIn
from viam.components.audio_out import AudioOut
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model
from viam.utils import struct_to_dict

import numpy as np
import pygame
from pygame import mixer
from elevenlabs.client import ElevenLabs
from elevenlabs import save as eleven_save
from gtts import gTTS
import openai
import speech_recognition as sr
from pydub import AudioSegment, silence
import webrtcvad
import tempfile

try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

from speech_service_api import SpeechService


class SpeechProvider(str, Enum):
    google = "google"
    elevenlabs = "elevenlabs"


class CompletionProvider(str, Enum):
    openai = "openai"


class Closer(Protocol):
    def __call__(self, wait_for_stop: bool = True) -> None: ...


class RecState:
    listen_closer: Optional[Closer] = None
    mic: Optional[sr.Microphone] = None
    rec: Optional[sr.Recognizer] = None
    # Vosk VAD components
    vosk_model: Optional[object] = None
    vosk_rec: Optional[object] = None
    vosk_stream: Optional[object] = None
    vosk_thread: Optional[threading.Thread] = None
    vosk_stop_event: Optional[threading.Event] = None
    vosk_closer: Optional[Callable] = None  # For microphone_client version
    # new AUDIO API fields
    audio_listen_task: Optional[asyncio.Task] = None
    audio_stop_event: Optional[asyncio.Event] = None
    # STT throttling
    stt_in_progress: bool = False


CACHEDIR = "/tmp/cache"

rec_state = RecState()


class SpeechIOService(SpeechService, EasyResource):
    """This is the specific implementation of a ``SpeechService`` (defined in api.py)

    It inherits from SpeechService, as well as conforms to the ``Reconfigurable`` protocol, which signifies that this component can be
    reconfigured. It also specifies a function ``SpeechIOService.new``, which conforms to the ``resource.types.ResourceCreator`` type,
    which is required for all models.
    """

    MODEL: ClassVar[Model] = Model.from_string("viam-labs:speech:speechio")
    speech_provider: SpeechProvider
    speech_provider_key: str
    speech_voice: str
    completion_provider: CompletionProvider
    completion_model: str
    completion_provider_org: str
    completion_provider_key: str
    completion_persona: str
    should_listen: bool
    stt_provider: str
    listen_trigger_say: str
    listen_trigger_completion: str
    listen_trigger_command: str
    listen_command_buffer_length: int
    mic_device_name: str
    command_list: list
    trigger_active: bool
    active_trigger_type: str
    disable_mic: bool
    disable_audioout: bool
    eleven_client: dict = {}
    main_loop: Optional[asyncio.AbstractEventLoop] = None
    microphone: str
    microphone_client: Optional[AudioIn] = None
    speaker: str
    speaker_client: Optional[AudioOut] = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        return super().new(config, dependencies)

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Sequence[str]:
        """This method allows you to validate the configuration object received from the machine,
        as well as to return any implicit dependencies based on that `config`.

        Args:
            config (ComponentConfig): The configuration for this resource

        Returns:
            Sequence[str]: A list of implicit dependencies
        """
        deps = []
        attrs = struct_to_dict(config.attributes)
        stt_provider = str(attrs.get("stt_provider", ""))
        if stt_provider != "" and stt_provider != "google":
            deps.append(stt_provider)
        microphone = str(attrs.get("microphone_name", ""))
        if microphone != "":
            deps.append(microphone)
        speaker = str(attrs.get("speaker_name", ""))
        if speaker != "":
            deps.append(speaker)
        return deps

    async def say(self, text: str, blocking: bool, cache_only: bool = False) -> str:
        if str == "":
            raise ValueError("No text provided")

        self.logger.info("Generating audio...")
        if not os.path.isdir(CACHEDIR):
            os.mkdir(CACHEDIR)

        file = os.path.join(
            CACHEDIR,
            self.speech_provider.value
            + self.speech_voice
            + self.completion_persona
            + hashlib.md5(text.encode()).hexdigest()
            + ".mp3",
        )
        try:
            if self.speech_provider == "elevenlabs":
                audio = self.eleven_client["client"].generate(
                    text=text, voice=self.speech_voice
                )
                eleven_save(audio=audio, filename=file)
            else:
                sp = gTTS(text=text, lang="en", slow=False)
                sp.save(file)
                audio_bytes = BytesIO()
                sp.write_to_fp(audio_bytes)

            if self.speaker_client is not None:
                # Get the actual bytes from BytesIO
                audio_data = audio_bytes.getvalue()

                # Resample MP3 to 48kHz for USB speaker compatibility
                try:
                    # Load MP3 audio data to get duration and audio properties
                    audio_segment = AudioSegment.from_file(BytesIO(audio_data), format="mp3")
                    duration_seconds = len(audio_segment) / 1000.0  # Convert ms to seconds

                    # Create AudioInfo for MP3 format with actual properties
                    # This tells the audio system the format and properties of the audio
                    audio_info = AudioInfo(
                        codec="mp3",
                        sample_rate_hz=audio_segment.frame_rate,
                        num_channels=audio_segment.channels
                    )

                    self.is_playing_audio = True
                    await self.speaker_client.play(
                        audio_data,
                        audio_info
                    )
                    if blocking:
                        await asyncio.sleep(duration_seconds)
                        self.is_playing_audio = False
                    else:
                        # If not blocking, schedule state reset after duration
                        asyncio.create_task(self._reset_playing_state(duration_seconds))
                    self.logger.info("Played audio...")

                except Exception as e:
                    print(f"Error in speaker play(): {e}")
                    self.logger.error(f"speaker client play error: {e}")
                    # # Fallback to original audio with best guess parameters
                    # try:
                    #     await self.microphone_client.play(audio_data, "mp3", 44100, 2)
                    #     print("here played audio")
            else:
                  # Fallback to pygame mixer if no speaker_client
                if not cache_only:
                        mixer.music.load(file)
                        self.logger.debug("Playing audio...")
                        mixer.music.play()  # Play it

                        if blocking:
                            while mixer.music.get_busy():
                                pygame.time.Clock().tick()

        except RuntimeError as err:
                self.logger.info("error")
                self.logger.error(err)
                raise ValueError("say() speech failure")


        return text

    async def listen_trigger(self, type: str) -> str:
        if type == "":
            raise ValueError("No trigger type provided")
        if type in ["command", "completion", "say"]:
            self.active_trigger_type = type
            self.trigger_active = True
            if self.should_listen:
                # close and re-open listener so any in-progress speech is not captured
                if rec_state.listen_closer is not None:
                    rec_state.listen_closer(True)

            # Use microphone_client if available, otherwise use pygame microphone
            if self.microphone_client is not None:
                rec_state.listen_closer = self.audio_listen_in_background(self.listen_callback)
            elif rec_state.rec is not None and rec_state.mic is not None:
                rec_state.listen_closer = rec_state.rec.listen_in_background(
                    source=rec_state.mic,
                    phrase_time_limit=self.listen_phrase_time_limit,
                    callback=self.listen_callback,
                )
        else:
            raise ValueError("Invalid trigger type provided")

        return "OK"

    async def is_speaking(self) -> bool:
        if self.speaker_client is not None:
            return self.is_playing_audio
        else:
            return mixer.music.get_busy()

    async def _reset_playing_state(self, delay: float):
        """Helper to reset playing state after delay"""
        await asyncio.sleep(delay)
        self.is_playing_audio = False

    # Background listening using audio client
    def audio_listen_in_background(self, callback):
        rec_state.audio_stop_event = asyncio.Event()

        async def listen_loop():
            audio_stream = None
            print("IN LISTEN LOOP")
            try:
                print(f"Calling get_audio with microphone_client: {self.microphone_client}")
                audio_stream = await self.microphone_client.get_audio("pcm16", 0, 0)
                print(f"Got audio_stream: {audio_stream}, type: {type(audio_stream)}")
            except Exception as e:
                print(f"ERROR getting audio stream: {e}")
                import traceback
                traceback.print_exc()
                return

            buffer = bytearray()
            speech_buffer = bytearray()

            # Create WebRTC VAD once (aggressiveness 0-3, where 2 is balanced)
            vad = webrtcvad.Vad(3)

            # WebRTC VAD requires specific frame sizes: 10, 20, or 30ms
            frame_duration = 20  # ms

            is_speech = False
            silence_frames = 0
            max_silence_frames = 30  # 30 frames * 20ms = 600ms of silence to end

            try:
                async for resp in audio_stream:
                    if rec_state.audio_stop_event.is_set():
                        print("stop event set")
                        break
                    sample_rate = resp.audio.audio_info.sample_rate_hz

                    # WebRTC VAD only supports specific sample rates
                    if sample_rate not in [8000, 16000, 32000, 48000]:
                        print(f"ERROR: Invalid sample rate {sample_rate} Hz for WebRTC VAD.")
                        print(f"Supported rates: 8000, 16000, 32000, 48000 Hz")
                        continue

                    buffer.extend(resp.audio.audio_data)
                    frame_size = int(sample_rate * frame_duration / 1000) * 2  # bytes

                    # Process frames of fixed size
                    while len(buffer) >= frame_size:
                        frame = bytes(buffer[:frame_size])
                        buffer = buffer[frame_size:]

                        try:
                            # Detect speech in this frame
                            speech_detected = vad.is_speech(frame, sample_rate)

                            if speech_detected:
                                is_speech = True
                                silence_frames = 0
                                speech_buffer.extend(frame)
                            elif is_speech:
                                # We were in speech, now silence
                                silence_frames += 1
                                speech_buffer.extend(frame)

                                if silence_frames >= max_silence_frames:
                                    print("End of speech detected, running audio to text")
                                    # Check if STT is already running
                                    if not rec_state.stt_in_progress:
                                        rec_state.stt_in_progress = True
                                        # End of speech - run callback in executor to avoid blocking
                                        audio_data = sr.AudioData(bytes(speech_buffer), sample_rate, 2)
                                        asyncio.get_event_loop().run_in_executor(None, callback, rec_state.rec, audio_data)
                                    else:
                                        print("STT already in progress, skipping this audio")
                                    speech_buffer.clear()
                                    is_speech = False
                                    silence_frames = 0
                                    break
                        except Exception as e:
                            print(f"VAD error: {e}")
            except asyncio.CancelledError:
                print("Listen loop cancelled cleanly")
            except Exception as e:
                print(f"FATAL ERROR in listen_loop: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Clean up audio stream
                if audio_stream is not None:
                    try:
                        await audio_stream.aclose()
                        print("Audio stream closed")
                    except Exception as e:
                        print(f"Error closing audio stream: {e}")

        # Create task in the event loop
        rec_state.audio_listen_task = asyncio.create_task(listen_loop())

        # Return stop function
        def stop_listening(wait_for_stop=True):
            print("stop_listening called")
            if rec_state.audio_stop_event:
                rec_state.audio_stop_event.set()
            if rec_state.audio_listen_task and not rec_state.audio_listen_task.done():
                rec_state.audio_listen_task.cancel()
                if wait_for_stop:
                    # Schedule cleanup of task
                    async def cleanup():
                        try:
                            await rec_state.audio_listen_task
                        except asyncio.CancelledError:
                            pass
                        # Clear the stop event for next use
                        if rec_state.audio_stop_event:
                            rec_state.audio_stop_event.clear()
                        print("Cleanup complete")
                    asyncio.create_task(cleanup())
        return stop_listening


    async def completion(
        self, text: str, blocking: bool, cache_only: bool = False
    ) -> str:
        if text == "":
            raise ValueError("No text provided")
        if self.completion_provider_org == "" or self.completion_provider_key == "":
            raise ValueError(
                "completion_provider_org or completion_provider_key missing"
            )

        completion = ""
        file = os.path.join(
            CACHEDIR,
            self.speech_provider.value
            + self.completion_persona
            + hashlib.md5(text.encode()).hexdigest()
            + ".txt",
        )
        if not cache_only and (self.cache_ahead_completions):
            self.logger.debug("Will try to read completion from cache")
            if os.path.isfile(file):
                self.logger.debug("Cache file exists")
                with open(file) as f:
                    completion = f.read()
                self.logger.debug(completion)

            # now cache next one
            asyncio.ensure_future(self.completion(text, blocking, True))

        if completion == "":
            self.logger.debug("Getting completion...")
            if self.completion_persona != "":
                text = "As " + self.completion_persona + " respond to '" + text + "'"
            completion = openai.chat.completions.create(
                model=self.completion_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": text}],
            )
            completion = completion.choices[0].message.content
            completion = re.sub("[^0-9a-zA-Z.!?,:'/ ]+", "", completion).lower()
            completion = completion.replace("as an ai language model", "")
            self.logger.debug("Got completion...")

        if cache_only:
            with open(file, "w") as f:
                f.write(completion)
            asyncio.ensure_future(self.say(completion, blocking, True))
        else:
            await self.say(completion, blocking)
        return completion

    async def get_commands(self, number: int = -1) -> list:
        # If number is -1 or not specified, return all commands
        if number == -1:
            to_return = self.command_list.copy()
            self.logger.debug("will get all commands from command list: " + str(to_return))
            self.command_list.clear()
        else:
            self.logger.debug("will get " + str(number) + " commands from command list")
            to_return = self.command_list[0:number]
            self.logger.debug("to return from command_list: " + str(to_return))
            del self.command_list[0:number]
        return to_return

    async def listen(self) -> str:
        print("LISTENING")
        if self.microphone_client is not None:
            audio_stream = await self.microphone_client.get_audio("pcm16", 0, 0)
            buffer = bytearray()

            # dB threshold for silence - a lower number = less silence will be detected. speech is -30 to -40 db.
            # for noisy environments, adjust closer to zero
            silence_threshold = -40
            min_silence_len = 1000
            should_stop = False
            async for resp in audio_stream:
                buffer.extend(resp.audio.audio_data)
                sample_rate = resp.audio.audio_info.sample_rate_hz
                num_channels = resp.audio.audio_info.num_channels
                sample_width = 2    # 2 bytes for 16-bit

                audio_segment = AudioSegment(
                    data=buffer,
                    sample_width=sample_width,
                    frame_rate=sample_rate,
                    channels=num_channels
                )

                # Check if we have enough audio to analyze (at least 3 seconds)
                if len(audio_segment) >= 3000:  # 3 seconds in ms
                      # Detect silence in the last portion
                      recent_audio = audio_segment[-3000:]  # Last 3 seconds
                      silence_ranges = silence.detect_silence(
                          recent_audio,
                          min_silence_len=min_silence_len,
                          silence_thresh=silence_threshold
                      )
                      # Calculate min dBFS manually
                      audio_array = numpy.array(recent_audio.get_array_of_samples())
                    #   min_sample = numpy.min(numpy.abs(audio_array))
                    #   print(f"Audio segment length: {len(recent_audio)}ms, Max dBFS: {recent_audio.max_dBFS:.1f}, average dBFS: {recent_audio.dBFS:.1f}")
                    #   print(f"Silence threshold: {silence_threshold} dB, Min silence len: {min_silence_len} ms")
                    #   print(f"Silence ranges found: {len(silence_ranges)}")
                      if silence_ranges:
                          print(f"Silence ranges: {silence_ranges[:3]}")  # Show first 3 ranges
                    # If recent audio is mostly silence, stop recording
                      if silence_ranges:
                          # Check if there's a silience period that's long enough
                          for silence_start, silence_end in silence_ranges:
                              silence_duration = silence_end - silence_start
                              if silence_duration >= min_silence_len:
                                  print("Silence detected, stopping recording")
                                  should_stop = True
                                  break
                if should_stop:
                    break
            if buffer:
              audio_data = sr.AudioData(bytes(buffer), sample_rate, sample_width)
              return await self.convert_audio_to_text(audio_data)

        elif rec_state.rec is not None and rec_state.mic is not None:
            with rec_state.mic as source:
                audio = rec_state.rec.listen(source)
            return await self.convert_audio_to_text(audio)


        self.logger.debug("Nothing to listen to")
        return ""

    async def to_text(self, speech: bytes, format: str = "mp3"):
        if self.stt is not None:
            self.logger.info("using stt provider")
            return await self.stt.to_text(speech, format)

        self.logger.info("using google stt")
        if rec_state.rec is not None:
            self.logger.info("rec_state.rec is not None")

            # Use temporary file for speech_recognition
            import tempfile
            import os

            try:
                # Create temporary file with proper extension
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}") as temp_file:
                    temp_file.write(speech)
                    temp_file_path = temp_file.name
                try:
                    # Convert to WAV if needed
                    if format != "wav":
                        self.logger.info(f"Converting {format} to WAV")
                        sound = AudioSegment.from_file(temp_file_path, format=format)
                        wav_path = temp_file_path.replace(f".{format}", ".wav")
                        sound.export(wav_path, format="wav")
                        os.unlink(temp_file_path)  # Remove original file
                        temp_file_path = wav_path

                    # Use AudioData.from_file() to create AudioData directly from file
                    audio = sr.AudioData.from_file(temp_file_path)
                    self.logger.info(f"Created AudioData from file: {len(audio.frame_data)} bytes, {audio.sample_rate}Hz")

                    return await self.convert_audio_to_text(audio)

                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass

            except Exception as e:
                self.logger.error(f"Error processing audio: {e}")
                return ""

        return ""

    async def to_speech(self, text):
        if self.speech_provider == "elevenlabs":
            audio = self.eleven_client["client"].generate(
                text=text, voice=self.speech_voice
            )
            return audio
        else:
            mp3_fp = BytesIO()
            sp = gTTS(text=text, lang="en", slow=False)
            sp.write_to_fp(mp3_fp)
            return mp3_fp.getvalue()

    def vosk_vad_callback(self, text: str):
        """Callback for Vosk VAD when speech is detected"""
        self.logger.info(f"Vosk VAD detected speech: '{text}'")

        if not self.main_loop or not self.main_loop.is_running():
            self.logger.error("Main event loop is not available for Vosk VAD task.")
            return

        # Process the detected speech similar to the regular callback
        if text != "":
            if (
                self.should_listen and re.search(".*" + self.listen_trigger_say, text)
            ) or (self.trigger_active and self.active_trigger_type == "say"):
                self.trigger_active = False
                to_say = re.sub(".*" + self.listen_trigger_say + r"\s+", "", text)
                asyncio.run_coroutine_threadsafe(
                    self.say(to_say, blocking=False), self.main_loop
                )
            elif (
                self.should_listen
                and re.search(".*" + self.listen_trigger_completion, text)
            ) or (self.trigger_active and self.active_trigger_type == "completion"):
                self.trigger_active = False
                to_say = re.sub(
                    ".*" + self.listen_trigger_completion + r"\s+", "", text
                )
                asyncio.run_coroutine_threadsafe(
                    self.completion(to_say, blocking=False), self.main_loop
                )
            elif (
                self.should_listen
                and re.search(".*" + self.listen_trigger_command, text)
            ) or (self.trigger_active and self.active_trigger_type == "command"):
                self.trigger_active = False
                command = re.sub(".*" + self.listen_trigger_command + r"\s+", "", text)
                self.command_list.insert(0, command)
                self.logger.debug("added to command_list: '" + command + "'")
                del self.command_list[self.listen_command_buffer_length :]

    def vosk_vad_thread(self):
        """Vosk VAD thread for voice activity detection"""
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8000
            )

            rec_state.vosk_stream = stream

            # Track phrase timing for Vosk VAD
            phrase_start_time = None
            phrase_time_limit = self.listen_phrase_time_limit

            while not rec_state.vosk_stop_event.is_set():
                try:
                    data = stream.read(4000, exception_on_overflow=False)

                    # Check if we have speech activity
                    if rec_state.vosk_rec.AcceptWaveform(data):
                        result = json.loads(rec_state.vosk_rec.Result())
                        if result.get('text', '').strip():
                            # Speech detected
                            if phrase_start_time is None:
                                phrase_start_time = time.time()
                                self.logger.debug("Vosk VAD: Phrase started")

                            # Check phrase time limit
                            if phrase_time_limit and phrase_start_time:
                                elapsed_time = time.time() - phrase_start_time
                                if elapsed_time >= phrase_time_limit:
                                    self.logger.debug(f"Vosk VAD: Phrase time limit reached ({elapsed_time:.1f}s)")
                                    # Reset for next phrase
                                    phrase_start_time = None
                                    continue

                            self.vosk_vad_callback(result['text'])
                        else:
                            # No speech detected, reset phrase timing
                            if phrase_start_time is not None:
                                self.logger.debug("Vosk VAD: Phrase ended (no speech)")
                                phrase_start_time = None

                except Exception as e:
                    self.logger.error(f"Vosk VAD error: {e}")
                    break

        except Exception as e:
            self.logger.error(f"Vosk VAD thread error: {e}")
        finally:
            if rec_state.vosk_stream:
                rec_state.vosk_stream.close()
            if p:
                p.terminate()

    def vosk_vad_listen_in_background(self, callback):
        """VOSK VAD for microphone_client (AudioIn component)"""
        print("STARTING VOSK VAD WITH MICROPHONE CLIENT")
        rec_state.vosk_stop_event = asyncio.Event()

        async def vosk_listen_loop():
            print("IN VOSK LISTEN LOOP")
            try:
                print(f"Calling get_audio with microphone_client: {self.microphone_client}")
                audio_stream = await self.microphone_client.get_audio("pcm16", 0, 0)
                print(f"Got audio_stream for VOSK: {audio_stream}")
            except Exception as e:
                print(f"ERROR getting audio stream for VOSK: {e}")
                import traceback
                traceback.print_exc()
                return

            # Track phrase timing
            phrase_start_time = None
            phrase_time_limit = self.listen_phrase_time_limit

            async for resp in audio_stream:
                if rec_state.vosk_stop_event.is_set():
                    print("VOSK stop event set, breaking")
                    break

                try:
                    chunk = resp.audio
                    audio_data = chunk.audio_data

                    # VOSK expects 16kHz mono, check if resampling is needed
                    sample_rate = chunk.audio_info.sample_rate_hz

                    if sample_rate != 16000:
                        # Resample to 16kHz for VOSK
                        audio_segment = AudioSegment(
                            data=audio_data,
                            sample_width=2,  # 16-bit = 2 bytes
                            frame_rate=sample_rate,
                            channels=1
                        )
                        audio_segment = audio_segment.set_frame_rate(16000)
                        audio_data = audio_segment.raw_data

                    # Feed data to VOSK recognizer
                    if rec_state.vosk_rec.AcceptWaveform(audio_data):
                        result = json.loads(rec_state.vosk_rec.Result())
                        if result.get('text', '').strip():
                            # Speech detected
                            if phrase_start_time is None:
                                phrase_start_time = time.time()
                                self.logger.debug("Vosk VAD: Phrase started")

                            # Check phrase time limit
                            if phrase_time_limit and phrase_start_time:
                                elapsed_time = time.time() - phrase_start_time
                                if elapsed_time >= phrase_time_limit:
                                    self.logger.debug(f"Vosk VAD: Phrase time limit reached ({elapsed_time:.1f}s)")
                                    phrase_start_time = None
                                    continue

                            self.vosk_vad_callback(result['text'])
                        else:
                            # No speech detected, reset phrase timing
                            if phrase_start_time is not None:
                                self.logger.debug("Vosk VAD: Phrase ended (no speech)")
                                phrase_start_time = None

                except Exception as e:
                    print(f"Error processing VOSK audio chunk: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            print("VOSK listen loop ended")

        # Start the async loop in a task
        task = asyncio.create_task(vosk_listen_loop())

        def stopper(wait_for_stop=True):
            print("VOSK VAD STOPPER CALLED")
            rec_state.vosk_stop_event.set()
            if wait_for_stop:
                # Cancel the task
                task.cancel()

        return stopper

    def start_vosk_vad(self):
        """Start Vosk VAD if available"""
        if not VOSK_AVAILABLE:
            self.logger.warning("Vosk not available, falling back to speech_recognition VAD")
            return False

        try:
            # Try to load a small Vosk model for VAD
            # You can download models from https://alphacephei.com/vosk/models
            model_path = os.path.expanduser("~/vosk-model-small-en-us-0.15")
            if not os.path.exists(model_path):
                self.logger.info("Vosk model not found, attempting to download...")
                if self.download_vosk_model():
                    self.logger.info("Successfully downloaded Vosk model")
                else:
                    self.logger.warning("Failed to download Vosk model, falling back to speech_recognition VAD")
                    return False

            rec_state.vosk_model = vosk.Model(model_path)
            rec_state.vosk_rec = vosk.KaldiRecognizer(rec_state.vosk_model, 16000)

            # Use microphone_client if available, otherwise use PyAudio
            if self.microphone_client is not None:
                self.logger.info("Starting Vosk VAD with microphone_client (AudioIn component)")
                rec_state.vosk_stop_event = asyncio.Event()
                # Start the async version for microphone_client
                rec_state.vosk_closer = self.vosk_vad_listen_in_background(self.vosk_vad_callback)
            else:
                self.logger.info("Starting Vosk VAD with PyAudio")
                rec_state.vosk_stop_event = threading.Event()
                rec_state.vosk_thread = threading.Thread(target=self.vosk_vad_thread, daemon=True)
                rec_state.vosk_thread.start()

            self.logger.info("Started Vosk VAD for voice activity detection")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start Vosk VAD: {e}")
            return False

    def download_vosk_model(self):
        """Download Vosk model automatically"""
        try:
            import urllib.request
            import zipfile
            import shutil

            model_name = "vosk-model-small-en-us-0.15"
            model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
            model_path = os.path.expanduser(f"~/{model_name}")
            zip_path = os.path.expanduser(f"~/{model_name}.zip")

            self.logger.info(f"Downloading Vosk model from {model_url}")

            # Download the model
            urllib.request.urlretrieve(model_url, zip_path)

            # Extract the model
            self.logger.info("Extracting Vosk model...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.expanduser("~/"))

            # Clean up zip file
            os.remove(zip_path)

            # Verify the model was extracted correctly
            if os.path.exists(model_path):
                self.logger.info(f"Vosk model downloaded successfully to {model_path}")
                return True
            else:
                self.logger.error("Failed to extract Vosk model")
                return False

        except Exception as e:
            self.logger.error(f"Failed to download Vosk model: {e}")
            return False

    def stop_vosk_vad(self):
        """Stop Vosk VAD (handles both microphone_client and PyAudio versions)"""
        # Stop async version (microphone_client)
        if rec_state.vosk_closer is not None:
            rec_state.vosk_closer(wait_for_stop=True)
            rec_state.vosk_closer = None

        # Stop threaded version (PyAudio)
        if rec_state.vosk_stop_event:
            rec_state.vosk_stop_event.set()
        if rec_state.vosk_thread and rec_state.vosk_thread.is_alive():
            rec_state.vosk_thread.join(timeout=1)
        if rec_state.vosk_stream:
            rec_state.vosk_stream.close()

    def listen_callback(self, recognizer, audio):
        self.logger.info("speechio heard audio")

        if not self.main_loop or not self.main_loop.is_running():
            self.logger.error("Main event loop is not available for STT task.")
            return

        future = asyncio.run_coroutine_threadsafe(
            self.convert_audio_to_text(audio), self.main_loop
        )
        try:
            heard = future.result(timeout=15)
        except Exception as e:
            self.logger.error(f"STT task failed: {e}")
            rec_state.stt_in_progress = False
            return

        self.logger.info("speechio heard " + heard)

        if heard != "":
            if (
                self.should_listen and re.search(".*" + self.listen_trigger_say, heard)
            ) or (self.trigger_active and self.active_trigger_type == "say"):
                self.trigger_active = False
                to_say = re.sub(".*" + self.listen_trigger_say + r"\s+", "", heard)
                asyncio.run_coroutine_threadsafe(
                    self.say(to_say, blocking=False), self.main_loop
                )
            elif (
                self.should_listen
                and re.search(".*" + self.listen_trigger_completion, heard)
            ) or (self.trigger_active and self.active_trigger_type == "completion"):
                self.trigger_active = False
                to_say = re.sub(
                    ".*" + self.listen_trigger_completion + r"\s+", "", heard
                )
                asyncio.run_coroutine_threadsafe(
                    self.completion(to_say, blocking=False), self.main_loop
                )
            elif (
                self.should_listen
                and re.search(".*" + self.listen_trigger_command, heard)
            ) or (self.trigger_active and self.active_trigger_type == "command"):
                self.trigger_active = False
                command = re.sub(".*" + self.listen_trigger_command + r"\s+", "", heard)
                self.command_list.insert(0, command)
                self.logger.debug("added to command_list: '" + command + "'")
                del self.command_list[self.listen_command_buffer_length :]
            if not self.should_listen:
                # stop listening if not in background listening mode
                self.logger.debug("will close background listener")
                if rec_state.listen_closer is not None:
                    rec_state.listen_closer()

    async def convert_audio_to_text(self, audio: sr.AudioData) -> str:
        try:
            if self.stt is not None:
                self.logger.info("getting wav data")
                audio_data = audio.get_wav_data()
                self.logger.info("using stt provider")
                return await self.stt.to_text(audio_data, format="wav")

            heard = ""

            try:
                self.logger.info("will convert audio to text")
                # for testing purposes, we're just using the default API key
                # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
                # instead of `r.recognize_google(audio)`
                transcript = rec_state.rec.recognize_google(audio, show_all=True)
                self.logger.info("transcript: " + str(transcript))
                if type(transcript) is dict and transcript.get("alternative"):
                    heard = transcript["alternative"][0]["transcript"]
                    self.logger.info("heard: " + heard)
            except sr.UnknownValueError:
                self.logger.warning("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                self.logger.warning(
                    "Could not request results from Google Speech Recognition service; {0}".format(
                        e
                    )
                )
            return heard
        finally:
            # Always reset the STT flag when done
            rec_state.stt_in_progress = False
            print("STT processing complete, ready for next request")

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        try:
            self.main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.main_loop = None
            self.logger.error("Could not get running event loop in reconfigure.")

        attrs = struct_to_dict(config.attributes)
        self.speech_provider = SpeechProvider[
            str(attrs.get("speech_provider", "google"))
        ]
        self.speech_provider_key = str(attrs.get("speech_provider_key", ""))
        self.speech_voice = str(attrs.get("speech_voice", "Josh"))
        self.completion_provider = CompletionProvider[
            str(attrs.get("completion_provider", "openai"))
        ]
        self.completion_model = str(attrs.get("completion_model", "gpt-4o"))
        self.completion_provider_org = str(attrs.get("completion_provider_org", ""))
        self.completion_provider_key = str(attrs.get("completion_provider_key", ""))
        if self.completion_provider == "openai":
            openai.api_key = self.completion_provider_key
            openai.organization = self.completion_provider_org
        self.completion_persona = str(attrs.get("completion_persona", ""))
        self.stt_provider = str(attrs.get("stt_provider", "google"))
        self.should_listen = bool(attrs.get("listen", False))
        self.listen_phrase_time_limit = attrs.get("listen_phrase_time_limit", None)
        self.mic_device_name = str(attrs.get("mic_device_name", ""))
        self.listen_trigger_say = str(attrs.get("listen_trigger_say", "robot say"))
        self.listen_trigger_completion = str(
            attrs.get("listen_trigger_completion", "hey robot")
        )
        self.listen_trigger_command = str(
            attrs.get("listen_trigger_command", "robot can you")
        )
        self.listen_command_buffer_length = int(
            attrs.get("listen_command_buffer_length", 10)
        )
        self.cache_ahead_completions = bool(attrs.get("cache_ahead_completions", False))
        self.disable_mic = bool(attrs.get("disable_mic", False))
        self.disable_audioout = bool(attrs.get("disable_audioout", False))
        self.use_vosk_vad = bool(attrs.get("use_vosk_vad", False))  # New option for Vosk VAD
        self.command_list = []
        self.trigger_active = False
        self.active_trigger_type = ""
        self.stt = None
        self.microphone = str(attrs.get("microphone_name", ""))
        self.speaker = str(attrs.get("speaker_name", ""))

        if (
            self.speech_provider == SpeechProvider.elevenlabs
            and self.speech_provider_key != ""
        ):
            self.eleven_client["client"] = ElevenLabs(api_key=self.speech_provider_key)
        else:
            self.speech_provider = SpeechProvider.google

        if self.stt_provider != "google":
            stt = dependencies[SpeechService.get_resource_name(self.stt_provider)]
            self.stt = cast(SpeechService, stt)

        # Set up microphone client if microphone_name is specified
        if self.microphone != "":
            mic = dependencies[AudioIn.get_resource_name(self.microphone)]
            self.microphone_client = cast(AudioIn, mic)
        else:
            self.microphone_client = None


        if self.speaker != "":
            speaker = dependencies[AudioOut.get_resource_name(self.speaker)]
            self.speaker_client = cast(AudioOut, speaker)
        else:
            self.speaker_client = None

        # Track audio playing state for speaker_client
        self.is_playing_audio = False

        if not self.disable_audioout and self.speaker_client is not None:
            if not mixer.get_init():
                try:
                    mixer.init(buffer=1024)
                except Exception as err:
                    try:
                        # try with pulse server
                        os.environ["PULSE_SERVER"] = "/run/user/1000/pulse/native"
                        mixer.init(buffer=1024)
                    except Exception as e:
                        self.logger.warning(f"Failed to initialize pygame mixer: {e}")
                        self.logger.warning("Audio playback via pygame will not be available")
        else:
            if mixer.get_init():
                mixer.quit()

        rec_state.rec = sr.Recognizer()

        # Only set up pygame microphone if microphone is not configured
        if not self.disable_mic and self.microphone == "":
            # Stop any existing VAD
            if rec_state.listen_closer is not None:
                rec_state.listen_closer(True)
            self.stop_vosk_vad()

            # Set up speech recognition with pygame microphone
            rec_state.rec.dynamic_energy_threshold = True

            try:
                mics = sr.Microphone.list_microphone_names()

                if self.mic_device_name != "":
                    rec_state.mic = sr.Microphone(mics.index(self.mic_device_name))
                else:
                    rec_state.mic = sr.Microphone()

                if rec_state.mic is not None:
                    with rec_state.mic as source:
                        rec_state.rec.adjust_for_ambient_noise(source, 2)
                else:
                    self.logger.warning("Microphone is None")
            except Exception as e:
                self.logger.warning(f"Failed to initialize pygame microphone: {e}")
                rec_state.mic = None

        # Set up background listening
        if self.should_listen:
            self.logger.info("Will listen in background")

            # Try Vosk VAD first if enabled
            if self.use_vosk_vad and self.start_vosk_vad():
                self.logger.info("Using Vosk VAD for voice activity detection")
            else:
                # Fall back to regular speech recognition VAD
                if self.microphone_client is not None:
                    # Use microphone_client for listening
                    self.logger.info("Using audioin component for microphone input")
                    rec_state.listen_closer = self.audio_listen_in_background(self.listen_callback)
                elif rec_state.mic is not None:
                    # Use pygame microphone for listening
                    self.logger.info("Using pygame microphone for input")
                    rec_state.listen_closer = rec_state.rec.listen_in_background(
                        source=rec_state.mic,
                        phrase_time_limit=self.listen_phrase_time_limit,
                        callback=self.listen_callback,
                    )
                else:
                    self.logger.warning("No microphone available for background listening")

