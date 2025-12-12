"""
Vosk VAD Handler

Handles all Vosk voice activity detection functionality.
"""

import os
import json
import time
import asyncio
import threading
import re
from typing import Optional, Callable
import pyaudio
from pydub import AudioSegment

try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False


class VoskState:
    """State container for Vosk VAD"""
    def __init__(self):
        self.model: Optional[object] = None
        self.rec: Optional[object] = None
        self.stream: Optional[object] = None  # PyAudio stream
        self.async_stop_event: Optional[asyncio.Event] = None
        self.closer: Optional[Callable] = None


class VoskHandler:
    """Handles Vosk VAD functionality for both Viam and PyAudio"""

    def __init__(
        self,
        logger,
        callback: Callable,
        main_loop: Optional[asyncio.AbstractEventLoop] = None,
        phrase_time_limit: Optional[float] = None,
        microphone_client=None
    ):
        """
        Args:
            logger: Logger instance
            callback: Callback function to handle detected speech
            main_loop: Event loop for async operations
            phrase_time_limit: Maximum phrase duration in seconds
            microphone_client: Optional Viam AudioIn client
        """
        self.logger = logger
        self.callback = callback
        self.main_loop = main_loop
        self.phrase_time_limit = phrase_time_limit
        self.microphone_client = microphone_client
        self.state = VoskState()

    def start(self) -> bool:
        """Start Vosk VAD if available"""
        if not VOSK_AVAILABLE:
            self.logger.warning("Vosk not available, cannot start")
            return False

        try:
            # Load Vosk model
            model_path = os.path.expanduser("~/vosk-model-small-en-us-0.15")
            if not os.path.exists(model_path):
                self.logger.debug("Vosk model not found, attempting to download...")
                if not self._download_model():
                    self.logger.warning("Failed to download Vosk model")
                    return False

            self.state.model = vosk.Model(model_path)
            self.state.rec = vosk.KaldiRecognizer(self.state.model, 16000)

            # Unified async listener for both Viam and PyAudio
            self.state.closer = self._listen_in_background_async()

            source = "Viam AudioIn" if self.microphone_client else "PyAudio"
            self.logger.debug(f"Started Vosk VAD ({source})")

            return True

        except Exception as e:
            self.logger.error(f"Failed to start Vosk VAD: {e}")
            return False

    def stop(self):
        """Stop Vosk VAD"""
        if self.state.closer is not None:
            self.state.closer(wait_for_stop=True)
            self.state.closer = None

        # Clean up stream if it exists
        if self.state.stream:
            try:
                self.state.stream.close()
            except:
                self.logger.error("failed to close VOSK VAD stream")

    def _listen_in_background_async(self):
        """Unified Vosk VAD for both Viam and PyAudio"""
        self.logger.debug("Vosk VAD: starting background listener")
        self.state.async_stop_event = asyncio.Event()

        async def listen_loop():
            phrase_start_time = None

            try:
                if self.microphone_client is not None:
                    audio_stream = await self.microphone_client.get_audio("pcm16", 0, 0)
                    async for resp in audio_stream:
                        if self.state.async_stop_event.is_set():
                            break

                        audio_data = resp.audio.audio_data
                        sample_rate = resp.audio.audio_info.sample_rate_hz

                        # Resample to 16kHz if needed
                        if sample_rate != 16000:
                            audio_segment = AudioSegment(
                                data=audio_data,
                                sample_width=2,
                                frame_rate=sample_rate,
                                channels=1
                            )
                            audio_segment = audio_segment.set_frame_rate(16000)
                            audio_data = audio_segment.raw_data

                        phrase_start_time = self._process_audio(audio_data, phrase_start_time)

                else:
                    # use legacy pyaudio microphone
                    p = pyaudio.PyAudio()
                    stream = p.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=8000,
                    )
                    self.state.stream = stream

                    try:
                        loop = asyncio.get_event_loop()
                        while not self.state.async_stop_event.is_set():
                            # Read in executor to avoid blocking
                            audio_data = await loop.run_in_executor(
                                None,
                                stream.read,
                                4000,
                                False  # exception_on_overflow
                            )
                            phrase_start_time = self._process_audio(audio_data, phrase_start_time)
                    finally:
                        stream.close()
                        p.terminate()

            except Exception as e:
                self.logger.error(f"Vosk listen error: {e}")

        task = asyncio.create_task(listen_loop())

        def stopper(wait_for_stop=True):
            self.state.async_stop_event.set()
            if wait_for_stop:
                task.cancel()

        return stopper

    def _process_audio(self, audio_data: bytes, phrase_start_time: Optional[float]) -> Optional[float]:
        """Process audio chunk through Vosk """
        try:
            if self.state.rec.AcceptWaveform(audio_data):
                result = json.loads(self.state.rec.Result())
                text = result.get("text", "").strip()

                if text:
                    # Speech detected
                    if phrase_start_time is None:
                        phrase_start_time = time.time()
                        self.logger.debug("Vosk VAD: Phrase started")

                    # Check phrase time limit
                    if self.phrase_time_limit and phrase_start_time:
                        elapsed = time.time() - phrase_start_time
                        if elapsed >= self.phrase_time_limit:
                            self.logger.debug(
                                f"Vosk VAD: Phrase time limit reached ({elapsed:.1f}s)"
                            )
                            return None

                    self.callback(text)
                else:
                    # No speech, reset timing
                    if phrase_start_time is not None:
                        self.logger.debug("Vosk VAD: Phrase ended (no speech)")
                        phrase_start_time = None

        except Exception as e:
            self.logger.error(f"Vosk processing error: {e}")

        return phrase_start_time

    def _download_model(self) -> bool:
        """Download Vosk model automatically"""
        try:
            import urllib.request
            import zipfile

            model_name = "vosk-model-small-en-us-0.15"
            model_url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
            model_path = os.path.expanduser(f"~/{model_name}")
            zip_path = os.path.expanduser(f"~/{model_name}.zip")

            self.logger.debug(f"Downloading Vosk model from {model_url}")

            # Download
            urllib.request.urlretrieve(model_url, zip_path)

            # Extract
            self.logger.debug("Extracting Vosk model...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(os.path.expanduser("~/"))

            # Clean up
            os.remove(zip_path)

            if os.path.exists(model_path):
                self.logger.debug(f"Vosk model downloaded to {model_path}")
                return True
            else:
                self.logger.error("Failed to extract Vosk model")
                return False

        except Exception as e:
            self.logger.error(f"Failed to download Vosk model: {e}")
            return False
