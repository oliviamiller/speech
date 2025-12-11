"""
Microphone Client Listener

Handles background audio listening from Viam AudioIn client with WebRTC VAD.
"""

import asyncio
import webrtcvad
import speech_recognition as sr
from typing import Callable, Optional
import numpy as np
from scipy import signal


class MicrophoneListenerState:
    """State container for microphone listener"""
    def __init__(self):
        self.audio_listen_task: Optional[asyncio.Task] = None
        self.audio_stop_event: Optional[asyncio.Event] = None


class MicrophoneListener:
    """Handles background audio listening from Viam AudioIn client"""

    def __init__(
        self,
        logger,
        microphone_client,
        callback: Callable,
        owner
    ):
        """
        Args:
            logger: Logger instance
            microphone_client: Viam AudioIn client
            callback: Callback function to handle detected speech
            recognizer: speech_recognition Recognizer instance
            owner: Owner instance with stt_in_progress attribute
        """
        self.logger = logger
        self.microphone_client = microphone_client
        self.callback = callback
        self.owner = owner
        self.state = MicrophoneListenerState()

    def resample_audio(self, audio_data: bytes, orig_rate: int, target_rate: int) -> bytes:
        """Resample audio data from orig_rate to target_rate.

        Args:
            audio_data: Raw PCM16 audio bytes
            orig_rate: Original sample rate in Hz
            target_rate: Target sample rate in Hz

        Returns:
            Resampled audio as bytes
        """
        # Convert bytes to numpy array (16-bit PCM = int16)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Calculate number of samples needed
        num_samples = int(len(audio_array) * target_rate / orig_rate)

        # Resample using scipy
        resampled = signal.resample(audio_array, num_samples)

        # Convert back to int16 and bytes
        resampled_int16 = resampled.astype(np.int16)
        return resampled_int16.tobytes()

    def start(self):
        """Start background listening with WebRTC VAD"""
        self.state.audio_stop_event = asyncio.Event()

        async def listen_loop():
            audio_stream = None
            try:
                audio_stream = await self.microphone_client.get_audio("pcm16", 0, 0)
            except Exception as e:
                self.logger.error(f"Failed to get audio stream: {e}")
                return

            buffer = bytearray()
            speech_buffer = bytearray()

            # Create WebRTC VAD once (aggressiveness 0-3)
            vad = webrtcvad.Vad(3)

            # WebRTC VAD requires specific frame sizes: 10, 20, or 30ms
            frame_duration = 20  # ms

            is_speech = False
            silence_frames = 0
            max_silence_frames = 30  # 30 frames * 20ms = 600ms of silence to end

            # Track the working sample rate (may be resampled)
            working_sample_rate = None

            try:
                async for resp in audio_stream:
                    if self.state.audio_stop_event.is_set():
                        self.logger.debug("stop event set")
                        break
                    sample_rate = resp.audio.audio_info.sample_rate_hz
                    audio_data = resp.audio.audio_data

                    # WebRTC VAD only supports specific sample rates
                    if sample_rate not in [8000, 16000, 32000, 48000]:
                        # Choose the nearest supported rate
                        if sample_rate < 12000:
                            target_rate = 16000
                        elif sample_rate < 24000:
                            target_rate = 16000
                        elif sample_rate < 40000:
                            target_rate = 32000
                        else:
                            target_rate = 48000

                        self.logger.warning(f"Resampling audio from {sample_rate} Hz to {target_rate} Hz for WebRTC VAD")
                        audio_data = self.resample_audio(audio_data, sample_rate, target_rate)
                        sample_rate = target_rate

                    # Set working rate if not set
                    if working_sample_rate is None:
                        working_sample_rate = sample_rate

                    buffer.extend(audio_data)
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
                                # We were in speech, now in silence
                                silence_frames += 1
                                speech_buffer.extend(frame)

                                if silence_frames >= max_silence_frames:
                                    self.logger.debug("End of speech detected, running speech to text")
                                    # Check if STT is already running
                                    if not self.owner.stt_in_progress:
                                        self.owner.stt_in_progress = True
                                        # End of speech - run callback in executor to avoid blocking
                                        audio_data = sr.AudioData(bytes(speech_buffer), working_sample_rate, 2)

                                        # Wrap callback to ensure flag is reset on error
                                        def safe_callback():
                                            try:
                                                self.callback(audio_data)
                                            except Exception as e:
                                                self.logger.error(f"STT callback error: {e}")
                                                self.owner.stt_in_progress = False

                                        asyncio.get_event_loop().run_in_executor(None, safe_callback)
                                    else:
                                        self.logger.debug("STT already in progress, skipping this audio")
                                    speech_buffer.clear()
                                    is_speech = False
                                    silence_frames = 0
                                    break
                        except Exception as e:
                            self.logger.error(f"VAD error: {e}")
            except asyncio.CancelledError:
                    self.logger.debug("asyncio cancelled")
            except Exception as e:
                self.logger.error(f"FATAL ERROR in listen_loop: {e}")
            finally:
                # Clean up audio stream
                if audio_stream is not None:
                    try:
                        await audio_stream.aclose()
                    except Exception as e:
                        self.logger.error(f"Error closing audio stream: {e}")

        # Create task in the event loop
        self.state.audio_listen_task = asyncio.create_task(listen_loop())

        # Return stop function
        def stop_listening(wait_for_stop=True):
            if self.state.audio_stop_event:
                self.state.audio_stop_event.set()
            if self.state.audio_listen_task and not self.state.audio_listen_task.done():
                self.state.audio_listen_task.cancel()
                if wait_for_stop:
                    # Schedule cleanup of task
                    async def cleanup():
                        try:
                            await self.state.audio_listen_task
                        except asyncio.CancelledError:
                            pass
                        # Clear the stop event for next use
                        if self.state.audio_stop_event:
                            self.state.audio_stop_event.clear()
                    asyncio.create_task(cleanup())
        return stop_listening

    def stop(self):
        """Stop background listening"""
        if self.state.audio_stop_event:
            self.state.audio_stop_event.set()
        if self.state.audio_listen_task and not self.state.audio_listen_task.done():
            self.state.audio_listen_task.cancel()
