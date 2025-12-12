"""
Viam AudioIn Source adapter for hearken Listener.

Allows hearken to use Viam's AudioIn component as an audio source.
"""

import asyncio
import threading
import time
from typing import Optional
from queue import Queue, Full, Empty
from viam.components.audio_in import AudioIn
from pydub import AudioSegment


class ViamAudioInSource:
    """Interface to use Viam AudioIn component with hearken Listener.

    This adapter bridges Viam's async AudioIn streaming API with hearken's
    synchronous audio source interface.
    """

    def __init__(
        self,
        microphone_client: AudioIn,
        sample_rate: int = 0,
        sample_width: int = 2,  # 2 bytes for 16-bit PCM
        chunk_size: int = 1024,
        logger=None
    ):
        """
        Args:
            microphone_client: Viam AudioIn component
            sample_rate: Target sample rate in Hz
            sample_width: Bytes per sample (2 for 16-bit PCM)
            chunk_size: Size of audio chunks in samples
            logger: Optional logger instance
        """
        self.microphone_client = microphone_client
        self._sample_rate = sample_rate
        self._sample_width = sample_width
        self.chunk_size = chunk_size
        self.logger = logger

        self._audio_stream = None
        self._stop_event: Optional[asyncio.Event] = None
        self._stream_ready = threading.Event()  # Signals when audio stream is consuming
        self._stream_ready.clear()  # Start in non-ready state
        self._stream_task: Optional[asyncio.Task] = None  # Track the stream task
        self._last_read_time: Optional[float] = None  # For rate limiting
        # Use a queue with limited size to prevent unbounded memory growth
        self._queue: Queue = Queue(maxsize=50)  # ~50 chunks buffer

    def open(self) -> None:
        """Open the audio source for reading."""
        # Don't open if already running
        if self._stream_task and not self._stream_task.done():
            if self.logger:
                self.logger.debug("Audio source already open, skipping")
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError("ViamAudioInSource.open() requires a running event loop")

        # Schedule the async setup in the event loop
        async def setup():
            self._stop_event = asyncio.Event()
            self._stream_task = asyncio.create_task(self._stream_audio())

        if self.logger:
            self.logger.debug("Scheduling audio stream setup (non-blocking)")
        asyncio.run_coroutine_threadsafe(setup(), loop)

    def close(self) -> None:
        """Close the audio source and release resources."""
        if self.logger:
            self.logger.debug("Closing audio source")

        # Signal the task to stop
        if self._stop_event:
            self._stop_event.set()

        # Cancel the stream task
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            if self.logger:
                self.logger.debug("Cancelled audio stream task")

        self._audio_stream = None
        self._stream_task = None

    def read(self, num_samples: int) -> bytes:
        """Read audio samples from the source.

        Args:
            num_samples: Number of samples to read

        Returns:
            Audio data as bytes
        """
        # Rate limit to simulate real-time audio capture
        if self._last_read_time and self._sample_rate > 0:
            expected_duration = num_samples / self._sample_rate
            elapsed = time.time() - self._last_read_time
            if elapsed < expected_duration:
                time.sleep(expected_duration - elapsed)

        self._last_read_time = time.time()

        chunk_size = num_samples * self._sample_width
        buffer = bytearray()

        # Read from queue until we have enough data
        while len(buffer) < chunk_size:
            try:
                # Try to get audio chunk from queue with timeout
                chunk = self._queue.get(timeout=0.1)
                buffer.extend(chunk)
            except:
                # Queue empty or timeout - return what we have
                break

        # Always return exactly the requested size
        if len(buffer) >= chunk_size:
            # Return exactly the requested amount
            result = bytes(buffer[:chunk_size])
            # Put remaining back in queue if any
            if len(buffer) > chunk_size:
                remaining = bytes(buffer[chunk_size:])
                try:
                    self._queue.put_nowait(remaining)
                except Full:
                    if self.logger:
                        self.logger.warning("Failed to return excess audio to queue")
            return result

        # Pad with silence (zeros) to match requested size if we don't have enough data
        silence_needed = chunk_size - len(buffer)
        buffer.extend(b'\x00' * silence_needed)
        return bytes(buffer)

    async def _stream_audio(self):
        """Continuously stream audio from Viam into the queue."""
        if self.logger:
            self.logger.debug("_stream_audio task started")
        chunk_count = 0
        try:
            # Get actual properties from the microphone
            properties = await self.microphone_client.get_properties()
            source_sample_rate = properties.sample_rate_hz
            if self.logger:
                self.logger.debug(f"Microphone sample rate: {source_sample_rate}Hz")

            # Target sample rate for VAD (WebRTC VAD requires 8/16/32 kHz)
            target_sample_rate = 16000
            needs_resampling = source_sample_rate != target_sample_rate

            if needs_resampling:
                if self.logger:
                    self.logger.debug(f"Will resample from {source_sample_rate}Hz to {target_sample_rate}Hz")

            # Expose target sample rate to hearken
            self._sample_rate = target_sample_rate

            # Get audio stream - this starts the capture
            self._audio_stream = await self.microphone_client.get_audio("pcm16", 0, 0)
            if self.logger:
                self.logger.debug("Audio stream acquired, starting consumption loop")

            self.logger.debug("setting stream ready")
            # Signal that we're ready to consume
            self._stream_ready.set()

            # Get event loop for running blocking operations
            loop = asyncio.get_event_loop()

            async for resp in self._audio_stream:
                if self._stop_event and self._stop_event.is_set():
                    break

                audio_data = resp.audio.audio_data
                chunk_count += 1

                # Resample in executor to avoid blocking event loop
                if needs_resampling:
                    def resample_chunk(data):
                        segment = AudioSegment(
                            data=data,
                            sample_width=self._sample_width,
                            frame_rate=source_sample_rate,
                            channels=1
                        )
                        segment = segment.set_frame_rate(target_sample_rate)
                        return segment.raw_data

                    audio_data = await loop.run_in_executor(None, resample_chunk, audio_data)

                # Put audio in queue with backpressure (non-blocking)
                try:
                    self._queue.put_nowait(audio_data)
                except Full:
                    # Queue full - drop this chunk (backpressure)
                    if self.logger:
                        self.logger.warning("Audio queue full, dropping chunk")
                    continue

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in audio streaming: {e}")
        finally:
            if self.logger:
                self.logger.debug(f"Audio streaming ended. Total chunks: {chunk_count}")

    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz (e.g., 16000)."""
        return self._sample_rate

    @property
    def sample_width(self) -> int:
        """Bytes per sample (e.g., 2 for 16-bit)."""
        return self._sample_width

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
