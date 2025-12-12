"""
Viam AudioIn Source adapter for hearken Listener.

Allows hearken to use Viam's AudioIn component as an audio source.
"""

import asyncio
from typing import Optional
from queue import Queue, Full, Empty
from viam.components.audio_in import AudioIn


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
        # Use a queue with limited size to prevent unbounded memory growth
        self._queue: Queue = Queue(maxsize=50)  # ~50 chunks buffer

    def open(self) -> None:
        """Open the audio source for reading."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError("ViamAudioInSource.open() requires a running event loop")

        self._stop_event = asyncio.Event()
        # Start async streaming in background
        asyncio.create_task(self._start_stream())

    def close(self) -> None:
        """Close the audio source and release resources."""
        if self._stop_event:
            self._stop_event.set()
        # The audio stream is an async iterator that will clean up when the task ends
        self._audio_stream = None

    async def _start_stream(self):
        """Internal method to start streaming audio from Viam component."""
        try:
            if self.logger:
                self.logger.debug("Starting Viam audio stream...")

            # Get actual properties from the microphone
            properties = await self.microphone_client.get_properties()
            self._sample_rate = properties.sample_rate_hz

            # Start the streaming task - it will call get_audio() when ready
            asyncio.create_task(self._stream_audio())
            if self.logger:
                self.logger.debug("Viam audio stream task created")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to start Viam audio stream: {e}")

    def read(self, num_samples: int) -> bytes:
        """Read audio samples from the source.

        Args:
            num_samples: Number of samples to read

        Returns:
            Audio data as bytes
        """
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
            # Get audio stream - this starts the capture
            self._audio_stream = await self.microphone_client.get_audio("pcm16", 0, 0)
            if self.logger:
                self.logger.debug("Audio stream acquired, starting consumption loop")

            async for resp in self._audio_stream:
                if self._stop_event and self._stop_event.is_set():
                    break

                audio_data = resp.audio.audio_data
                chunk_count += 1

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
