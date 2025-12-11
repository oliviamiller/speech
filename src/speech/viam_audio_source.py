"""
Viam AudioIn Source adapter for hearken Listener.

Allows hearken to use Viam's AudioIn component as an audio source.
"""

import asyncio
from typing import Optional
from viam.components.audio_in import AudioIn


class ViamAudioInSource:
    """Adapter to use Viam AudioIn component with hearken Listener.

    This adapter bridges Viam's async AudioIn streaming API with hearken's
    synchronous audio source interface.
    """

    def __init__(
        self,
        microphone_client: AudioIn,
        sample_rate: int = 16000,
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
        self._buffer = bytearray()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def open(self) -> None:
        """Open the audio source for reading."""
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create a new one
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        self._stop_event = asyncio.Event()

        # Start async streaming in background
        asyncio.create_task(self._start_stream())

    def close(self) -> None:
        """Close the audio source and release resources."""
        if self._stop_event:
            self._stop_event.set()

        if self._audio_stream and self._loop:
            # Close the stream asynchronously
            async def close_stream():
                try:
                    await self._audio_stream.aclose()
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error closing audio stream: {e}")

            asyncio.run_coroutine_threadsafe(close_stream(), self._loop)

    async def _start_stream(self):
        """Internal method to start streaming audio from Viam component."""
        try:
            self._audio_stream = await self.microphone_client.get_audio("pcm16", 0, 0)
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

        if self._loop is None or self._audio_stream is None:
            return b''

        # Run async read in the event loop
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._read_async(chunk_size),
                self._loop
            )
            result = future.result(timeout=1.0)
            return result if result is not None else b''
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error reading from Viam audio source: {e}")
            return b''

    async def _read_async(self, chunk_size: int) -> Optional[bytes]:
        """Internal async method to read from audio stream."""
        if self._audio_stream is None:
            return None

        # Fill buffer until we have enough data
        while len(self._buffer) < chunk_size:
            if self._stop_event and self._stop_event.is_set():
                break

            try:
                # Get next audio chunk from stream
                resp = await self._audio_stream.__anext__()
                audio_data = resp.audio.audio_data

                # Note: We assume the sample rate matches. For production,
                # you might want to add resampling here if needed.
                self._buffer.extend(audio_data)

            except StopAsyncIteration:
                break
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error reading from audio stream: {e}")
                break

        # Return requested chunk
        if len(self._buffer) >= chunk_size:
            chunk = bytes(self._buffer[:chunk_size])
            self._buffer = self._buffer[chunk_size:]
            return chunk
        elif len(self._buffer) > 0:
            # Return what we have if stream ended
            chunk = bytes(self._buffer)
            self._buffer.clear()
            return chunk
        else:
            return None

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
