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
from hearken import AsyncAudioSource


class ViamAudioInSource(AsyncAudioSource):
    """Async audio source for Viam AudioIn component.

    Implements hearken's AsyncAudioSource interface to provide native async
    streaming from Viam's AudioIn component.
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

    def stream(self):
        """Return async iterator for streaming audio."""
        if self.logger:
            self.logger.info("ViamAudioInSource.stream() called - returning async iterator")
        return self._stream_audio_chunks()

    async def _stream_audio_chunks(self):
        """Async generator that yields resampled audio chunks."""
        chunk_count = 0
        try:
            self.logger.debug("_stream_audio_chunks: Starting")
            # Get actual properties from the microphone
            properties = await self.microphone_client.get_properties()
            self.logger.info("_stream_audio_chunks: Got properties")
            source_sample_rate = properties.sample_rate_hz
            if self.logger:
                self.logger.info(f"Stream: Microphone sample rate: {source_sample_rate}Hz")

            # Target sample rate for VAD (WebRTC VAD requires 8/16/32 kHz)
            target_sample_rate = 16000
            needs_resampling = source_sample_rate != target_sample_rate

            if needs_resampling:
                self.logger.debug(f"Stream: Will resample from {source_sample_rate}Hz to {target_sample_rate}Hz")

            # Expose target sample rate to hearken
            self._sample_rate = target_sample_rate

            # Get audio stream - this starts the capture
            audio_stream = await self.microphone_client.get_audio("pcm16", 0, 0)
            self.logger.info("Stream: Audio stream acquired, starting to yield audio chunks")

            # Get event loop for running blocking operations
            loop = asyncio.get_event_loop()

            async for resp in audio_stream:
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

                yield audio_data

        except asyncio.CancelledError:
            if self.logger:
                self.logger.debug("Stream: Audio streaming cancelled")
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(f"Stream: Error in audio streaming: {e}")
            raise

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
