import abc
from typing import Sequence

from grpclib.client import Channel
from grpclib.server import Stream

from viam.resource.types import RESOURCE_TYPE_COMPONENT, API
from viam.components.component_base import ComponentBase

from .audio_grpc import AudioServiceStub
from .audio_pb2 import (
    RecordRequest,
    AudioChunk,
    PlayRequest,
    AudioInfo,
    AudioFormat
)

from viam.streams import StreamWithIterator

AudioStream = Stream[AudioChunk]

class Audio(ComponentBase):
    API = API("olivia", RESOURCE_TYPE_COMPONENT, "audio")

    @abc.abstractmethod
    async def record(self, format: str, sampleRate: int, channels:int, durationSeconds: int) -> AudioStream: ...

    @abc.abstractmethod
    async def play(self, audio: bytearray, format:str, sampleRate: int, channels: int): ...



class AudioClient(Audio):
    def __init__(self, name: str, channel: Channel) -> None:
        self.channel = channel
        self.client = AudioServiceStub(channel)
        super().__init__(name)

    async def record(self, Audioformat: str, sampleRate: int, channels:int, durationSeconds: int) -> AudioStream:
        request = RecordRequest(name=self.name, duration_seconds=durationSeconds, info=AudioInfo(channels=channels, sample_rate=sampleRate, format=AudioFormat(Audioformat)))
        async def read():
            audio_stream: Stream[RecordRequest, AudioChunk]
            async with self.client.Record.open() as audio_stream:
                try:
                    await audio_stream.send_message(request, end=True)
                    async for audioChunk in audio_stream:
                        yield audioChunk
                except Exception as e:
                    raise (e)

        return StreamWithIterator(read())

    async def play(self, audio: bytearray, Audioformat:str, sampleRate: int, channels: int):
        request = PlayRequest(name=self.name, audio_data=audio, sample_rate=sampleRate, channels=channels, format=AudioFormat(Audioformat))
        await self.client.Play(request)









