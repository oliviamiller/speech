import abc
from typing import Sequence

from grpclib.client import Channel
from grpclib.server import Stream

from viam.resource.types import RESOURCE_TYPE_COMPONENT, API
from viam.components.component_base import ComponentBase
from viam.resource.rpc_service_base import ResourceRPCServiceBase

from .audio_grpc import AudioServiceStub, AudioServiceBase
from .audio_pb2 import (
    RecordRequest,
    AudioChunk,
    PlayRequest,
    AudioInfo,
    AudioFormat,
    PropertiesRequest,
    PropertiesResponse,
    PlayResponse
)

from viam.streams import StreamWithIterator

AudioStream = Stream[AudioChunk]

class Audio(ComponentBase):
    API = API("olivia", RESOURCE_TYPE_COMPONENT, "audio")

    @abc.abstractmethod
    async def record(self, format: str, sampleRate: int, channels:int, durationSeconds: int) -> AudioStream: ...

    @abc.abstractmethod
    async def play(self, audio: bytearray, format:str, sampleRate: int, channels: int): ...


class AudioRPCService(AudioServiceBase, ResourceRPCServiceBase):
    RESOURCE_TYPE = Audio

    async def Record(self, stream: Stream[RecordRequest, AudioChunk]) -> None:
        return

    async def Play(self, stream: Stream[PlayRequest, PlayResponse]) -> None:
        print("here play")
        return

    async def Properties(self, stream: Stream[PropertiesRequest, PropertiesResponse]):
        return await super().Properties(stream)



class AudioClient(Audio):
    def __init__(self, name: str, channel: Channel) -> None:
        self.channel = channel
        self.client = AudioServiceStub(channel)
        super().__init__(name)

    async def record(self, Audioformat: str, sampleRate: int, channels:int, durationSeconds: int) -> AudioStream:
        # Convert string format to enum value
        if Audioformat.upper() == "MP3":
            format_enum = AudioFormat.MP3
        elif Audioformat.upper() == "PCM16":
            format_enum = AudioFormat.PCM16
        elif Audioformat.upper() == "PCM32":
            format_enum = AudioFormat.PCM32
        elif Audioformat.upper() == "PCM32_FLOAT":
            format_enum = AudioFormat.PCM32_FLOAT
        elif Audioformat.upper() == "WAV":
            format_enum = AudioFormat.WAV
        else:
            format_enum = AudioFormat.MP3  # default

        request = RecordRequest(name=self.name, duration_seconds=durationSeconds, info=AudioInfo(channels=channels, sample_rate=sampleRate, format=format_enum))
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

    async def play(self, audio: bytes, Audioformat:str, sampleRate: int, channels: int):
        # Convert string format to enum value
        if Audioformat.upper() == "MP3":
            format_enum = AudioFormat.MP3
        elif Audioformat.upper() == "PCM16":
            format_enum = AudioFormat.PCM16
        elif Audioformat.upper() == "PCM32":
            format_enum = AudioFormat.PCM32
        elif Audioformat.upper() == "PCM32_FLOAT":
            format_enum = AudioFormat.PCM32_FLOAT
        elif Audioformat.upper() == "WAV":
            format_enum = AudioFormat.WAV
        else:
            format_enum = AudioFormat.MP3  # default

        request = PlayRequest(name=self.name, audio_data=audio, sample_rate=sampleRate, channels=channels, format=format_enum)
        print("here audio client")
        await self.client.Play(request)
