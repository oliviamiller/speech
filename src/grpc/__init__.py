"""
This file registers the speech subtype with the Viam Registry, as well as the specific SpeechIOService model.
"""

from viam.resource.registry import Registry, ResourceRegistration

from .api import AudioClient, Audio, AudioRPCService

Registry.register_api(
    ResourceRegistration(
        Audio,
        AudioRPCService,
        lambda name, channel: AudioClient(name, channel),
    )
)
