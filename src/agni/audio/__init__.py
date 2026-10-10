"""Audio feature extraction for AGNI's audio-video synchronization pipeline.

Provides SyncNet-compatible MFCC feature extraction from 16 kHz mono PCM WAV
audio. This module is a pure feature-extraction stage: it does not perform
model inference, synchronization scoring, or deepfake classification.
"""

from agni.audio.feature_extractor import (
    AudioFeatureExtractor,
    AudioFeatureExtractionError,
    InsufficientAudioDuration,
    MalformedAudioInput,
    SilentAudioInput,
)

__all__ = [
    "AudioFeatureExtractor",
    "AudioFeatureExtractionError",
    "InsufficientAudioDuration",
    "MalformedAudioInput",
    "SilentAudioInput",
]