"""SyncNet audio-video synchronization model (Phase 2A: foundation only).

The verified SyncNet architecture is **not** yet implemented in this
repository and no local checkpoint is bundled, so the symbols below are
*guardians*: importing this package is always safe (it neither requires
PyTorch nor weights), but building or loading the model raises
:class:`~agni.models.syncnet.loader.SyncNetCheckpointError` so no fake
synchronization scores can be silently produced.

Upstream reference: Chung & Zisserman, "Out of time: automated lip sync in the
wild", ACCV 2016; reference implementation
https://github.com/joonson/syncnet_python (MIT).  Weights (syncnet_v2.model)
are CC BY 4.0 from the Oxford VGG lipsync page and must be provisioned
separately; they are never downloaded here.
"""

from agni.models.syncnet.loader import (
    SYNCNET_ARCHITECTURE_AVAILABLE,
    SyncNetCheckpointError,
    load_syncnet_strict,
)
from agni.models.syncnet.model import SyncNet, load_syncnet
from agni.models.syncnet.analyzer import SyncAnalyzer

__all__ = [
    "SYNCNET_ARCHITECTURE_AVAILABLE",
    "SyncNet",
    "load_syncnet",
    "load_syncnet_strict",
    "SyncNetCheckpointError",
    "SyncAnalyzer",
]
