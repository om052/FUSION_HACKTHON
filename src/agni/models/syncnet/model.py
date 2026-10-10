"""SyncNet audio-video synchronization model (Phase 2A: foundation only).

The upstream SyncNet architecture is **not** reproduced in this repository.
A verified architecture requires bit-exact submodule/parameter names that
match a locally-verified ``syncnet_v2.model`` checkpoint, and no such
checkpoint exists locally (nor is one ever downloaded here).

Accordingly this module exposes *guardian* symbols — :class:`SyncNet` and
:func:`load_syncnet` — that are explicitly non-functional: instantiating the
model or calling the loader raises
:class:`~agni.models.syncnet.loader.SyncNetCheckpointError` with a clear
message.  This guarantees that merely importing
``agni.models.syncnet`` never fails on missing weights (and does not even
require PyTorch), while making it impossible to silently obtain random
predictions or pretend synchronization inference ran.

Upstream reference: Chung & Zisserman, "Out of time: automated lip sync in the
wild", ACCV 2016; reference implementation
https://github.com/joonson/syncnet_python (``SyncNetModel.py``, MIT license).
Weights (syncnet_v2.model) are CC BY 4.0 from the Oxford VGG lipsync page and
must be provisioned separately; they are never fetched here.
"""

from typing import Any, Union

from agni.models.syncnet.loader import (
    SYNCNET_ARCHITECTURE_AVAILABLE,
    SyncNetCheckpointError,
)

__all__ = [
    "SYNCNET_ARCHITECTURE_AVAILABLE",
    "SyncNet",
    "load_syncnet",
]


class SyncNet:
    """Unavailable-state guardian for the SyncNet model.

    The real SyncNet architecture is not implemented in this repository and no
    verified checkpoint exists.  Instantiating this class raises
    :class:`SyncNetCheckpointError` so callers cannot accidentally receive a
    randomly-initialised model or pretend synchronization scores.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise SyncNetCheckpointError(
            "SyncNet model architecture is not implemented in this repository. "
            "A verified architecture (bit-exact submodule and parameter names "
            "matching syncnet_v2.model) is required before the model can be "
            "instantiated. Refusing to construct random weights."
        )

    @staticmethod
    def is_available() -> bool:
        """Return ``False`` because the SyncNet architecture is not available."""
        return SYNCNET_ARCHITECTURE_AVAILABLE


def load_syncnet(path: Union[str, None] = None, **kwargs: Any) -> Any:
    """Load a SyncNet model from a checkpoint.

    Raises :class:`SyncNetCheckpointError` because the verified SyncNet
    architecture is not available.  No network access, no download, and no
    random-weight fallback occur.

    Args:
        path: Optional checkpoint path (ignored — unavailable).
        **kwargs: Ignored.

    Raises:
        SyncNetCheckpointError: Always, with a clear explanation.
    """
    raise SyncNetCheckpointError(
        "load_syncnet is unavailable: no verified SyncNet architecture is "
        "implemented in this repository. Install a verified architecture and a "
        "compatible checkpoint before calling load_syncnet."
    )
