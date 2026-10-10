"""SyncNet checkpoint loading (Phase 2A: foundation only).

This module provides the contract for loading a Syncnet checkpoint with
``strict=True`` semantics, plus a single typed error,
:class:`SyncNetCheckpointError`.

At present the verified SyncNet architecture is **not** implemented inside
this repository and no local checkpoint exists, so :func:`load_syncnet_strict`
always raises :class:`SyncNetCheckpointError` rather than silently constructing
random weights or falling back to ``strict=False``.  No network access and no
checkpoint download is performed.

Upstream reference: Chung & Zisserman, "Out of time: automated lip sync in the
wild", Workshop on Multi-view Lip-reading, ACCV 2016; reference implementation
https://github.com/joonson/syncnet_python (``SyncNetModel.py`` /
``SyncNetInstance.py``, MIT license).  The ``syncnet_v2.model`` weights are
CC BY 4.0 from the Oxford VGG lipsync page and must be provisioned separately
— they are intentionally never fetched here.
"""

from pathlib import Path
from typing import Any, Union

__all__ = [
    "SyncNetCheckpointError",
    "SYNCNET_ARCHITECTURE_AVAILABLE",
    "load_syncnet_strict",
]

#: A bit-exact SyncNet architecture (matching the parameter names of the
#: upstream ``syncnet_v2.model`` checkpoint) has NOT yet been verified against a
#: local checkpoint in this repository.  When it is, this flag becomes ``True``
#: and :func:`load_syncnet_strict` may perform a strict ``state_dict`` load.
SYNCNET_ARCHITECTURE_AVAILABLE: bool = False


class SyncNetCheckpointError(Exception):
    """Raised when a verified SyncNet model implementation or a compatible
    checkpoint is unavailable for strict loading.

    This is a *typed* error (not a bare ``RuntimeError`` or
    ``FileNotFoundError``) so callers can distinguish "SyncNet cannot run" from
    other failure modes and surface an honest ``NOT_AVAILABLE`` result on the
    evidence receipt, rather than silently emitting fabricated synchronization
    scores.
    """


def load_syncnet_strict(path: Union[str, Path]) -> Any:
    """Resolve and strictly load a Syncnet checkpoint.

    Verifies that the checkpoint file exists at ``path`` and that a verified
    architecture is registered for it.  Because the SyncNet architecture is
    not yet implemented in this repository, this function always raises
    :class:`SyncNetCheckpointError` — never returning random weights and never
    using ``strict=False``.

    Args:
        path: Filesystem path to an expected ``syncnet_v2.model`` checkpoint.

    Returns:
        Never returns in the current (unavailable) state.  When an architecture
        is later verified, this will return a model populated with weights
        loaded via ``strict=True``.

    Raises:
        SyncNetCheckpointError: If the requested path does not exist, or if no
            verified model implementation / compatible checkpoint is available.
    """
    checkpoint_path = Path(path).resolve()

    if not checkpoint_path.exists():
        raise SyncNetCheckpointError(
            f"SyncNet checkpoint not found at requested path: {checkpoint_path}. "
            "No verified model implementation or compatible checkpoint is available."
        )

    if not SYNCNET_ARCHITECTURE_AVAILABLE:
        raise SyncNetCheckpointError(
            f"A checkpoint file was found at {checkpoint_path}, but no verified "
            "SyncNet model architecture is implemented in this repository, so it "
            "cannot be loaded with strict=True. A bit-exact architecture matching "
            "the syncnet_v2.model parameter names must be established before any "
            "checkpoint can be strictly loaded. Refusing to fall back to random "
            "weights or use strict=False."
        )

    # Unreachable while SYNCNET_ARCHITECTURE_AVAILABLE is False; documents the
    # intended strict-load behaviour once a verified architecture exists.
    raise SyncNetCheckpointError(
        "SyncNet architecture is unavailable; strict checkpoint loading is disabled."
    )
