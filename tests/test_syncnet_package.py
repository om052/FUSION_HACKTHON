"""Focused tests for the AGNI SyncNet package foundation (Phase 2A).

These tests verify the *unavailable-state* contract: importing the package
must not fail merely because weights are missing, the loader/model guardians
must raise a clear, typed error, no network or checkpoint download is ever
attempted, and the existing ``SyncAnalysisResult`` result contract is unchanged.
No real synchronization inference is claimed here.
"""

from pathlib import Path
import importlib
from unittest.mock import patch

import pytest

from agni.models.syncnet import (
    SYNCNET_ARCHITECTURE_AVAILABLE,
    SyncNet,
    SyncNetCheckpointError,
    load_syncnet,
    load_syncnet_strict,
)
from agni.contracts.analyzers import SyncAnalysisResult, VisualAnalysisResult


# ---------------------------------------------------------------------------
#  Package import without a checkpoint (and without torch)
# ---------------------------------------------------------------------------

def test_package_import_does_not_require_a_checkpoint():
    """Importing agni.models.syncnet must succeed even with no weights present."""
    module = importlib.import_module("agni.models.syncnet")
    assert module is not None
    for name in (
        "SYNCNET_ARCHITECTURE_AVAILABLE",
        "SyncNet",
        "load_syncnet",
        "load_syncnet_strict",
        "SyncNetCheckpointError",
    ):
        assert hasattr(module, name), f"missing exported name: {name}"


def test_syncnet_package_is_importable_without_torch():
    """The unavailable-state package must not import torch at module load."""
    for modname in (
        "agni.models.syncnet",
        "agni.models.syncnet.loader",
        "agni.models.syncnet.model",
    ):
        spec_src = (Path(modname.replace(".", "/"))).as_posix()
    # Inspect the source files directly so we do not depend on import side effects.
    pkg_dir = Path(__import__("agni.models.syncnet", fromlist=["__path__"]).__path__[0])
    for pyfile in ("loader.py", "model.py", "__init__.py"):
        source = (pkg_dir / pyfile).read_text()
        assert "import torch" not in source, f"{pyfile} imports torch"
        assert "from torch" not in source, f"{pyfile} imports from torch"
    assert SYNCNET_ARCHITECTURE_AVAILABLE is False


def test_syncnet_checkpoint_error_is_exported_typing():
    """SyncNetCheckpointError must be a public, typed Exception subclass."""
    assert issubclass(SyncNetCheckpointError, Exception)
    # Must NOT be a bare/generic alias that hides its type.
    assert SyncNetCheckpointError.__module__ == "agni.models.syncnet.loader"


# ---------------------------------------------------------------------------
#  Missing checkpoint handling
# ---------------------------------------------------------------------------

def test_load_syncnet_strict_raises_on_missing_path(tmp_path: Path):
    """A nonexistent checkpoint path must raise the typed error."""
    missing = tmp_path / "does_not_exist.model"
    with pytest.raises(SyncNetCheckpointError) as excinfo:
        load_syncnet_strict(missing)
    assert "not found" in str(excinfo.value).lower()
    assert str(missing.resolve()) in str(excinfo.value)


def test_load_syncnet_strict_raises_on_existing_path_when_arch_unavailable(tmp_path: Path):
    """Even with a file present, an unverified architecture must not load it."""
    fake_checkpoint = tmp_path / "syncnet_v2.model"
    fake_checkpoint.write_bytes(b"\x00" * 16)
    with pytest.raises(SyncNetCheckpointError) as excinfo:
        load_syncnet_strict(fake_checkpoint)
    # The error must mention the architecture, not pretend to have loaded weights.
    msg = str(excinfo.value).lower()
    assert "verified" in msg or "architecture" in msg
    assert "random" in msg or "strict=false" in msg.replace("strict = false", "strict=false")


def test_load_syncnet_strict_no_strict_false_concealment():
    """The loader must never use strict=False to hide architecture mismatches.

    Because the architecture is unverified, the loader performs no torch
    state-dict loading at all — which is the strongest possible guarantee
    against strict=False concealment.
    """
    pkg_dir = Path(
        __import__("agni.models.syncnet.loader", fromlist=["__file__"]).__file__
    )
    source = pkg_dir.read_text()
    assert "load_state_dict" not in source
    assert "torch.load" not in source


# ---------------------------------------------------------------------------
#  Clear, typed error messages
# ---------------------------------------------------------------------------

def test_syncnet_instantiation_raises_typed_error():
    """Instantiating SyncNet must raise SyncNetCheckpointError (not generic)."""
    with pytest.raises(SyncNetCheckpointError) as excinfo:
        SyncNet()
    msg = str(excinfo.value).lower()
    assert "not implemented" in msg or "not available" in msg
    assert "random weights" in msg


def test_load_syncnet_raises_typed_error():
    """load_syncnet must raise SyncNetCheckpointError (not perform inference)."""
    with pytest.raises(SyncNetCheckpointError) as excinfo:
        load_syncnet()
    assert "unavailable" in str(excinfo.value).lower()


def test_syncnet_is_available_reports_false():
    """The availability flag and helper must both report unavailable."""
    assert SYNCNET_ARCHITECTURE_AVAILABLE is False
    assert SyncNet.is_available() is False


def test_error_messages_are_actionable():
    """Error messages must point at the root cause and remediation."""
    try:
        SyncNet()
    except SyncNetCheckpointError as exc:
        msg = str(exc)
        assert "SyncNet" in msg
        assert "architecture" in msg.lower()
    else:
        pytest.fail("SyncNet() did not raise SyncNetCheckpointError")


# ---------------------------------------------------------------------------
#  No network or model download attempts
# ---------------------------------------------------------------------------

def test_load_syncnet_strict_never_reaches_network(tmp_path: Path):
    """Calling the loader must not perform any network request."""
    sentinel = AssertionError("network access attempted during SyncNet load")

    # Patch every plausible network entry point. If any fires, the sentinel
    # aborts the test; load_syncnet_strict must instead raise SyncNetCheckpointError.
    with patch("urllib.request.urlopen", side_effect=sentinel), \
         patch("urllib.request.urlretrieve", side_effect=sentinel), \
         patch("urllib.request.Request", side_effect=sentinel), \
         patch("socket.socket", side_effect=sentinel):
        with pytest.raises(SyncNetCheckpointError):
            load_syncnet_strict(tmp_path / "missing.model")


def test_syncnet_loader_source_has_no_network_or_download_primitives():
    """The loader source must contain no URL/HTTP/download calls."""
    pkg_dir = Path(
        __import__("agni.models.syncnet.loader", fromlist=["__file__"]).__file__
    )
    source = pkg_dir.read_text().lower()
    forbidden = ["urlopen", "requests.get", "requests.post", "http.client",
                 "download_artifact", "download(", "urllib.request"]
    for token in forbidden:
        assert token not in source, f"loader.py references network primitive: {token}"


def test_no_checkpoint_is_written_by_loader(tmp_path: Path):
    """Loading must never create/overwrite checkpoint files on disk."""
    target = tmp_path / "would_be_downloaded.model"
    assert not target.exists()
    with pytest.raises(SyncNetCheckpointError):
        load_syncnet_strict(target)
    assert not target.exists(), "loader wrote a checkpoint file to disk"


# ---------------------------------------------------------------------------
#  Existing result contracts remain unchanged
# ---------------------------------------------------------------------------

def test_sync_analysis_result_contract_unchanged():
    """SyncAnalysisResult must still accept the contract fields used elsewhere."""
    result = SyncAnalysisResult(
        model_architecture="SyncNet",
        model_version="v2",
        status="COMPLETED",
        has_audio_track=True,
        mouth_rois_extracted_count=30,
        audio_speech_detected=True,
        temporal_offset_frames=0.0,
        temporal_offset_ms=0.0,
        confidence_score=4.5,
        is_score_calibrated=False,
    )
    assert result.analyzer_name == "av_sync_detector"
    assert result.model_architecture == "SyncNet"
    assert result.status.value == "COMPLETED"
    assert result.temporal_offset_ms == 0.0
    assert result.is_score_calibrated is False


def test_sync_analysis_result_not_available_status_valid():
    """The NOT_AVAILABLE status (used when SyncNet is unavailable) must be valid."""
    result = SyncAnalysisResult(status="NOT_AVAILABLE", error_message="SyncNet unavailable")
    assert result.status.value == "NOT_AVAILABLE"
    assert result.error_message == "SyncNet unavailable"


def test_visual_analysis_result_contract_unchanged():
    """VisualAnalysisResult contract must remain intact (sibling contract)."""
    result = VisualAnalysisResult(
        model_architecture="Xception",
        model_version="v1",
        status="COMPLETED",
        faces_detected_count=30,
        frames_evaluated_count=30,
        raw_score=0.12,
        is_score_calibrated=False,
    )
    assert result.analyzer_name == "visual_deepfake_detector"
    assert result.raw_score == 0.12
