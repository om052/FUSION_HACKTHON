"""Tests for the AGNI audio preprocessing + sync diagnostic pipeline (Phase 2B).

Covers the distinct audio cases (no audio track, extraction failure, silent,
too short, malformed, MFCC success) plus dashboard diagnostics rendering.
Uses real generated video/audio fixtures where possible and mocks only for
branches that are hard to trigger deterministically with ffmpeg.  No external
downloads or model weights are required.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agni.audio.feature_extractor import (
    InsufficientAudioDuration,
    MalformedAudioInput,
    SilentAudioInput,
)
from agni.contracts.analyzers import SyncAnalysisResult
from agni.contracts.enums import AnalyzerStatus, MediaType
from agni.contracts.media import AudioStreamMetadata, MediaInspectionResult
from agni.dashboard_helpers import (
    render_audio_sync_diagnostics,
    run_sync_analysis,
)
from agni.intake.validator import MediaValidator
from agni.models.syncnet import SYNCNET_ARCHITECTURE_AVAILABLE, SyncAnalyzer


# ---------------------------------------------------------------------------
#  Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer() -> SyncAnalyzer:
    return SyncAnalyzer()


def _inspection_with_audio() -> MediaInspectionResult:
    return MediaInspectionResult(
        file_path="x.mp4",
        file_name="x.mp4",
        file_size_bytes=1,
        media_type=MediaType.VIDEO,
        is_valid=True,
        audio=AudioStreamMetadata(has_audio=True, sample_rate=16000, channels=1),
    )


def _inspection_without_audio() -> MediaInspectionResult:
    return MediaInspectionResult(
        file_path="x.mp4",
        file_name="x.mp4",
        file_size_bytes=1,
        media_type=MediaType.VIDEO,
        is_valid=True,
        audio=AudioStreamMetadata(has_audio=False),
    )


# ---------------------------------------------------------------------------
#  Real end-to-end cases (use conftest ffmpeg fixtures)
# ---------------------------------------------------------------------------

def test_analyze_video_with_audio_extracts_mfcc_and_reports_syncnet_unavailable(
    analyzer: SyncAnalyzer, valid_video_with_audio: Path, tmp_path: Path
):
    """Success path: MFCC extracted, but SyncNet inference stays NOT_AVAILABLE."""
    inspection = MediaValidator().inspect_media(valid_video_with_audio)
    assert inspection.audio is not None and inspection.audio.has_audio

    result = analyzer.analyze(
        valid_video_with_audio, media_inspection=inspection, output_wav_dir=tmp_path
    )

    assert isinstance(result, SyncAnalysisResult)
    assert result.status == AnalyzerStatus.NOT_AVAILABLE
    assert result.has_audio_track is True
    assert result.audio_speech_detected is True
    assert result.mouth_rois_extracted_count == 0
    # No invented synchronization metrics
    assert result.temporal_offset_frames is None
    assert result.temporal_offset_ms is None
    assert result.min_distance_score is None
    assert result.confidence_score is None
    assert result.is_score_calibrated is False
    assert result.model_architecture is None
    # Audio feature evidence is recorded truthfully
    assert result.diagnostics["stage"] == "mfcc_extracted_syncnet_unavailable"
    assert result.diagnostics["syncnet_available"] is False
    assert result.diagnostics["mfcc_frames"] >= 20
    assert "unavailable" in result.error_message.lower()


def test_analyze_video_without_audio_reports_not_available(
    analyzer: SyncAnalyzer, valid_video_no_audio: Path, tmp_path: Path
):
    """No-audio-track case is classified NOT_AVAILABLE without attempting extraction."""
    inspection = MediaValidator().inspect_media(valid_video_no_audio)
    assert inspection.audio is not None and inspection.audio.has_audio is False

    result = analyzer.analyze(
        valid_video_no_audio, media_inspection=inspection, output_wav_dir=tmp_path
    )

    assert result.status == AnalyzerStatus.NOT_AVAILABLE
    assert result.has_audio_track is False
    assert result.audio_speech_detected is False
    assert result.mouth_rois_extracted_count == 0
    assert result.temporal_offset_ms is None
    assert result.confidence_score is None
    assert "no audio track" in result.error_message.lower()


def test_analyze_silent_audio_reports_not_available(
    analyzer: SyncAnalyzer, valid_video_silent_audio: Path, tmp_path: Path
):
    """Silent audio yields NOT_AVAILABLE (no speech signal)."""
    inspection = MediaValidator().inspect_media(valid_video_silent_audio)
    assert inspection.audio is not None and inspection.audio.has_audio

    result = analyzer.analyze(
        valid_video_silent_audio, media_inspection=inspection, output_wav_dir=tmp_path
    )

    assert result.status == AnalyzerStatus.NOT_AVAILABLE
    assert result.has_audio_track is True
    assert result.audio_speech_detected is False
    assert result.diagnostics["stage"] == "silent_audio"
    assert result.temporal_offset_ms is None
    assert result.confidence_score is None


# ---------------------------------------------------------------------------
#  Mocked edge-case handling (deterministic, no ffmpeg)
# ---------------------------------------------------------------------------

def test_analyze_extraction_failure_is_error(tmp_path: Path):
    """A failed audio extraction is an ERROR, distinct from no-audio-track."""
    mock_processor = MagicMock()
    mock_processor.extract_audio_wav.return_value = False
    analyzer = SyncAnalyzer(processor=mock_processor)

    result = analyzer.analyze(
        tmp_path / "x.mp4", media_inspection=_inspection_with_audio(),
        output_wav_dir=tmp_path,
    )

    assert result.status == AnalyzerStatus.ERROR
    assert result.has_audio_track is True
    assert result.audio_speech_detected is False
    assert result.diagnostics["stage"] == "extraction_failed"
    assert "extraction failed" in result.error_message.lower()
    assert result.confidence_score is None


def test_analyze_insufficient_duration_is_error(tmp_path: Path):
    """Audio too short for the Syncnet window is an ERROR, not a fake score."""
    mock_processor = MagicMock()
    mock_processor.extract_audio_wav.return_value = True
    mock_extractor = MagicMock()
    mock_extractor.min_model_frames = 20
    mock_extractor.extract_mfcc_for_syncnet.side_effect = InsufficientAudioDuration(
        "only 3 frames"
    )
    analyzer = SyncAnalyzer(processor=mock_processor, feature_extractor=mock_extractor)

    result = analyzer.analyze(
        tmp_path / "x.mp4", media_inspection=_inspection_with_audio(),
        output_wav_dir=tmp_path,
    )

    assert result.status == AnalyzerStatus.ERROR
    assert result.has_audio_track is True
    assert result.audio_speech_detected is False
    assert result.diagnostics["stage"] == "insufficient_duration"
    assert result.temporal_offset_ms is None
    assert result.confidence_score is None


def test_analyze_malformed_audio_is_error(tmp_path: Path):
    """Malformed audio during MFCC extraction is an ERROR."""
    mock_processor = MagicMock()
    mock_processor.extract_audio_wav.return_value = True
    mock_extractor = MagicMock()
    mock_extractor.min_model_frames = 20
    mock_extractor.extract_mfcc_for_syncnet.side_effect = MalformedAudioInput("bad dtype")
    analyzer = SyncAnalyzer(processor=mock_processor, feature_extractor=mock_extractor)

    result = analyzer.analyze(
        tmp_path / "x.mp4", media_inspection=_inspection_with_audio(),
        output_wav_dir=tmp_path,
    )

    assert result.status == AnalyzerStatus.ERROR
    assert result.diagnostics["stage"] == "malformed_audio"
    assert result.confidence_score is None


def test_analyze_success_path_records_mfcc_and_no_invented_sync(
    tmp_path: Path,
):
    """When MFCC succeeds, diagnostics record features and sync is still unavailable."""
    feats = np.zeros((50, 13), dtype=np.float64)
    mock_processor = MagicMock()
    mock_processor.extract_audio_wav.return_value = True
    mock_extractor = MagicMock()
    mock_extractor.extract_mfcc_for_syncnet.return_value = feats
    mock_extractor.min_model_frames = 20
    analyzer = SyncAnalyzer(processor=mock_processor, feature_extractor=mock_extractor)

    result = analyzer.analyze(
        tmp_path / "x.mp4", media_inspection=_inspection_with_audio(),
        output_wav_dir=tmp_path,
    )

    assert result.status == AnalyzerStatus.NOT_AVAILABLE
    assert result.has_audio_track is True
    assert result.audio_speech_detected is True
    assert result.diagnostics["stage"] == "mfcc_extracted_syncnet_unavailable"
    assert result.diagnostics["mfcc_frames"] == 50
    assert result.diagnostics["syncnet_available"] is False
    assert result.temporal_offset_ms is None
    assert result.confidence_score is None
    # No weights were ever touched
    mock_processor.extract_audio_wav.assert_called_once()
    mock_extractor.extract_mfcc_for_syncnet.assert_called_once()


def test_analyzer_never_attempts_checkpoint_load():
    """The analyzer must not reference strict checkpoint loading or torch."""
    import agni.models.syncnet.analyzer as az_mod
    assert az_mod.SyncAnalyzer is SyncAnalyzer
    src = Path(az_mod.__file__).read_text()
    assert "load_syncnet_strict" not in src
    assert "load_syncnet" not in src
    assert "import torch" not in src
    assert "from torch" not in src


def test_syncnet_architecture_flag_is_false():
    """The pipeline can only ever report SyncNet as unavailable right now."""
    assert SYNCNET_ARCHITECTURE_AVAILABLE is False


# ---------------------------------------------------------------------------
#  Dashboard diagnostics rendering
# ---------------------------------------------------------------------------

def _patch_st():
    mock_st = MagicMock()
    return patch("agni.dashboard_helpers.st", mock_st), mock_st


def _all_text(mock_st: MagicMock) -> str:
    """Flatten every rendered string from the common streamlit calls."""
    parts = []
    for method in ("write", "info", "warning", "error", "caption"):
        for call_obj in getattr(mock_st, method).call_args_list:
            for arg in call_obj.args:
                parts.append(str(arg))
            for v in call_obj.kwargs.values():
                parts.append(str(v))
    for call_obj in mock_st.metric.call_args_list:
        for arg in call_obj.args:
            parts.append(str(arg))
    return " ".join(parts)


def _start_st():
    mock_st = MagicMock()
    patcher = patch("agni.dashboard_helpers.st", mock_st)
    patcher.start()
    return mock_st, patcher


def test_run_sync_analysis_delegates_to_analyzer(tmp_path: Path):
    """run_sync_analysis should construct SyncAnalyzer and delegate to analyze."""
    fake_result = SyncAnalysisResult(status=AnalyzerStatus.NOT_AVAILABLE)
    with patch("agni.dashboard_helpers.SyncAnalyzer") as MockAnalyzer:
        MockAnalyzer.return_value.analyze.return_value = fake_result
        result = run_sync_analysis(tmp_path / "x.mp4")
    assert result is fake_result
    MockAnalyzer.return_value.analyze.assert_called_once()


def test_render_audio_sync_diagnostics_no_audio_track():
    """No-audio case renders an info and the sync-unavailable warning."""
    result = SyncAnalysisResult(
        status=AnalyzerStatus.NOT_AVAILABLE,
        has_audio_track=False,
        audio_speech_detected=False,
        error_message="No audio track detected in media; synchronization cannot be assessed.",
        diagnostics={"stage": "no_audio_track", "syncnet_available": False},
    )
    mock_st, patcher = _start_st()
    try:
        render_audio_sync_diagnostics(result)
    finally:
        patcher.stop()
    text = _all_text(mock_st)
    assert "Audio track present: False" in text
    assert "no audio track" in text.lower()
    mock_st.info.assert_called()  # no-audio informational message


def test_render_audio_sync_diagnostics_success_features_unavailable():
    """MFCC-success case shows feature status separately from sync availability."""
    result = SyncAnalysisResult(
        status=AnalyzerStatus.NOT_AVAILABLE,
        has_audio_track=True,
        audio_speech_detected=True,
        mouth_rois_extracted_count=0,
        error_message="SyncNet unavailable: MFCC features pre-extracted.",
        diagnostics={
            "stage": "mfcc_extracted_syncnet_unavailable",
            "syncnet_available": False,
            "mfcc_frames": 200,
            "mfcc_shape": [200, 13],
        },
    )
    mock_st, patcher = _start_st()
    try:
        render_audio_sync_diagnostics(result)
    finally:
        patcher.stop()
    text = _all_text(mock_st)
    assert "Audio track present: True" in text
    assert "Speech signal detected: True" in text
    assert "MFCC frames extracted: 200" in text
    assert "Synchronization analyzer: SyncNet" in text
    mock_st.warning.assert_called()  # sync not available


def test_render_audio_sync_diagnostics_error_state(tmp_path: Path):
    """ERROR status renders an error, not a fake score."""
    result = SyncAnalysisResult(
        status=AnalyzerStatus.ERROR,
        has_audio_track=True,
        audio_speech_detected=False,
        error_message="Audio extraction failed.",
        diagnostics={"stage": "extraction_failed", "syncnet_available": False},
    )
    mock_st, patcher = _start_st()
    try:
        render_audio_sync_diagnostics(result)
    finally:
        patcher.stop()
    text = _all_text(mock_st)
    assert "Audio track present" in text
    assert "Synchronization analyzer" in text
    mock_st.error.assert_called()


def test_render_audio_sync_diagnostics_never_shows_invented_scores():
    """The renderer must never emit a metric widget or numeric sync measurement."""
    result = SyncAnalysisResult(
        status=AnalyzerStatus.NOT_AVAILABLE,
        has_audio_track=True,
        audio_speech_detected=True,
        diagnostics={
            "stage": "mfcc_extracted_syncnet_unavailable",
            "syncnet_available": False,
            "mfcc_frames": 100,
        },
    )
    mock_st, patcher = _start_st()
    try:
        render_audio_sync_diagnostics(result)
    finally:
        patcher.stop()
    # No metric widgets => no numeric offset/confidence value can be displayed.
    mock_st.metric.assert_not_called()
    # And the result carries no invented measurements to render.
    assert result.temporal_offset_ms is None
    assert result.confidence_score is None
    assert result.min_distance_score is None
    assert result.temporal_offset_frames is None
