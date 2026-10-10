"""Tests for the AGNI Verification Pipeline.

Covers successful orchestration, invalid media, unavailable SyncNet,
detector errors, receipt generation, and the rule that missing evidence
cannot result in an unsupported VERIFIED decision.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agni.contracts.enums import (
    ActionRecommendation,
    AnalyzerStatus,
    AssessmentDecision,
    MediaType,
    UncertaintyFlag,
)
from agni.contracts.media import (
    AudioStreamMetadata,
    MediaInspectionResult,
    VideoStreamMetadata,
)
from agni.contracts.analyzers import SyncAnalysisResult, VisualAnalysisResult
from agni.contracts.receipt import EvidenceReceipt
from agni.pipeline import AGNIPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def yunet_path() -> Path:
    return Path("models/yunet/face_detection_yunet_2023mar.onnx")


@pytest.fixture
def xception_path() -> Path:
    return Path("models/visual_detector/xception_best.pth")


def _make_pipeline(yunet_path: Path, xception_path: Path) -> AGNIPipeline:
    """Factory to create a pipeline with standard test config."""
    return AGNIPipeline(
        yunet_model_path=yunet_path,
        xception_checkpoint_path=xception_path,
        max_frames=8,
    )


# ---------------------------------------------------------------------------
# Valid video fixtures (from conftest) - tests without mocking
# ---------------------------------------------------------------------------

def test_pipeline_successful_orchestration_with_audio(
    yunet_path: Path, xception_path: Path, valid_video_with_audio: Path, tmp_path: Path
):
    """Full pipeline runs all stages and produces a complete EvidenceReceipt."""
    pipeline = _make_pipeline(yunet_path, xception_path)
    receipt = pipeline.run(valid_video_with_audio, session_id="test-session", output_wav_dir=tmp_path)

    assert isinstance(receipt, EvidenceReceipt)
    assert receipt.session_id == "test-session"
    assert receipt.system_version == "AGNI-0.1.0"
    assert receipt.receipt_id is not None
    assert receipt.created_at_utc is not None

    # Media inspection should be valid
    assert receipt.media_inspection.is_valid is True
    assert receipt.media_inspection.media_type == MediaType.VIDEO
    assert receipt.media_inspection.audio is not None
    assert receipt.media_inspection.audio.has_audio is True

    # Visual analysis should complete
    assert receipt.visual_analysis.status in (
        AnalyzerStatus.COMPLETED,
        AnalyzerStatus.NOT_AVAILABLE,
        AnalyzerStatus.ERROR,
    )
    assert receipt.visual_analysis.frames_evaluated_count >= 0

    # Sync analysis should run and report NOT_AVAILABLE (SyncNet not integrated)
    assert receipt.sync_analysis.status in (
        AnalyzerStatus.NOT_AVAILABLE,
        AnalyzerStatus.ERROR,
    )
    assert receipt.sync_analysis.has_audio_track is True
    # No invented sync metrics
    assert receipt.sync_analysis.temporal_offset_ms is None
    assert receipt.sync_analysis.confidence_score is None
    assert receipt.sync_analysis.min_distance_score is None

    # Uncertainty report should exist
    assert isinstance(receipt.uncertainty_report, type(receipt.uncertainty_report))
    assert receipt.uncertainty_report.summary_reason is not None

    # Decision should be INCONCLUSIVE (missing SyncNet) or other non-VERIFIED
    assert receipt.assessment_decision != AssessmentDecision.VERIFIED
    assert receipt.recommended_action in ActionRecommendation

    # Mandatory disclosures present
    assert len(receipt.limitations_and_disclosures) >= 4
    assert any("uncalibrated" in d.lower() for d in receipt.limitations_and_disclosures)
    assert any("audio" in d.lower() for d in receipt.limitations_and_disclosures)

    # Total latency recorded
    assert receipt.total_processing_latency_ms > 0


def test_pipeline_invalid_media_corrupted_file(
    yunet_path: Path, xception_path: Path, corrupted_media_file: Path
):
    """Corrupted media produces receipt with UNSUPPORTED_MEDIA or SYSTEM_ERROR."""
    pipeline = _make_pipeline(yunet_path, xception_path)
    receipt = pipeline.run(corrupted_media_file)

    assert isinstance(receipt, EvidenceReceipt)
    assert receipt.media_inspection.is_valid is False
    assert receipt.media_inspection.validation_error is not None

    # Visual and sync should be IDLE (skipped)
    assert receipt.visual_analysis.status == AnalyzerStatus.IDLE
    assert receipt.sync_analysis.status == AnalyzerStatus.IDLE

    # Uncertainty gate should flag corrupted media
    assert UncertaintyFlag.CORRUPTED_MEDIA in receipt.uncertainty_report.flags
    assert receipt.uncertainty_report.is_gate_closed is True

    # Decision should be SYSTEM_ERROR
    assert receipt.assessment_decision == AssessmentDecision.SYSTEM_ERROR
    assert receipt.recommended_action == ActionRecommendation.ABORT_UNSUPPORTED_MEDIA


def test_pipeline_invalid_media_unsupported_format(
    yunet_path: Path, xception_path: Path, unsupported_format_file: Path
):
    """Unsupported format produces UNSUPPORTED_MEDIA decision."""
    pipeline = _make_pipeline(yunet_path, xception_path)
    receipt = pipeline.run(unsupported_format_file)

    assert receipt.media_inspection.is_valid is False
    assert "Unsupported media container format" in receipt.media_inspection.validation_error

    assert UncertaintyFlag.UNSUPPORTED_CONTAINER in receipt.uncertainty_report.flags
    assert receipt.assessment_decision == AssessmentDecision.UNSUPPORTED_MEDIA
    assert receipt.recommended_action == ActionRecommendation.ABORT_UNSUPPORTED_MEDIA


def test_pipeline_invalid_media_empty_file(
    yunet_path: Path, xception_path: Path, empty_media_file: Path
):
    """Empty file produces SYSTEM_ERROR decision."""
    pipeline = _make_pipeline(yunet_path, xception_path)
    receipt = pipeline.run(empty_media_file)

    assert receipt.media_inspection.is_valid is False
    assert receipt.media_inspection.file_size_bytes == 0
    assert receipt.assessment_decision == AssessmentDecision.SYSTEM_ERROR


# ---------------------------------------------------------------------------
# SyncNet unavailable / audio cases
# ---------------------------------------------------------------------------

@patch("agni.pipeline.VisualDeepfakeDetector")
def test_pipeline_no_audio_track_reports_unavailable(
    mock_detector_cls, yunet_path: Path, xception_path: Path, valid_video_no_audio: Path, tmp_path: Path
):
    """Video without audio track produces NOT_AVAILABLE sync result."""
    mock_detector = MagicMock()
    mock_detector_cls.return_value = mock_detector
    mock_detector.analyze_frames.return_value = VisualAnalysisResult(
        analyzer_name="visual_deepfake_detector",
        status=AnalyzerStatus.COMPLETED,
        faces_detected_count=8,
        frames_evaluated_count=8,
        raw_score=0.1,
        is_score_calibrated=False,
    )

    pipeline = _make_pipeline(yunet_path, xception_path)
    receipt = pipeline.run(valid_video_no_audio, output_wav_dir=tmp_path)

    assert receipt.media_inspection.is_valid is True
    assert receipt.media_inspection.audio is not None
    assert receipt.media_inspection.audio.has_audio is False

    # Sync analysis should report no audio track
    assert receipt.sync_analysis.status == AnalyzerStatus.NOT_AVAILABLE
    assert receipt.sync_analysis.has_audio_track is False
    assert receipt.sync_analysis.audio_speech_detected is False
    assert "no audio track" in receipt.sync_analysis.error_message.lower()

    # Uncertainty gate should flag missing audio
    assert UncertaintyFlag.NO_AUDIO_TRACK in receipt.uncertainty_report.flags
    assert "audio_track" in receipt.uncertainty_report.missing_signals

    # Decision must be INCONCLUSIVE, never VERIFIED or REJECTED
    assert receipt.assessment_decision == AssessmentDecision.INCONCLUSIVE
    assert receipt.recommended_action == ActionRecommendation.RECAPTURE_WITH_AUDIO
    assert "Audio stream missing" in receipt.action_reason


@patch("agni.pipeline.VisualDeepfakeDetector")
def test_pipeline_silent_audio_reports_unavailable(
    mock_detector_cls, yunet_path: Path, xception_path: Path, valid_video_silent_audio: Path, tmp_path: Path
):
    """Video with silent audio produces NOT_AVAILABLE sync result."""
    mock_detector = MagicMock()
    mock_detector_cls.return_value = mock_detector
    mock_detector.analyze_frames.return_value = VisualAnalysisResult(
        analyzer_name="visual_deepfake_detector",
        status=AnalyzerStatus.COMPLETED,
        faces_detected_count=8,
        frames_evaluated_count=8,
        raw_score=0.1,
        is_score_calibrated=False,
    )

    pipeline = _make_pipeline(yunet_path, xception_path)
    receipt = pipeline.run(valid_video_silent_audio, output_wav_dir=tmp_path)

    assert receipt.sync_analysis.status == AnalyzerStatus.NOT_AVAILABLE
    assert receipt.sync_analysis.has_audio_track is True
    assert receipt.sync_analysis.audio_speech_detected is False
    assert receipt.sync_analysis.diagnostics.get("stage") == "silent_audio"

    assert UncertaintyFlag.AUDIO_SILENT in receipt.uncertainty_report.flags
    assert receipt.assessment_decision == AssessmentDecision.INCONCLUSIVE
    assert receipt.recommended_action == ActionRecommendation.RECAPTURE_WITH_AUDIO


# ---------------------------------------------------------------------------
# Visual detector error handling - with mocking (create pipeline inside test)
# ---------------------------------------------------------------------------

@patch("agni.pipeline.VisualDeepfakeDetector")
def test_pipeline_visual_detector_error_handling(
    mock_detector_cls, yunet_path: Path, xception_path: Path, valid_video_with_audio: Path, tmp_path: Path
):
    """Visual detector ERROR status propagates through pipeline."""
    mock_detector = MagicMock()
    mock_detector_cls.return_value = mock_detector
    mock_detector.analyze_frames.return_value = VisualAnalysisResult(
        analyzer_name="visual_deepfake_detector",
        status=AnalyzerStatus.ERROR,
        frames_evaluated_count=8,
        faces_detected_count=0,
        raw_score=None,
        is_score_calibrated=False,
        error_message="RuntimeError: CUDA out of memory",
    )

    pipeline = _make_pipeline(yunet_path, xception_path)
    receipt = pipeline.run(valid_video_with_audio, output_wav_dir=tmp_path)

    assert receipt.visual_analysis.status == AnalyzerStatus.ERROR
    assert "CUDA out of memory" in receipt.visual_analysis.error_message

    assert UncertaintyFlag.ANALYZER_FAILURE in receipt.uncertainty_report.flags
    assert receipt.assessment_decision == AssessmentDecision.SYSTEM_ERROR
    assert receipt.recommended_action == ActionRecommendation.MANUAL_REVIEW_REQUIRED


@patch("agni.pipeline.VisualDeepfakeDetector")
def test_pipeline_visual_no_faces_detected(
    mock_detector_cls, yunet_path: Path, xception_path: Path, valid_video_with_audio: Path, tmp_path: Path
):
    """Visual detector NOT_AVAILABLE (no faces) propagates correctly."""
    mock_detector = MagicMock()
    mock_detector_cls.return_value = mock_detector
    mock_detector.analyze_frames.return_value = VisualAnalysisResult(
        analyzer_name="visual_deepfake_detector",
        status=AnalyzerStatus.NOT_AVAILABLE,
        faces_detected_count=0,
        frames_evaluated_count=8,
        raw_score=None,
        is_score_calibrated=False,
        diagnostics={"reason": "no_faces_detected_in_frames"},
    )

    pipeline = _make_pipeline(yunet_path, xception_path)
    receipt = pipeline.run(valid_video_with_audio, output_wav_dir=tmp_path)

    assert receipt.visual_analysis.status == AnalyzerStatus.NOT_AVAILABLE
    assert receipt.visual_analysis.faces_detected_count == 0
    assert receipt.visual_analysis.raw_score is None

    assert UncertaintyFlag.NO_FACES_DETECTED in receipt.uncertainty_report.flags
    assert receipt.assessment_decision == AssessmentDecision.INCONCLUSIVE
    assert receipt.recommended_action == ActionRecommendation.RECAPTURE_CLEAR_FACE


@patch("agni.pipeline.VisualDeepfakeDetector")
def test_pipeline_visual_partial_face_evidence(
    mock_detector_cls, yunet_path: Path, xception_path: Path, valid_video_with_audio: Path, tmp_path: Path
):
    """Visual detector with insufficient face frames flags PARTIAL_FACE_EVIDENCE."""
    mock_detector = MagicMock()
    mock_detector_cls.return_value = mock_detector
    mock_detector.analyze_frames.return_value = VisualAnalysisResult(
        analyzer_name="visual_deepfake_detector",
        status=AnalyzerStatus.COMPLETED,
        faces_detected_count=2,
        frames_evaluated_count=8,
        raw_score=0.3,
        is_score_calibrated=False,
        decision_threshold=0.5,
        diagnostics={"face_detection_rate": 0.25},
    )

    pipeline = _make_pipeline(yunet_path, xception_path)
    receipt = pipeline.run(valid_video_with_audio, output_wav_dir=tmp_path)

    assert receipt.visual_analysis.faces_detected_count == 2
    assert UncertaintyFlag.PARTIAL_FACE_EVIDENCE in receipt.uncertainty_report.flags
    assert receipt.assessment_decision == AssessmentDecision.INCONCLUSIVE


# ---------------------------------------------------------------------------
# Receipt generation and structure
# ---------------------------------------------------------------------------

def test_pipeline_receipt_contains_all_required_fields(
    yunet_path: Path, xception_path: Path, valid_video_with_audio: Path, tmp_path: Path
):
    """Receipt has all mandatory fields populated."""
    pipeline = _make_pipeline(yunet_path, xception_path)
    receipt = pipeline.run(valid_video_with_audio, session_id="sess-123", output_wav_dir=tmp_path)

    # Identity
    assert receipt.receipt_id is not None
    assert len(receipt.receipt_id) == 36  # UUIDv4
    assert receipt.created_at_utc is not None
    assert "T" in receipt.created_at_utc  # ISO 8601
    assert receipt.session_id == "sess-123"
    assert receipt.system_version == "AGNI-0.1.0"

    # Telemetry
    assert receipt.media_inspection is not None
    assert receipt.visual_analysis is not None
    assert receipt.sync_analysis is not None
    assert receipt.uncertainty_report is not None

    # Decisioning
    assert receipt.assessment_decision in AssessmentDecision
    assert receipt.recommended_action in ActionRecommendation
    assert receipt.action_reason is not None
    assert len(receipt.action_reason) > 0

    # Disclosures
    assert isinstance(receipt.limitations_and_disclosures, list)
    assert len(receipt.limitations_and_disclosures) >= 4

    # Timing
    assert receipt.total_processing_latency_ms > 0


def test_pipeline_visual_only_mode(
    yunet_path: Path, xception_path: Path, valid_video_with_audio: Path
):
    """run_visual_only skips sync analysis and marks it appropriately."""
    pipeline = _make_pipeline(yunet_path, xception_path)
    receipt = pipeline.run_visual_only(valid_video_with_audio, session_id="visual-only")

    assert receipt.session_id == "visual-only"
    assert receipt.media_inspection.is_valid is True

    # Visual analysis should run
    assert receipt.visual_analysis.frames_evaluated_count > 0

    # Sync analysis should be NOT_AVAILABLE with not_requested stage
    assert receipt.sync_analysis.status == AnalyzerStatus.NOT_AVAILABLE
    assert receipt.sync_analysis.error_message == "Audio-video synchronization analysis not requested"
    assert receipt.sync_analysis.diagnostics.get("stage") == "not_requested"

    # Uncertainty gate should still process (will flag missing audio)
    assert receipt.assessment_decision == AssessmentDecision.INCONCLUSIVE


# ---------------------------------------------------------------------------
# Critical rule: Missing evidence cannot yield VERIFIED
# ---------------------------------------------------------------------------

@patch("agni.pipeline.VisualDeepfakeDetector")
def test_pipeline_missing_audio_never_verified(
    mock_detector_cls, yunet_path: Path, xception_path: Path, valid_video_no_audio: Path, tmp_path: Path
):
    """Even with perfect visual score, missing audio prevents VERIFIED decision."""
    mock_detector = MagicMock()
    mock_detector_cls.return_value = mock_detector
    mock_detector.analyze_frames.return_value = VisualAnalysisResult(
        analyzer_name="visual_deepfake_detector",
        status=AnalyzerStatus.COMPLETED,
        faces_detected_count=8,
        frames_evaluated_count=8,
        raw_score=0.01,  # Very low fake score (appears real)
        is_score_calibrated=False,
        decision_threshold=0.5,
        diagnostics={"face_detection_rate": 1.0},
    )

    pipeline = _make_pipeline(yunet_path, xception_path)
    receipt = pipeline.run(valid_video_no_audio, output_wav_dir=tmp_path)

    # Visual score is low (appears real) but audio is missing
    assert receipt.visual_analysis.raw_score == 0.01
    assert receipt.visual_analysis.is_score_calibrated is False

    # Decision MUST NOT be VERIFIED
    assert receipt.assessment_decision != AssessmentDecision.VERIFIED
    assert receipt.assessment_decision != AssessmentDecision.REJECTED
    assert receipt.assessment_decision == AssessmentDecision.INCONCLUSIVE

    # Must recommend recapture with audio
    assert receipt.recommended_action == ActionRecommendation.RECAPTURE_WITH_AUDIO


@patch("agni.pipeline.VisualDeepfakeDetector")
def test_pipeline_missing_syncnet_never_verified(
    mock_detector_cls, yunet_path: Path, xception_path: Path, valid_video_with_audio: Path, tmp_path: Path
):
    """Even with perfect visual score and audio, missing SyncNet prevents VERIFIED."""
    mock_detector = MagicMock()
    mock_detector_cls.return_value = mock_detector
    mock_detector.analyze_frames.return_value = VisualAnalysisResult(
        analyzer_name="visual_deepfake_detector",
        status=AnalyzerStatus.COMPLETED,
        faces_detected_count=8,
        frames_evaluated_count=8,
        raw_score=0.01,
        is_score_calibrated=False,
        decision_threshold=0.5,
    )

    pipeline = _make_pipeline(yunet_path, xception_path)
    receipt = pipeline.run(valid_video_with_audio, output_wav_dir=tmp_path)

    # Audio present, visual score low, but SyncNet unavailable
    assert receipt.sync_analysis.has_audio_track is True
    assert receipt.sync_analysis.audio_speech_detected is True
    assert receipt.sync_analysis.status == AnalyzerStatus.NOT_AVAILABLE
    assert receipt.sync_analysis.temporal_offset_ms is None

    # Decision MUST NOT be VERIFIED
    assert receipt.assessment_decision != AssessmentDecision.VERIFIED
    assert receipt.assessment_decision != AssessmentDecision.REJECTED
    assert receipt.assessment_decision == AssessmentDecision.INCONCLUSIVE


@patch("agni.pipeline.VisualDeepfakeDetector")
def test_pipeline_visual_not_available_never_verified(
    mock_detector_cls, yunet_path: Path, xception_path: Path, valid_video_with_audio: Path, tmp_path: Path
):
    """Visual analyzer NOT_AVAILABLE (no faces) prevents VERIFIED regardless of audio."""
    mock_detector = MagicMock()
    mock_detector_cls.return_value = mock_detector
    mock_detector.analyze_frames.return_value = VisualAnalysisResult(
        analyzer_name="visual_deepfake_detector",
        status=AnalyzerStatus.NOT_AVAILABLE,
        faces_detected_count=0,
        frames_evaluated_count=8,
        raw_score=None,
        is_score_calibrated=False,
        diagnostics={"reason": "no_faces_detected_in_frames"},
    )

    pipeline = _make_pipeline(yunet_path, xception_path)
    receipt = pipeline.run(valid_video_with_audio, output_wav_dir=tmp_path)

    assert receipt.visual_analysis.status == AnalyzerStatus.NOT_AVAILABLE
    assert receipt.visual_analysis.raw_score is None

    # Decision MUST NOT be VERIFIED
    assert receipt.assessment_decision != AssessmentDecision.VERIFIED
    assert receipt.assessment_decision == AssessmentDecision.INCONCLUSIVE
    assert receipt.recommended_action == ActionRecommendation.RECAPTURE_CLEAR_FACE


# ---------------------------------------------------------------------------
# Short duration handling
# ---------------------------------------------------------------------------

@patch("agni.pipeline.VisualDeepfakeDetector")
@patch("agni.pipeline.SyncAnalyzer")
@patch("agni.pipeline.MediaValidator")
def test_pipeline_short_duration_video(
    mock_validator_cls, mock_sync_cls, mock_detector_cls, yunet_path: Path, xception_path: Path, short_clip_video: Path, tmp_path: Path
):
    """Sub-0.5s clips trigger SHORT_DURATION uncertainty."""
    mock_detector = MagicMock()
    mock_detector_cls.return_value = mock_detector
    mock_detector.analyze_frames.return_value = VisualAnalysisResult(
        analyzer_name="visual_deepfake_detector",
        status=AnalyzerStatus.COMPLETED,
        faces_detected_count=3,
        frames_evaluated_count=3,
        raw_score=0.1,
        is_score_calibrated=False,
    )

    # Mock sync analyzer to return audio features extracted but SyncNet unavailable
    # (so AUDIO_SILENT doesn't trigger before SHORT_DURATION)
    mock_sync = MagicMock()
    mock_sync_cls.return_value = mock_sync
    mock_sync.analyze.return_value = SyncAnalysisResult(
        analyzer_name="av_sync_detector",
        status=AnalyzerStatus.NOT_AVAILABLE,
        has_audio_track=True,
        audio_speech_detected=True,
        diagnostics={"stage": "mfcc_extracted_syncnet_unavailable", "syncnet_available": False},
    )

    # Mock media validator to return valid media with audio (so SHORT_DURATION can trigger)
    mock_validator = MagicMock()
    mock_validator_cls.return_value = mock_validator
    mock_validator.inspect_media.return_value = MediaInspectionResult(
        file_path=str(short_clip_video),
        file_name=short_clip_video.name,
        file_size_bytes=1000,
        is_valid=True,
        media_type=MediaType.VIDEO,
        validation_error=None,
        video=VideoStreamMetadata(
            width=160,
            height=120,
            fps=25.0,
            frame_count=3,
            duration_sec=0.12,
            codec="mp4v",
        ),
        audio=AudioStreamMetadata(
            has_audio=True,
            sample_rate=16000,
            channels=1,
            duration_sec=0.12,
            is_silent=False,
            codec="aac",
        ),
        inspection_latency_ms=10.0,
    )

    pipeline = _make_pipeline(yunet_path, xception_path)
    receipt = pipeline.run(short_clip_video, output_wav_dir=tmp_path)

    assert UncertaintyFlag.SHORT_DURATION in receipt.uncertainty_report.flags
    assert "sufficient_temporal_frames" in receipt.uncertainty_report.missing_signals
    assert receipt.assessment_decision == AssessmentDecision.INCONCLUSIVE
    assert receipt.recommended_action == ActionRecommendation.MANUAL_REVIEW_REQUIRED
    assert "duration too short" in receipt.action_reason.lower()


# ---------------------------------------------------------------------------
# Sync analyzer error handling - with mocking
# ---------------------------------------------------------------------------

@patch("agni.pipeline.VisualDeepfakeDetector")
@patch("agni.pipeline.SyncAnalyzer")
def test_pipeline_sync_analyzer_error_propagates(
    mock_sync_cls, mock_detector_cls, yunet_path: Path, xception_path: Path, valid_video_with_audio: Path, tmp_path: Path
):
    """Sync analyzer ERROR status propagates through pipeline."""
    mock_detector = MagicMock()
    mock_detector_cls.return_value = mock_detector
    mock_detector.analyze_frames.return_value = VisualAnalysisResult(
        analyzer_name="visual_deepfake_detector",
        status=AnalyzerStatus.COMPLETED,
        faces_detected_count=8,
        frames_evaluated_count=8,
        raw_score=0.2,
        is_score_calibrated=False,
    )

    mock_sync = MagicMock()
    mock_sync_cls.return_value = mock_sync
    mock_sync.analyze.return_value = SyncAnalysisResult(
        analyzer_name="av_sync_detector",
        status=AnalyzerStatus.ERROR,
        has_audio_track=True,
        audio_speech_detected=True,
        error_message="Audio extraction failed: ffmpeg error",
        diagnostics={"stage": "extraction_failed"},
    )

    pipeline = _make_pipeline(yunet_path, xception_path)
    receipt = pipeline.run(valid_video_with_audio, output_wav_dir=tmp_path)

    assert receipt.sync_analysis.status == AnalyzerStatus.ERROR
    assert "extraction failed" in receipt.sync_analysis.error_message.lower()

    assert UncertaintyFlag.ANALYZER_FAILURE in receipt.uncertainty_report.flags
    assert receipt.assessment_decision == AssessmentDecision.SYSTEM_ERROR
    assert receipt.recommended_action == ActionRecommendation.MANUAL_REVIEW_REQUIRED


# ---------------------------------------------------------------------------
# Model path validation
# ---------------------------------------------------------------------------

def test_pipeline_invalid_yunet_path_raises(xception_path: Path):
    """Constructor raises FileNotFoundError for missing YuNet model."""
    with pytest.raises(FileNotFoundError, match="YuNet ONNX model missing"):
        AGNIPipeline(
            yunet_model_path="non_existent_yunet.onnx",
            xception_checkpoint_path=xception_path,
        )


def test_pipeline_invalid_xception_path_raises(yunet_path: Path):
    """Constructor raises FileNotFoundError for missing Xception checkpoint."""
    with pytest.raises(FileNotFoundError, match="Xception checkpoint missing"):
        AGNIPipeline(
            yunet_model_path=yunet_path,
            xception_checkpoint_path="non_existent_xception.pth",
        )


# ---------------------------------------------------------------------------
# Receipt disclosures verify mandatory content
# ---------------------------------------------------------------------------

def test_receipt_mandatory_disclosures_content(
    yunet_path: Path, xception_path: Path, valid_video_with_audio: Path, tmp_path: Path
):
    """Receipt contains all required scientific disclosures."""
    pipeline = _make_pipeline(yunet_path, xception_path)
    receipt = pipeline.run(valid_video_with_audio, output_wav_dir=tmp_path)

    disclosures = receipt.limitations_and_disclosures

    # Check for key mandatory disclosures
    text = " ".join(disclosures).lower()
    assert "uncalibrated" in text
    assert "probability" in text or "probabilities" in text
    assert "audio" in text
    assert "lip sync" in text or "synchronization" in text
    assert "identity" in text
    assert "liveness" in text or "blinking" in text
    assert "prototype" in text
    assert "production" in text or "clearance" in text
    assert "evaluation" in text or "audit" in text