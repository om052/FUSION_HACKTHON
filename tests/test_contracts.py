"""Unit tests verifying AGNI data contracts and schema invariants."""

import json
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
from agni.contracts.uncertainty import UncertaintyReport
from agni.contracts.receipt import EvidenceReceipt
from agni.receipt.generator import ReceiptGenerator


def test_enum_distinctness_invariants():
    """Confirms that distinct operational statuses cannot be confused."""
    statuses = [
        AnalyzerStatus.IDLE,
        AnalyzerStatus.COMPLETED,
        AnalyzerStatus.NOT_AVAILABLE,
        AnalyzerStatus.INCONCLUSIVE,
        AnalyzerStatus.UNSUPPORTED,
        AnalyzerStatus.ERROR,
    ]
    # Invariant: Every enum value must be unique
    assert len(statuses) == len(set(statuses))
    assert AnalyzerStatus.NOT_AVAILABLE != AnalyzerStatus.INCONCLUSIVE
    assert AnalyzerStatus.UNSUPPORTED != AnalyzerStatus.ERROR
    assert AnalyzerStatus.NOT_AVAILABLE != AnalyzerStatus.ERROR


def test_visual_analysis_contract_defaults():
    """Confirms that uncalibrated scores are explicitly tracked."""
    result = VisualAnalysisResult(
        model_architecture="Xception",
        model_version="ffpp_c23",
        status=AnalyzerStatus.COMPLETED,
        raw_score=0.74,
    )
    assert result.is_score_calibrated is False
    assert result.raw_score == 0.74
    assert result.faces_detected_count == 0


def test_sync_analysis_contract_defaults():
    """Confirms default state for synchronization analyzer contract."""
    result = SyncAnalysisResult(
        status=AnalyzerStatus.NOT_AVAILABLE,
        has_audio_track=False,
    )
    assert result.is_score_calibrated is False
    assert result.temporal_offset_frames is None
    assert result.confidence_score is None


def test_evidence_receipt_serialization_round_trip():
    """Ensures Evidence Receipt serializes to JSON and deserializes losslessly."""
    media_meta = MediaInspectionResult(
        file_path="C:/fake/sample.mp4",
        file_name="sample.mp4",
        file_size_bytes=10240,
        media_type=MediaType.VIDEO,
        is_valid=True,
        video=VideoStreamMetadata(
            width=1280,
            height=720,
            fps=30.0,
            frame_count=90,
            duration_sec=3.0,
        ),
        audio=AudioStreamMetadata(
            has_audio=True,
            sample_rate=16000,
            channels=1,
            duration_sec=3.0,
            is_silent=False,
        ),
    )

    visual = VisualAnalysisResult(
        model_architecture="Xception",
        model_version="v1",
        status=AnalyzerStatus.COMPLETED,
        faces_detected_count=30,
        frames_evaluated_count=30,
        raw_score=0.12,
        is_score_calibrated=False,
    )

    sync = SyncAnalysisResult(
        model_architecture="SyncNet",
        model_version="v2",
        status=AnalyzerStatus.COMPLETED,
        has_audio_track=True,
        mouth_rois_extracted_count=30,
        audio_speech_detected=True,
        temporal_offset_frames=0.0,
        temporal_offset_ms=0.0,
        confidence_score=4.5,
        is_score_calibrated=False,
    )

    uncertainty = UncertaintyReport(
        flags=[],
        missing_signals=[],
        conflicting_signals=False,
        is_gate_closed=False,
        summary_reason="All signals present and evaluated.",
    )

    receipt = ReceiptGenerator.generate(
        media_inspection=media_meta,
        visual_analysis=visual,
        sync_analysis=sync,
        uncertainty_report=uncertainty,
        assessment_decision=AssessmentDecision.VERIFIED,
        recommended_action=ActionRecommendation.PROCEED_ONBOARDING,
        action_reason="Multi-modal evidence corroborates consistent identity.",
        session_id="session-1234",
        total_latency_ms=150.2,
    )

    receipt_json = receipt.model_dump_json()
    assert "session-1234" in receipt_json
    assert "AGNI-0.1.0" in receipt_json

    # Test round trip parsing
    parsed_receipt = EvidenceReceipt.model_validate_json(receipt_json)
    assert parsed_receipt.receipt_id == receipt.receipt_id
    assert parsed_receipt.assessment_decision == AssessmentDecision.VERIFIED
    assert len(parsed_receipt.limitations_and_disclosures) >= 5
