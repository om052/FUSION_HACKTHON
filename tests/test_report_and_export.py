"""Tests for the case report, final-assessment rendering, and
temporary-upload cleanup.

These exercise the privacy-conscious report builder, the
uncertainty-gate final assessment display, and the guaranteed
cleanup of untrusted uploaded media.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agni.contracts.enums import (
    ActionRecommendation,
    AnalyzerStatus,
    AssessmentDecision,
    MediaType,
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
from agni.uncertainty.gate import UncertaintyGate
from agni.dashboard_helpers import (
    build_report_dict,
    decision_headline,
    render_final_assessment,
    render_report_export,
)
from agni.intake.upload import temporary_upload


def _make_receipt(
    decision: AssessmentDecision = AssessmentDecision.SUSPICIOUS,
    action: ActionRecommendation = ActionRecommendation.MANUAL_REVIEW_REQUIRED,
    raw_score: float = 0.75,
    has_audio: bool = True,
) -> EvidenceReceipt:
    """Build a real receipt through the gate + receipt generator."""
    mi = MediaInspectionResult(
        file_path="/tmp/agni_upload_abc123.mp4",
        file_name="private_video.mp4",
        file_size_bytes=12345,
        media_type=MediaType.VIDEO,
        is_valid=True,
        video=VideoStreamMetadata(
            width=640, height=480, fps=30.0, frame_count=60, duration_sec=2.0
        ),
        audio=AudioStreamMetadata(
            has_audio=has_audio, is_silent=False, sample_rate=16000
        ),
    )
    visual = VisualAnalysisResult(
        status=AnalyzerStatus.COMPLETED,
        faces_detected_count=8,
        frames_evaluated_count=8,
        raw_score=raw_score,
        decision_threshold=0.5,
        is_score_calibrated=False,
        diagnostics={
            "per_frame_fake_scores": [0.7, 0.8],
            "face_detection_rate": 1.0,
        },
    )
    sync = SyncAnalysisResult(
        status=AnalyzerStatus.NOT_AVAILABLE,
        has_audio_track=has_audio,
        audio_speech_detected=has_audio,
    )
    gate = UncertaintyGate()
    report, dec, act, reason = gate.evaluate(mi, visual, sync)
    return ReceiptGenerator.generate(
        media_inspection=mi,
        visual_analysis=visual,
        sync_analysis=sync,
        uncertainty_report=report,
        assessment_decision=dec,
        recommended_action=act,
        action_reason=reason,
        total_latency_ms=1234.5,
    )


class TestDecisionHeadline:
    def test_suspicious_headline(self):
        assert decision_headline(AssessmentDecision.SUSPICIOUS) == (
            "POTENTIAL MANIPULATION SIGNAL"
        )

    def test_inconclusive_headline(self):
        assert decision_headline(AssessmentDecision.INCONCLUSIVE) == (
            "INCONCLUSIVE"
        )

    def test_system_error_headline(self):
        assert decision_headline(AssessmentDecision.SYSTEM_ERROR) == (
            "SYSTEM ERROR"
        )

    def test_unsupported_media_headline(self):
        assert decision_headline(AssessmentDecision.UNSUPPORTED_MEDIA) == (
            "UNSUPPORTED MEDIA"
        )

    def test_defensive_fallback_for_unexpected_decision(self):
        # VERIFIED/REJECTED are never produced; a defensive
        # inconclusive headline is returned instead.
        assert decision_headline(AssessmentDecision.VERIFIED) == "INCONCLUSIVE"


class TestBuildReportDict:
    def test_redacts_temporary_path_keeps_filename(self):
        receipt = _make_receipt()
        report = build_report_dict(receipt)
        media = report["media_inspection"]
        assert media["file_path"] == "<redacted temporary path>"
        assert media["file_name"] == "private_video.mp4"

    def test_drops_per_frame_scores(self):
        receipt = _make_receipt()
        report = build_report_dict(receipt)
        diag = report["visual_analysis"]["diagnostics"]
        assert "per_frame_fake_scores" not in diag
        # Aggregate diagnostics are retained.
        assert diag.get("face_detection_rate") == 1.0

    def test_preserves_calibration_status(self):
        receipt = _make_receipt()
        report = build_report_dict(receipt)
        assert report["visual_analysis"]["is_score_calibrated"] is False

    def test_json_serializable(self):
        receipt = _make_receipt()
        report = build_report_dict(receipt)
        # Must not raise.
        json.dumps(report)

    def test_contains_decision_and_action(self):
        receipt = _make_receipt()
        report = build_report_dict(receipt)
        assert report["assessment_decision"] == "SUSPICIOUS"
        assert report["recommended_action"] == "MANUAL_REVIEW_REQUIRED"

    def test_contains_limitations(self):
        receipt = _make_receipt()
        report = build_report_dict(receipt)
        assert isinstance(report["limitations_and_disclosures"], list)
        assert len(report["limitations_and_disclosures"]) >= 4


class TestRenderFinalAssessment:
    def test_renders_suspicious_headline_and_action(self):
        receipt = _make_receipt()
        with patch("agni.dashboard_helpers.st") as mock_st:
            mock_st.container = MagicMock()
            # The container context manager yields a mock.
            mock_st.container.return_value.__enter__ = MagicMock(
                return_value=MagicMock()
            )
            mock_st.container.return_value.__exit__ = MagicMock(
                return_value=False
            )
            render_final_assessment(receipt)
            # The POTENTIAL MANIPULATION SIGNAL decision is rendered in red
            # (error) because it is a serious manipulation signal, not an
            # informational/benign outcome.
            mock_st.error.assert_called()
            error_text = mock_st.error.call_args[0][0]
            assert "POTENTIAL MANIPULATION SIGNAL" in error_text
            # Recommended action is written (st.write may receive
            # several positional args per call; capture them all).
            written_parts = []
            for call in mock_st.write.call_args_list:
                args = call.args if hasattr(call, "args") else call[0]
                written_parts.extend(str(a) for a in args)
            written = " ".join(written_parts)
            assert "Manual Review Required" in written

    def test_inconclusive_renders_as_amber_not_authentic(self):
        """An INCONCLUSIVE decision renders in amber (warning), never green/success,
        so a low uncalibrated score is never shown as an authentic-positive."""
        # A low uncalibrated score (no strong signal) with complete-enough
        # evidence yields INCONCLUSIVE via the gate (vestigial decision=
        # param is ignored by _make_receipt; the gate computes the real one).
        receipt = _make_receipt(raw_score=0.10)
        assert receipt.assessment_decision == AssessmentDecision.INCONCLUSIVE
        with patch("agni.dashboard_helpers.st") as mock_st:
            mock_st.container = MagicMock()
            mock_st.container.return_value.__enter__ = MagicMock(
                return_value=MagicMock()
            )
            mock_st.container.return_value.__exit__ = MagicMock(
                return_value=False
            )
            mock_st.success = MagicMock()
            render_final_assessment(receipt)
            mock_st.warning.assert_called()
            warn_text = mock_st.warning.call_args[0][0]
            assert "INCONCLUSIVE" in warn_text
            # No green/success may imply an authenticity claim.
            mock_st.success.assert_not_called()


class TestRenderReportExport:
    def test_download_button_renders_with_json(self):
        receipt = _make_receipt()
        with patch("agni.dashboard_helpers.st") as mock_st:
            render_report_export(receipt)
            mock_st.download_button.assert_called_once()
            kwargs = mock_st.download_button.call_args.kwargs
            assert kwargs["mime"] == "application/json"
            # The payload must be valid JSON containing the decision.
            payload = json.loads(kwargs["data"])
            assert payload["assessment_decision"] == "SUSPICIOUS"


class TestGateDashboardConsistency:
    """The dashboard label and the gate decision must agree."""

    def test_strong_signal_agrees_between_gate_and_dashboard(self):
        from agni.dashboard_helpers import assess_visual_signal
        from agni.uncertainty.gate import classify_visual_signal

        visual = VisualAnalysisResult(
            status=AnalyzerStatus.COMPLETED,
            faces_detected_count=8,
            raw_score=0.75,
            decision_threshold=0.5,
        )
        # Dashboard label
        label = assess_visual_signal(visual)
        # Gate band
        band = classify_visual_signal(visual.raw_score, visual.decision_threshold)
        assert label == "Potential manipulation signal"
        assert band == "potential_manipulation_signal"

    def test_weak_signal_agrees_between_gate_and_dashboard(self):
        from agni.dashboard_helpers import assess_visual_signal
        from agni.uncertainty.gate import classify_visual_signal

        visual = VisualAnalysisResult(
            status=AnalyzerStatus.COMPLETED,
            faces_detected_count=8,
            raw_score=0.10,
            decision_threshold=0.5,
        )
        assert assess_visual_signal(visual) == "No strong signal from this detector"
        assert (
            classify_visual_signal(visual.raw_score, visual.decision_threshold)
            == "no_strong_visual_signal"
        )


class TestTemporaryUpload:
    def test_writes_and_cleans_up(self):
        payload = b"\x00\x00\x00\x18ftypisom" + b"fakebytes"
        seen_path = None
        with temporary_upload(payload, suffix=".mp4") as path:
            seen_path = path
            assert path.exists()
            assert path.suffix == ".mp4"
            assert path.read_bytes() == payload
            # Must live in the system temp dir, not the project tree.
            assert str(path).startswith(str(Path(tempfile.gettempdir())))
        # After the context exits the file is gone.
        assert not seen_path.exists()

    def test_cleans_up_on_exception(self):
        payload = b"not-a-video"
        seen_path = None
        with pytest.raises(RuntimeError):
            with temporary_upload(payload, suffix=".mp4") as path:
                seen_path = path
                assert path.exists()
                raise RuntimeError("consumer failed")
        assert seen_path is not None
        assert not seen_path.exists()

    def test_randomized_names_do_not_collide(self):
        paths = []
        for _ in range(3):
            with temporary_upload(b"x", suffix=".mp4") as path:
                paths.append(path)
        assert len({str(p) for p in paths}) == 3
