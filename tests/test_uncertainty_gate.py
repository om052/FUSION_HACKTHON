"""Unit tests verifying AGNI's Uncertainty Gate rules and epistemic boundaries."""

from pathlib import Path
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
from agni.uncertainty.gate import UncertaintyGate


def test_missing_audio_does_not_forge_decision():
    """CRITICAL RULE: Missing audio must NOT count as evidence of genuine or fake."""
    gate = UncertaintyGate()

    media_inspection = MediaInspectionResult(
        file_path="C:/test/no_audio.mp4",
        file_name="no_audio.mp4",
        file_size_bytes=1000,
        media_type=MediaType.VIDEO,
        is_valid=True,
        video=VideoStreamMetadata(
            width=640, height=480, fps=30.0, frame_count=60, duration_sec=2.0
        ),
        audio=AudioStreamMetadata(has_audio=False),
    )

    visual = VisualAnalysisResult(
        status=AnalyzerStatus.COMPLETED,
        faces_detected_count=30,
        frames_evaluated_count=30,
        raw_score=0.10,  # Even if visual model score is low
    )

    sync = SyncAnalysisResult(
        status=AnalyzerStatus.NOT_AVAILABLE,
        has_audio_track=False,
    )

    report, decision, action, reason = gate.evaluate(media_inspection, visual, sync)

    # Uncertainty gate must intercept missing audio
    assert UncertaintyFlag.NO_AUDIO_TRACK in report.flags
    assert "audio_track" in report.missing_signals
    assert report.is_gate_closed is True

    # MUST NOT be VERIFIED and MUST NOT be REJECTED
    assert decision == AssessmentDecision.INCONCLUSIVE
    assert action == ActionRecommendation.RECAPTURE_WITH_AUDIO
    assert "Audio stream missing" in reason


def test_silent_audio_triggers_recapture():
    """Silent audio stream must be detected and trigger recapture."""
    gate = UncertaintyGate()

    media_inspection = MediaInspectionResult(
        file_path="C:/test/silent_audio.mp4",
        file_name="silent_audio.mp4",
        file_size_bytes=1000,
        media_type=MediaType.VIDEO,
        is_valid=True,
        video=VideoStreamMetadata(
            width=640, height=480, fps=30.0, frame_count=60, duration_sec=2.0
        ),
        audio=AudioStreamMetadata(has_audio=True, is_silent=True, sample_rate=16000),
    )

    visual = VisualAnalysisResult(status=AnalyzerStatus.COMPLETED, faces_detected_count=10)
    sync = SyncAnalysisResult(status=AnalyzerStatus.NOT_AVAILABLE, has_audio_track=True)

    report, decision, action, _ = gate.evaluate(media_inspection, visual, sync)

    assert UncertaintyFlag.AUDIO_SILENT in report.flags
    assert decision == AssessmentDecision.INCONCLUSIVE
    assert action == ActionRecommendation.RECAPTURE_WITH_AUDIO


def test_no_face_detected_triggers_recapture():
    """If no human face is detected in video frames, cannot conclude."""
    gate = UncertaintyGate()

    media_inspection = MediaInspectionResult(
        file_path="C:/test/no_face.mp4",
        file_name="no_face.mp4",
        file_size_bytes=1000,
        media_type=MediaType.VIDEO,
        is_valid=True,
        video=VideoStreamMetadata(
            width=640, height=480, fps=30.0, frame_count=60, duration_sec=2.0
        ),
        audio=AudioStreamMetadata(has_audio=True, is_silent=False, sample_rate=16000),
    )

    visual = VisualAnalysisResult(
        status=AnalyzerStatus.COMPLETED,
        faces_detected_count=0,  # Zero faces found
        frames_evaluated_count=30,
    )
    # Fixture isolates the no-face rule: declared non-silent audio must
    # carry detected speech so the AUDIO_SILENT branch is not spuriously
    # flagged alongside NO_FACES_DETECTED.
    sync = SyncAnalysisResult(
        status=AnalyzerStatus.COMPLETED,
        has_audio_track=True,
        audio_speech_detected=True,
    )

    report, decision, action, _ = gate.evaluate(media_inspection, visual, sync)

    assert UncertaintyFlag.NO_FACES_DETECTED in report.flags
    assert decision == AssessmentDecision.INCONCLUSIVE
    assert action == ActionRecommendation.RECAPTURE_CLEAR_FACE


def test_short_clip_triggers_inconclusive():
    """Sub-0.5s clips must trigger short duration uncertainty."""
    gate = UncertaintyGate()

    media_inspection = MediaInspectionResult(
        file_path="C:/test/short.mp4",
        file_name="short.mp4",
        file_size_bytes=500,
        media_type=MediaType.VIDEO,
        is_valid=True,
        video=VideoStreamMetadata(
            width=640, height=480, fps=30.0, frame_count=6, duration_sec=0.2
        ),
        audio=AudioStreamMetadata(has_audio=True, is_silent=False, sample_rate=16000),
    )

    visual = VisualAnalysisResult(status=AnalyzerStatus.COMPLETED, faces_detected_count=5)
    # Fixture isolates the short-duration rule: the declared non-silent
    # audio track must carry detected speech, otherwise the gate's
    # AUDIO_SILENT branch (which takes precedence) would mask the rule
    # under test.
    sync = SyncAnalysisResult(
        status=AnalyzerStatus.COMPLETED,
        has_audio_track=True,
        audio_speech_detected=True,
    )

    report, decision, action, _ = gate.evaluate(media_inspection, visual, sync)

    assert UncertaintyFlag.SHORT_DURATION in report.flags
    assert decision == AssessmentDecision.INCONCLUSIVE
    assert action == ActionRecommendation.MANUAL_REVIEW_REQUIRED


def test_corrupted_file_handling():
    """Corrupted files must result in clear SYSTEM_ERROR without simulation."""
    gate = UncertaintyGate()

    media_inspection = MediaInspectionResult(
        file_path="C:/test/bad.mp4",
        file_name="bad.mp4",
        file_size_bytes=100,
        media_type=MediaType.VIDEO,
        is_valid=False,
        validation_error="Corrupted or invalid media container: Invalid data found",
    )

    visual = VisualAnalysisResult(status=AnalyzerStatus.IDLE)
    sync = SyncAnalysisResult(status=AnalyzerStatus.IDLE)

    report, decision, action, _ = gate.evaluate(media_inspection, visual, sync)

    assert UncertaintyFlag.CORRUPTED_MEDIA in report.flags
    assert decision == AssessmentDecision.SYSTEM_ERROR
    assert action == ActionRecommendation.ABORT_UNSUPPORTED_MEDIA


def test_unsupported_format_handling():
    """Unsupported formats must be distinctly marked UNSUPPORTED_MEDIA."""
    gate = UncertaintyGate()

    media_inspection = MediaInspectionResult(
        file_path="C:/test/doc.xyz",
        file_name="doc.xyz",
        file_size_bytes=100,
        media_type=MediaType.UNKNOWN,
        is_valid=False,
        validation_error="Unsupported media container format: '.xyz'",
    )

    visual = VisualAnalysisResult(status=AnalyzerStatus.IDLE)
    sync = SyncAnalysisResult(status=AnalyzerStatus.IDLE)

    report, decision, action, _ = gate.evaluate(media_inspection, visual, sync)

    assert UncertaintyFlag.UNSUPPORTED_CONTAINER in report.flags
    assert decision == AssessmentDecision.UNSUPPORTED_MEDIA
    assert action == ActionRecommendation.ABORT_UNSUPPORTED_MEDIA


def test_analyzer_error_handling():
    """Model execution failures must produce distinct ERROR status."""
    gate = UncertaintyGate()

    media_inspection = MediaInspectionResult(
        file_path="C:/test/valid.mp4",
        file_name="valid.mp4",
        file_size_bytes=1000,
        media_type=MediaType.VIDEO,
        is_valid=True,
        video=VideoStreamMetadata(
            width=640, height=480, fps=30.0, frame_count=60, duration_sec=2.0
        ),
        audio=AudioStreamMetadata(has_audio=True, is_silent=False, sample_rate=16000),
    )

    visual = VisualAnalysisResult(
        status=AnalyzerStatus.ERROR,
        error_message="RuntimeError: CUDA out of memory or CPU fault",
    )
    sync = SyncAnalysisResult(status=AnalyzerStatus.IDLE)

    report, decision, action, _ = gate.evaluate(media_inspection, visual, sync)

    assert UncertaintyFlag.ANALYZER_FAILURE in report.flags
    assert decision == AssessmentDecision.SYSTEM_ERROR
    assert action == ActionRecommendation.MANUAL_REVIEW_REQUIRED


# ---------------------------------------------------------------------------
# Visual signal band outcomes (experimental, uncalibrated)
# ---------------------------------------------------------------------------


def _strong_evidence_inspection() -> MediaInspectionResult:
    """Valid video with a present, non-silent audio track."""
    return MediaInspectionResult(
        file_path="C:/test/strong.mp4",
        file_name="strong.mp4",
        file_size_bytes=1000,
        media_type=MediaType.VIDEO,
        is_valid=True,
        video=VideoStreamMetadata(
            width=640, height=480, fps=30.0, frame_count=60, duration_sec=2.0
        ),
        audio=AudioStreamMetadata(has_audio=True, is_silent=False, sample_rate=16000),
    )


def test_strong_fake_signal_yields_suspicious():
    """A clearly-high uncalibrated score routes to manual review as a SIGNAL.

    The outcome is SUSPICIOUS (a detector signal), never VERIFIED or
    REJECTED, and the action is always manual review.
    """
    gate = UncertaintyGate()
    visual = VisualAnalysisResult(
        status=AnalyzerStatus.COMPLETED,
        faces_detected_count=8,
        frames_evaluated_count=8,
        raw_score=0.75,
        decision_threshold=0.5,
        is_score_calibrated=False,
    )
    # Speech detected so the AUDIO_SILENT branch does not mask this rule.
    sync = SyncAnalysisResult(
        status=AnalyzerStatus.NOT_AVAILABLE,
        has_audio_track=True,
        audio_speech_detected=True,
    )

    report, decision, action, reason = gate.evaluate(
        _strong_evidence_inspection(), visual, sync
    )

    assert decision == AssessmentDecision.SUSPICIOUS
    assert action == ActionRecommendation.MANUAL_REVIEW_REQUIRED
    assert decision != AssessmentDecision.VERIFIED
    assert decision != AssessmentDecision.REJECTED
    assert report.rule_evaluations.get("visual_signal_band") == (
        "potential_manipulation_signal"
    )
    assert "detector signal" in reason or "signal" in reason.lower()


def test_low_fake_signal_is_inconclusive_not_genuine():
    """A clearly-low uncalibrated score must NOT establish authenticity.

    The outcome stays INCONCLUSIVE (gate closed) because a low fake-class
    score is not proof that the media is genuine.
    """
    gate = UncertaintyGate()
    visual = VisualAnalysisResult(
        status=AnalyzerStatus.COMPLETED,
        faces_detected_count=8,
        frames_evaluated_count=8,
        raw_score=0.10,
        decision_threshold=0.5,
        is_score_calibrated=False,
    )
    sync = SyncAnalysisResult(
        status=AnalyzerStatus.NOT_AVAILABLE,
        has_audio_track=True,
        audio_speech_detected=True,
    )

    report, decision, action, _ = gate.evaluate(
        _strong_evidence_inspection(), visual, sync
    )

    assert decision == AssessmentDecision.INCONCLUSIVE
    assert decision != AssessmentDecision.VERIFIED
    assert report.is_gate_closed is True
    assert report.rule_evaluations.get("visual_signal_band") == (
        "no_strong_visual_signal"
    )


def test_boundary_score_is_inconclusive():
    """A score near the default operating point is too close to call."""
    gate = UncertaintyGate()
    visual = VisualAnalysisResult(
        status=AnalyzerStatus.COMPLETED,
        faces_detected_count=8,
        frames_evaluated_count=8,
        raw_score=0.52,
        decision_threshold=0.5,
        is_score_calibrated=False,
    )
    sync = SyncAnalysisResult(
        status=AnalyzerStatus.NOT_AVAILABLE,
        has_audio_track=True,
        audio_speech_detected=True,
    )

    report, decision, action, _ = gate.evaluate(
        _strong_evidence_inspection(), visual, sync
    )

    assert decision == AssessmentDecision.INCONCLUSIVE
    assert report.rule_evaluations.get("visual_signal_band") == "inconclusive"


def test_partial_face_evidence_never_suspicious():
    """A high score with too few face frames stays INCONCLUSIVE.

    Strong signals are only honoured when there is sufficient face
    evidence (>= 3 frames); with 1-2 frames the detector output is
    not trusted enough to raise even an experimental signal.
    """
    gate = UncertaintyGate()
    visual = VisualAnalysisResult(
        status=AnalyzerStatus.COMPLETED,
        faces_detected_count=2,
        frames_evaluated_count=8,
        raw_score=0.9,
        decision_threshold=0.5,
        is_score_calibrated=False,
    )
    sync = SyncAnalysisResult(
        status=AnalyzerStatus.NOT_AVAILABLE,
        has_audio_track=True,
        audio_speech_detected=True,
    )

    report, decision, action, _ = gate.evaluate(
        _strong_evidence_inspection(), visual, sync
    )

    assert UncertaintyFlag.PARTIAL_FACE_EVIDENCE in report.flags
    assert decision == AssessmentDecision.INCONCLUSIVE
    assert decision != AssessmentDecision.SUSPICIOUS
