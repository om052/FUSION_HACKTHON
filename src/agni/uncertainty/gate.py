"""The AGNI Uncertainty Gate.

Enforces strict epistemic honesty:
1. Rejects simulated/fabricated decisions when evidence is insufficient or missing.
2. Missing audio NEVER counts as evidence of genuine or fake.
3. Distinguishes NOT_AVAILABLE, INCONCLUSIVE, UNSUPPORTED, and ERROR.
"""

from typing import List, Tuple
from agni.contracts.enums import (
    ActionRecommendation,
    AnalyzerStatus,
    AssessmentDecision,
    MediaType,
    UncertaintyFlag,
)
from agni.contracts.media import MediaInspectionResult
from agni.contracts.analyzers import SyncAnalysisResult, VisualAnalysisResult
from agni.contracts.uncertainty import UncertaintyReport


#: Half-width of the experimental "inconclusive band" around the
#: detector's default operating point (0.5).  This is an UNVALIDATED
#: display heuristic: it has NOT been derived from labelled validation
#: data and must never be presented as a calibrated decision boundary.
#: It only separates "clearly above the reference point", "clearly
#: below", and "too close to call" for display purposes.
VISUAL_SIGNAL_MARGIN = 0.1

#: The detector's default operating point.  This is the model's
#: softmax midpoint, NOT a validated decision threshold.
_DEFAULT_OPERATING_POINT = 0.5


def classify_visual_signal(
    raw_score: float | None,
    decision_threshold: float | None = None,
    margin: float = VISUAL_SIGNAL_MARGIN,
) -> str:
    """Classify an UNCALIBRATED detector score into an experimental band.

    The result is a display aid only.  It carries no demonstrated
    sensitivity, specificity, or calibration and must never be
    presented as a probability of manipulation, proof of fraud, or
    proof of authenticity.

    Args:
        raw_score: Mean fake-class softmax output (uncailibrated),
            or None when the detector produced no score.
        decision_threshold: The detector's reference operating point
            (defaults to 0.5 when None).
        margin: Half-width of the inconclusive band (experimental).

    Returns:
        One of:
          ``"potential_manipulation_signal"``  (score >= threshold + margin)
          ``"no_strong_visual_signal"``        (score <= threshold - margin)
          ``"inconclusive"``                   (within band, or no score)
    """
    if raw_score is None:
        return "inconclusive"
    score = float(raw_score)
    threshold = (
        float(decision_threshold)
        if decision_threshold is not None
        else _DEFAULT_OPERATING_POINT
    )
    if score >= threshold + margin:
        return "potential_manipulation_signal"
    if score <= threshold - margin:
        return "no_strong_visual_signal"
    return "inconclusive"


class UncertaintyGate:
    """Evaluates multi-modal evidence and identifies epistemic gaps."""

    def evaluate(
        self,
        media_inspection: MediaInspectionResult,
        visual_analysis: VisualAnalysisResult,
        sync_analysis: SyncAnalysisResult,
    ) -> Tuple[UncertaintyReport, AssessmentDecision, ActionRecommendation, str]:
        """Runs rule-based uncertainty arbitration across all available inputs.

        Rule priority (highest first).  Each rule returns immediately
        when it fires, so higher-priority missing-evidence conditions
        always mask lower-priority ones:

        1. Media container integrity / decodability
           (``UNSUPPORTED_MEDIA`` / ``SYSTEM_ERROR``).
        2. No face detected (``INCONCLUSIVE`` + ``RECAPTURE_CLEAR_FACE``).
        3. Missing or silent audio (``INCONCLUSIVE`` +
           ``RECAPTURE_WITH_AUDIO``).  Missing audio is never allowed
           to become evidence of a genuine video, and it also prevents
           a strong visual signal from being promoted to ``SUSPICIOUS``:
           without an audio track the multimodal input is incomplete, so
           the honest recommendation is to recapture with audio.
        4. Duration too short (``INCONCLUSIVE`` + ``MANUAL_REVIEW``).
        5. Analyzer execution failure (``SYSTEM_ERROR`` + ``MANUAL_REVIEW``).
        6. Analyzer idle / unavailable (``INCONCLUSIVE`` + ``MANUAL_REVIEW``).
        7. Visual signal band (only with sufficient face evidence and a
           complete-enough input):
             - strong fake-class signal  -> ``SUSPICIOUS``
               (``MANUAL_REVIEW_REQUIRED``) — an experimental detector
               signal, never proof of manipulation;
             - weak / inconclusive signal -> ``INCONCLUSIVE``
               (``MANUAL_REVIEW_REQUIRED``) — a weak score does NOT
               establish authenticity.

        The gate never emits ``VERIFIED`` or ``REJECTED``: an
        uncalibrated score is never treated as proof of authenticity
        or as grounds to accept or reject a real person.

        Returns:
            Tuple of (UncertaintyReport, AssessmentDecision, ActionRecommendation, action_reason)
        """
        flags: List[UncertaintyFlag] = []
        missing_signals: List[str] = []
        rule_evals: dict = {}
        conflicting_signals = False

        # --- Rule 1: Media Container Integrity & Decodability ---
        if not media_inspection.is_valid:
            if "Unsupported media container format" in (media_inspection.validation_error or ""):
                flags.append(UncertaintyFlag.UNSUPPORTED_CONTAINER)
                report = UncertaintyReport(
                    flags=flags,
                    missing_signals=["supported_media_container"],
                    conflicting_signals=False,
                    is_gate_closed=True,
                    summary_reason=f"Media intake rejected: {media_inspection.validation_error}",
                    rule_evaluations={"container_check": "FAILED_UNSUPPORTED"}
                )
                return (
                    report,
                    AssessmentDecision.UNSUPPORTED_MEDIA,
                    ActionRecommendation.ABORT_UNSUPPORTED_MEDIA,
                    f"Unsupported format: {media_inspection.validation_error}"
                )

            # Corrupted / unreadable / empty file
            flags.append(UncertaintyFlag.CORRUPTED_MEDIA)
            report = UncertaintyReport(
                flags=flags,
                missing_signals=["valid_media_stream"],
                conflicting_signals=False,
                is_gate_closed=True,
                summary_reason=f"Media intake failed: {media_inspection.validation_error}",
                rule_evaluations={"container_check": "FAILED_CORRUPT"}
            )
            return (
                report,
                AssessmentDecision.SYSTEM_ERROR,
                ActionRecommendation.ABORT_UNSUPPORTED_MEDIA,
                f"File corrupted or unreadable: {media_inspection.validation_error}"
            )

        rule_evals["container_check"] = "PASSED"

        # --- Rule 2: Temporal Duration Check ---
        if media_inspection.video and media_inspection.video.duration_sec < 0.5:
            flags.append(UncertaintyFlag.SHORT_DURATION)
            missing_signals.append("sufficient_temporal_frames")
            rule_evals["duration_check"] = "FAILED_TOO_SHORT"
        else:
            rule_evals["duration_check"] = "PASSED"

        # --- Rule 3: Audio Presence & Silence Check ---
        has_audio = media_inspection.audio is not None and media_inspection.audio.has_audio
        if not has_audio:
            flags.append(UncertaintyFlag.NO_AUDIO_TRACK)
            missing_signals.append("audio_track")
            rule_evals["audio_track_check"] = "MISSING"
        elif media_inspection.audio and media_inspection.audio.is_silent:
            flags.append(UncertaintyFlag.AUDIO_SILENT)
            missing_signals.append("non_silent_speech_audio")
            rule_evals["audio_track_check"] = "SILENT"
        elif has_audio and sync_analysis.has_audio_track and not sync_analysis.audio_speech_detected:
            # Sync analyzer detected audio track but no speech signal (silent/non-speech)
            flags.append(UncertaintyFlag.AUDIO_SILENT)
            missing_signals.append("non_silent_speech_audio")
            rule_evals["audio_track_check"] = "SILENT"
        else:
            rule_evals["audio_track_check"] = "PRESENT"

        # --- Rule 4: Visual Analyzer Evidence Checks ---
        if visual_analysis.status == AnalyzerStatus.ERROR:
            flags.append(UncertaintyFlag.ANALYZER_FAILURE)
            missing_signals.append("visual_detector_output")
            rule_evals["visual_analyzer"] = f"ERROR: {visual_analysis.error_message}"
        elif visual_analysis.status == AnalyzerStatus.NOT_AVAILABLE:
            missing_signals.append("visual_detector_output")
            # Detector returns NOT_AVAILABLE with faces_detected_count=0 when no faces found
            if visual_analysis.faces_detected_count == 0:
                flags.append(UncertaintyFlag.NO_FACES_DETECTED)
                missing_signals.append("facial_region")
                rule_evals["visual_analyzer"] = "NO_FACES"
            else:
                rule_evals["visual_analyzer"] = "NOT_AVAILABLE"
        elif visual_analysis.faces_detected_count == 0:
            flags.append(UncertaintyFlag.NO_FACES_DETECTED)
            missing_signals.append("facial_region")
            rule_evals["visual_analyzer"] = "NO_FACES"
        elif visual_analysis.faces_detected_count < 3:
            flags.append(UncertaintyFlag.PARTIAL_FACE_EVIDENCE)
            rule_evals["visual_analyzer"] = "INSUFFICIENT_FACE_FRAMES"
        else:
            rule_evals["visual_analyzer"] = "EVALUATED"

        # --- Rule 5: Audio-Video Synchronization Evidence Checks ---
        if sync_analysis.status == AnalyzerStatus.ERROR:
            flags.append(UncertaintyFlag.ANALYZER_FAILURE)
            missing_signals.append("sync_analyzer_output")
            rule_evals["sync_analyzer"] = f"ERROR: {sync_analysis.error_message}"
        elif sync_analysis.status == AnalyzerStatus.NOT_AVAILABLE:
            missing_signals.append("sync_analyzer_output")
            rule_evals["sync_analyzer"] = "NOT_AVAILABLE"
        elif not has_audio:
            rule_evals["sync_analyzer"] = "SKIPPED_NO_AUDIO"
        else:
            rule_evals["sync_analyzer"] = "EVALUATED"

        # --- Rule 6: Epistemic Gate Decision Synthesis ---
        # If any fundamental signal is missing or analyzer errored:
        is_gate_closed = len(flags) > 0 or len(missing_signals) > 0

        # Scenario A: Crucial face evidence is completely missing
        if UncertaintyFlag.NO_FACES_DETECTED in flags:
            summary = "Cannot perform verification: no human face detected in the provided video frames."
            report = UncertaintyReport(
                flags=flags,
                missing_signals=missing_signals,
                conflicting_signals=False,
                is_gate_closed=True,
                summary_reason=summary,
                rule_evaluations=rule_evals,
            )
            return (
                report,
                AssessmentDecision.INCONCLUSIVE,
                ActionRecommendation.RECAPTURE_CLEAR_FACE,
                "No face detected in video. Request user recapture with the face clearly visible in frame."
            )

        # Scenario B: Audio track is missing or silent
        if UncertaintyFlag.NO_AUDIO_TRACK in flags or UncertaintyFlag.AUDIO_SILENT in flags:
            summary = "Audio stream is absent or silent. Audio-video synchronization cannot be computed. Cannot conclude authenticity without multimodal evidence."
            report = UncertaintyReport(
                flags=flags,
                missing_signals=missing_signals,
                conflicting_signals=False,
                is_gate_closed=True,
                summary_reason=summary,
                rule_evaluations=rule_evals,
            )
            return (
                report,
                AssessmentDecision.INCONCLUSIVE,
                ActionRecommendation.RECAPTURE_WITH_AUDIO,
                "Audio stream missing or silent. Multi-modal verification requires an audible speech track."
            )

        # Scenario C: Duration is too short
        if UncertaintyFlag.SHORT_DURATION in flags:
            summary = "Video duration is under minimum duration threshold (<0.5s). Evidence is insufficient for temporal consistency."
            report = UncertaintyReport(
                flags=flags,
                missing_signals=missing_signals,
                conflicting_signals=False,
                is_gate_closed=True,
                summary_reason=summary,
                rule_evaluations=rule_evals,
            )
            return (
                report,
                AssessmentDecision.INCONCLUSIVE,
                ActionRecommendation.MANUAL_REVIEW_REQUIRED,
                "Video clip duration too short for temporal evaluation."
            )

        # Scenario D: Analyzer execution error
        if UncertaintyFlag.ANALYZER_FAILURE in flags:
            summary = "One or more analyzer components encountered an execution error."
            report = UncertaintyReport(
                flags=flags,
                missing_signals=missing_signals,
                conflicting_signals=False,
                is_gate_closed=True,
                summary_reason=summary,
                rule_evaluations=rule_evals,
            )
            return (
                report,
                AssessmentDecision.SYSTEM_ERROR,
                ActionRecommendation.MANUAL_REVIEW_REQUIRED,
                "Analyzer failure encountered during processing. Route to manual review."
            )

        # Scenario E: When models haven't executed yet (Milestone 1 state)
        # Only applies if visual analyzer is IDLE (not run) or NOT_AVAILABLE for reasons other than no faces
        if visual_analysis.status == AnalyzerStatus.IDLE:
            flags.append(UncertaintyFlag.INSUFFICIENT_EVIDENCE)
            summary = "Model inference has not been performed or analyzer is idle. No synthetic decisions generated."
            report = UncertaintyReport(
                flags=flags,
                missing_signals=["visual_inference", "sync_inference"],
                conflicting_signals=False,
                is_gate_closed=True,
                summary_reason=summary,
                rule_evaluations=rule_evals,
            )
            return (
                report,
                AssessmentDecision.INCONCLUSIVE,
                ActionRecommendation.MANUAL_REVIEW_REQUIRED,
                "Model inference not yet executed. System maintains honest non-verdict."
            )

        # Scenario F: Visual analyzer NOT_AVAILABLE but not due to missing faces (e.g., model not loaded)
        if visual_analysis.status == AnalyzerStatus.NOT_AVAILABLE:
            flags.append(UncertaintyFlag.INSUFFICIENT_EVIDENCE)
            summary = "Visual analyzer unavailable. No visual evidence could be computed."
            report = UncertaintyReport(
                flags=flags,
                missing_signals=["visual_inference", "sync_inference"],
                conflicting_signals=False,
                is_gate_closed=True,
                summary_reason=summary,
                rule_evaluations=rule_evals,
            )
            return (
                report,
                AssessmentDecision.INCONCLUSIVE,
                ActionRecommendation.MANUAL_REVIEW_REQUIRED,
                "Visual analyzer unavailable. Cannot reach determination without visual evidence."
            )

        # Base fallback is replaced by Scenario G (visual signal
        # assessment) and an honest no-strong-signal fallback below.

        # --- Scenario G: Visual signal assessment (experimental) ---
        # Reached only when the visual analyzer COMPLETED with
        # sufficient face evidence and no higher-priority
        # missing-evidence rule (no face, no audio, short clip,
        # analyzer failure) fired.  The raw score is uncalibrated,
        # so the outcome is a *signal*, never a verdict.
        visual_signal = classify_visual_signal(
            visual_analysis.raw_score, visual_analysis.decision_threshold
        )
        rule_evals["visual_signal_band"] = visual_signal

        sufficient_face_evidence = (
            visual_analysis.faces_detected_count >= 3
            and UncertaintyFlag.PARTIAL_FACE_EVIDENCE not in flags
        )

        if sufficient_face_evidence and visual_signal == "potential_manipulation_signal":
            summary = (
                "The visual detector returned a strong fake-class signal "
                "above its experimental reference band. This is an "
                "uncalibrated detector signal, not proof of manipulation, "
                "and it does not establish that the person or recording is "
                "fraudulent."
            )
            report = UncertaintyReport(
                flags=flags,
                missing_signals=missing_signals,
                conflicting_signals=False,
                is_gate_closed=True,
                summary_reason=summary,
                rule_evaluations=rule_evals,
            )
            return (
                report,
                AssessmentDecision.SUSPICIOUS,
                ActionRecommendation.MANUAL_REVIEW_REQUIRED,
                "Potential manipulation signal from the uncalibrated visual "
                "detector (experimental). Route to manual review. This is a "
                "detector signal, not proof of manipulation, and no "
                "authenticity or identity claim is made."
            )

        # Honest no-strong-signal fallback: a low fake-class score
        # does NOT establish authenticity.
        if visual_signal == "no_strong_visual_signal" and sufficient_face_evidence:
            summary = (
                "The visual detector did not return a strong manipulation "
                "signal (uncalibrated, experimental). This does NOT "
                "establish that the media is authentic or genuine."
            )
        elif visual_signal == "inconclusive":
            summary = (
                "The visual detector score is within the inconclusive band "
                "around the model's default operating point (experimental). "
                "No reliable visual signal can be reported."
            )
        else:
            summary = (
                "Evidence evaluated through the Uncertainty Gate without a "
                "strong visual signal."
            )
        report = UncertaintyReport(
            flags=flags,
            missing_signals=missing_signals,
            conflicting_signals=conflicting_signals,
            is_gate_closed=is_gate_closed,
            summary_reason=summary,
            rule_evaluations=rule_evals,
        )
        return (
            report,
            AssessmentDecision.INCONCLUSIVE,
            ActionRecommendation.MANUAL_REVIEW_REQUIRED,
            "No strong visual manipulation signal (experimental, uncalibrated). "
            "This does not establish authenticity. Route to manual review."
        )
