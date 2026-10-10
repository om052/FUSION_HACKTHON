"""Dashboard helper functions for AGNI Streamlit UI.

Provides cached detector loading, frame sampling controls, timing utilities,
and UI rendering helpers without depending on real model weights at import time.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Sequence
import json
import streamlit as st

from agni.models.visual.detector import VisualDeepfakeDetector
from agni.models.syncnet.analyzer import SyncAnalyzer
from agni.contracts.enums import (
    AnalyzerStatus,
    AssessmentDecision,
    ActionRecommendation,
)
from agni.contracts.analyzers import (
    SyncAnalysisResult,
    VisualAnalysisResult,
)
from agni.contracts.receipt import EvidenceReceipt
from agni.intake.processor import MediaProcessor
from agni.contracts.media import MediaInspectionResult
from agni.uncertainty.gate import (
    VISUAL_SIGNAL_MARGIN,
    classify_visual_signal,
)


DEFAULT_MAX_FRAMES = 8
MIN_FRAMES = 1
MAX_FRAMES = 30

#: Width of the assessment bands around the model's default operating
#: point.  This is an EXPERIMENTAL, UNVALIDATED heuristic chosen for
#: display purposes only.  It has NOT been derived from labeled
#: validation data and must never be presented as a reliable decision
#: threshold or as a calibrated boundary.  It is shared with the
#: uncertainty gate so the UI and the gate never disagree.
ASSESSMENT_MARGIN = VISUAL_SIGNAL_MARGIN


def get_detector_device_info(detector: VisualDeepfakeDetector) -> Dict[str, Any]:
    """Return the actual compute device and thread configuration of a detector.

    Reads the attributes recorded at detector construction time so the UI can
    display what really ran, rather than assuming CUDA/GPU availability.

    Args:
        detector: A loaded VisualDeepfakeDetector instance.

    Returns:
        Dict with device, intra-op and interop thread counts, and the
        thread-calibration trace.  Missing attributes (e.g. on older
        detector builds) are reported as None rather than crashing the UI.
    """
    return {
        "device": getattr(detector, "device_info", None),
        "intra_op_threads": getattr(detector, "intra_op_threads", None),
        "interop_threads": getattr(detector, "interop_threads", None),
        "thread_calibration": getattr(detector, "thread_calibration", None),
    }


def render_detector_diagnostics(detector: VisualDeepfakeDetector) -> None:
    """Render the actual device and thread configuration in the sidebar.

    Displays only what was measured at construction time.  Never assumes GPU
    or CUDA availability.
    """
    info = get_detector_device_info(detector)
    with st.expander("Runtime configuration", expanded=False):
        st.write("Compute device:", info["device"] or "Unavailable")
        if info["intra_op_threads"] is not None:
            st.write("Intra-op threads:", info["intra_op_threads"])
        if info["interop_threads"] is not None:
            st.write("Interop threads:", info["interop_threads"])
        calib = info.get("thread_calibration")
        if calib:
            st.caption("Thread calibration:")
            st.json(calib)


def sample_frames(frames: Sequence, max_frames: int) -> List:
    """Select up to `max_frames` evenly spaced frames from the input sequence.

    Args:
        frames: Sequence of frames (any type, typically numpy arrays).
        max_frames: Maximum number of frames to return (1-30).

    Returns:
        List of sampled frames. Returns a new list; does not mutate the input.
        If frames is empty, returns an empty list.
        If max_frames >= len(frames), returns a copy of all frames.
        If max_frames < 1, clamps to 1.
        If max_frames > 30, clamps to 30.
    """
    if not frames:
        return []

    clamped_max = max(MIN_FRAMES, min(MAX_FRAMES, max_frames))
    total = len(frames)

    if clamped_max >= total:
        return list(frames)

    if clamped_max == 1:
        return [frames[0]]

    indices = [int(round(i * (total - 1) / (clamped_max - 1))) for i in range(clamped_max)]
    # Ensure unique indices in case of rounding collisions
    seen = set()
    unique_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    return [frames[i] for i in unique_indices]


@st.cache_resource(show_spinner=False)
def load_detector(
    yunet_path: str,
    xception_path: str,
    face_score_threshold: float = 0.6,
    face_margin: float = 1.3,
) -> VisualDeepfakeDetector:
    """Load and cache the VisualDeepfakeDetector instance.

    Args:
        yunet_path: Path to YuNet ONNX model.
        xception_path: Path to Xception checkpoint.
        face_score_threshold: Minimum face detection confidence.
        face_margin: Bounding box expansion factor.

    Returns:
        Cached VisualDeepfakeDetector instance.
    """
    return VisualDeepfakeDetector(
        yunet_model_path=yunet_path,
        xception_checkpoint_path=xception_path,
        face_score_threshold=face_score_threshold,
        face_margin=face_margin,
    )


def clear_detector_cache() -> None:
    """Clear the cached detector resource.

    Call this when model paths change or to force a fresh detector load.
    """
    load_detector.clear()


def extract_video_frames(
    video_path: Path,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> List:
    """Extract frames from a video file.

    Args:
        video_path: Path to video file.
        max_frames: Maximum number of frames to extract (1-30).

    Returns:
        List of BGR numpy frames.
    """
    processor = MediaProcessor()
    return processor.extract_frames(video_path, max_frames=max_frames, rgb=False)


def run_visual_analysis(
    detector: VisualDeepfakeDetector,
    frames: List,
    record_timing: bool = True,
) -> VisualAnalysisResult:
    """Run visual deepfake analysis on extracted frames.

    Args:
        detector: Loaded VisualDeepfakeDetector instance.
        frames: List of BGR numpy frames.
        record_timing: Whether to include timing breakdown in diagnostics.

    Returns:
        VisualAnalysisResult with analysis outcome and optional timing.
    """
    return detector.analyze_frames(frames, record_timing=record_timing)


def run_sync_analysis(
    video_path: Path,
    media_inspection: Optional[MediaInspectionResult] = None,
    analyzer: Optional[SyncAnalyzer] = None,
    output_wav_dir: Optional[Path] = None,
) -> SyncAnalysisResult:
    """Run the audio preprocessing + synchronization diagnostic pipeline.

    Unlike the visual analyzer, the SyncNet stage is not yet available, so the
    returned :class:`SyncAnalysisResult` always reports synchronization as
    ``NOT_AVAILABLE`` (or ``ERROR`` for an audio processing failure) and never
    emits an invented temporal offset or confidence score.  The audio
    feature-extraction status is recorded in ``diagnostics``.

    Args:
        video_path: Path to the source media container.
        media_inspection: Optional pre-run media inspection (enables precise
            "no audio track" classification without re-probing).
        analyzer: Optional :class:`SyncAnalyzer` override (for caching/testing).
        output_wav_dir: Directory for the extracted WAV.

    Returns:
        A truthful :class:`SyncAnalysisResult`.
    """
    instance = analyzer if analyzer is not None else SyncAnalyzer()
    return instance.analyze(
        video_path,
        media_inspection=media_inspection,
        output_wav_dir=output_wav_dir,
    )


def render_frame_slider(
    key: str = "frame_slider",
    default: int = DEFAULT_MAX_FRAMES,
) -> int:
    """Render a frame count slider in the sidebar.

    Args:
        key: Streamlit widget key.
        default: Default frame count.

    Returns:
        Selected number of frames (1-30).
    """
    with st.sidebar:
        st.header("Frame sampling")
        return st.slider(
            "Maximum frames to analyze",
            min_value=MIN_FRAMES,
            max_value=MAX_FRAMES,
            value=default,
            step=1,
            key=key,
            help="Number of frames sampled from the video for visual analysis. "
                 "More frames increase accuracy but take longer.",
        )


def render_cache_clear_button(key: str = "clear_cache_btn") -> bool:
    """Render a development cache-clear button in the sidebar.

    Args:
        key: Streamlit widget key.

    Returns:
        True if button was clicked.
    """
    with st.sidebar:
        st.header("Development")
        if st.button("Clear detector cache", key=key, help="Force reload of YuNet/Xception models"):
            clear_detector_cache()
            st.success("Detector cache cleared. Next run will reload models.")
            return True
    return False


def render_timing_breakdown(
    diagnostics: Dict[str, Any],
    expander_label: str = "Timing breakdown",
) -> None:
    """Render timing breakdown from diagnostics if available.

    Shows timing in both milliseconds and seconds. Handles missing, malformed,
    or unavailable timing values gracefully without crashing or inventing data.

    Args:
        diagnostics: Diagnostics dict from VisualAnalysisResult.
        expander_label: Label for the expander widget.
    """
    timing = diagnostics.get("timing_ms")
    if not timing or not isinstance(timing, dict):
        return

    # Validate and sanitize timing values
    valid_timing = {}
    for key, value in timing.items():
        try:
            valid_timing[key] = float(value)
        except (TypeError, ValueError):
            continue

    if not valid_timing:
        return

    with st.expander(expander_label, expanded=False):
        cols = st.columns(len(valid_timing))
        for idx, (metric, value_ms) in enumerate(valid_timing.items()):
            value_sec = value_ms / 1000.0
            label = metric.replace("_", " ").replace("ms", "").strip().title()
            if not label.endswith("Ms") and not label.endswith("Sec"):
                label = f"{label} (ms)"
            cols[idx].metric(label, f"{value_ms:.1f} ms", f"{value_sec:.3f} s")


def render_visual_results(
    result: VisualAnalysisResult,
    show_frame_diagnostics: bool = True,
) -> None:
    """Render visual analysis results in the UI.

    Args:
        result: VisualAnalysisResult from detector.
        show_frame_diagnostics: Whether to show frame-level diagnostics expander.
    """
    c1, c2, c3 = st.columns(3)
    c1.metric("Frames evaluated", result.frames_evaluated_count)
    c2.metric("Frames with faces", result.faces_detected_count)
    c3.metric("Inference time", f"{result.execution_latency_ms:.1f} ms")

    st.write("Analyzer status:", result.status.value)
    st.write("Model:", result.model_architecture or "Unavailable")
    st.write("Model version:", result.model_version or "Unavailable")

    if result.status == AnalyzerStatus.COMPLETED and result.raw_score is not None:
        st.metric("Raw visual fake-class score", f"{result.raw_score:.4f}")
        st.warning(
            "Uncalibrated model output. Do not interpret this value as "
            "a verified probability or use it alone to approve/reject a person."
        )
        st.write("Score calibrated:", result.is_score_calibrated)
        st.write("Decision threshold configured by model code:", result.decision_threshold)

        if show_frame_diagnostics:
            with st.expander("Frame-level diagnostics"):
                st.json(result.diagnostics)

        render_timing_breakdown(result.diagnostics)

    elif result.status == AnalyzerStatus.NOT_AVAILABLE:
        st.warning(
            "Visual analysis is unavailable, for example because no face "
            "was detected. This does not mean the video is genuine."
        )
        render_timing_breakdown(result.diagnostics)

    else:
        st.error(
            f"Visual analysis did not complete: "
            f"{result.error_message or result.status.value}"
        )


def assess_visual_signal(result: VisualAnalysisResult) -> str:
    """Return one of three cautious, non-definitive assessment labels.

    The labels are deliberately hedged.  ``raw_score`` is an uncalibrated
    softmax output from a model trained on a specific benchmark
    distribution; it is reported here as a signal, never as proof of
    manipulation or authenticity.  The 0.5 value is only the model's
    default operating point, not a calibrated decision boundary, so
    scores near it are reported as inconclusive.

    The classification is delegated to the shared
    :func:`agni.uncertainty.gate.classify_visual_signal` so the
    dashboard label and the uncertainty-gate decision can never drift
    apart.  The band width is an experimental, unvalidated heuristic
    (see :data:`agni.uncertainty.gate.VISUAL_SIGNAL_MARGIN`).

    Labels:
        "Potential manipulation signal" -- score clearly above reference.
        "No strong signal from this detector" -- score clearly below reference.
        "Inconclusive" -- error, no faces, unavailable, or score near boundary.
    """
    band = classify_visual_signal(result.raw_score, result.decision_threshold)
    return {
        "potential_manipulation_signal": "Potential manipulation signal",
        "no_strong_visual_signal": "No strong signal from this detector",
        "inconclusive": "Inconclusive",
    }[band]


def render_visual_assessment(result: VisualAnalysisResult) -> None:
    """Render the final cautious visual assessment with mandatory disclosures.

    Shows the chosen label plus the honest limitations: the score is not a
    probability, the prototype cannot guarantee authenticity, and it cannot
    verify identity or liveness.
    """
    label = assess_visual_signal(result)

    st.subheader("Visual assessment")
    # Colour semantics mirror _DECISION_DISPLAY so the dashboard label
    # and the gate decision can never disagree:
    #   - a strong *manipulation* signal is rendered in red (serious);
    #   - "No strong signal" / "Inconclusive" are cautionary, not positive,
    #     so they use amber (warning).  Neither label is green, because a
    #     low uncalibrated fake-class score is NOT proof of authenticity.
    if label == "Potential manipulation signal":
        st.error(f"**{label}**")
    else:
        # "No strong signal from this detector" and
        # "Inconclusive" are both inconclusive/unavailable-evidence
        # outcomes -> amber (warning), never green.
        st.warning(f"**{label}**")

    st.caption(
        "Experimental heuristic, not a validated threshold. This label comes "
        "from an uncalibrated score compared against a ±"
        f"{ASSESSMENT_MARGIN:.1f} band around the model's default operating "
        "point. The band width is a display convention chosen without labeled "
        "validation data; it has no demonstrated sensitivity, specificity, or "
        "calibration, and must not be used to approve or reject anyone."
    )

    if result.status == AnalyzerStatus.COMPLETED and result.raw_score is not None:
        st.write(
            "The model's mean raw fake-class score is "
            f"**{result.raw_score:.4f}** against a default reference point of "
            f"{float(result.decision_threshold) if result.decision_threshold is not None else 0.5:.2f}. "
            "This is an uncalibrated softmax output from a benchmark-trained "
            "model, not a probability of manipulation and not proof of "
            "authenticity, and it says nothing about whether the person or "
            "the recording device is genuine."
        )
    elif result.status == AnalyzerStatus.NOT_AVAILABLE:
        st.write(
            "No usable face evidence was produced by the detector (for example "
            "no face was detected). This is reported as inconclusive; it does "
            "not mean the video is authentic."
        )
    else:
        st.write(
            f"The visual analyzer did not complete successfully "
            f"({result.status.value}). No visual verdict is rendered."
        )

    st.divider()
    st.markdown("**What this prototype cannot do**")
    st.write(
        "- Guarantee authenticity, liveness, or identity of the person shown.\n"
        "- Verify that a recording device or account is genuine.\n"
        "- Produce a calibrated probability or a legally defensible clearance.\n"
        "- Assess lip-sync or audio-video alignment (SyncNet not integrated)."
    )


def render_audio_sync_diagnostics(result: SyncAnalysisResult) -> None:
    """Render audio preprocessing and synchronization diagnostics.

    Shows audio-track status and feature-extraction status **separately**, and
    makes clear that synchronization inference is unavailable.  No invented
    synchronization scores, temporal offsets, or authenticity decisions are
    displayed.

    Args:
        result: :class:`SyncAnalysisResult` from :func:`run_sync_analysis`.
    """
    st.write("Analyzer status:", result.status.value)

    # --- Audio track status (separate from synchronization) ---
    has_audio = result.has_audio_track
    st.write("Audio track present:", has_audio)
    if not has_audio:
        st.info("This media has no audio track. Audio-video sync cannot be assessed.")
        if result.error_message:
            st.caption(result.error_message)
        return

    # --- Feature-extraction status (separate from synchronization) ---
    st.write("Speech signal detected:", result.audio_speech_detected)
    mfcc_frames = result.diagnostics.get("mfcc_frames")
    stage = result.diagnostics.get("stage")
    if mfcc_frames is not None:
        st.write("MFCC frames extracted:", mfcc_frames)
    if stage is not None:
        st.caption(f"Audio feature stage: {stage}")

    # --- Synchronization availability (never pretended) ---
    st.write("Synchronization analyzer:", "SyncNet")
    if result.status == AnalyzerStatus.NOT_AVAILABLE:
        st.warning(
            "Audio-video synchronization is **not available**. "
            "Audio features may have been pre-extracted, but no verified SyncNet "
            "architecture or checkpoint is integrated, so no temporal offset, "
            "confidence score, or lip-sync classification is produced. "
            "This does not imply media is genuine or authentic."
        )
    elif result.status == AnalyzerStatus.ERROR:
        st.error(
            f"Audio preprocessing failed: {result.error_message or result.status.value}"
        )
        st.warning(
            "Synchronization remains unassessed. No temporal offset or confidence "
            "score is reported."
        )
    else:
        # COMPLETED/INCONCLUSIVE for the audio stage still means SyncNet is
        # unavailable; surface that honestly rather than inventing sync numbers.
        st.warning(
            "Audio features are available, but SyncNet synchronization inference "
            "is not integrated, so no temporal offset or confidence score is "
            "produced."
        )

    if result.error_message and result.status == AnalyzerStatus.NOT_AVAILABLE:
        st.caption(result.error_message)


# ---------------------------------------------------------------------------
# Final assessment (from the Uncertainty Gate) and report export
# ---------------------------------------------------------------------------

#: Maps each gate decision to a display headline, a Streamlit
#: message colour, and a one-line honest description.  VERIFIED
#: and REJECTED are intentionally absent: the prototype never
#: produces them, so a defensive fallback is used if one ever
#: appears.
#:
#: Colour semantics (see README "Colour semantics"):
#:   * ``error`` (red)      -> potential manipulation signal, system
#:                             error, or unsupported media (serious).
#:   * ``warning`` (amber)  -> inconclusive / missing evidence /
#:                             manual review required (caution, never a
#:                             genuine/authentic claim).
#:   * ``info`` (blue)      -> purely informational diagnostics.
#:   * ``success`` (green)  -> operational success only (e.g. cache
#:                             cleared); NEVER a verdict or an
#:                             authenticity claim.
_DECISION_DISPLAY = {
    AssessmentDecision.INCONCLUSIVE: (
        "INCONCLUSIVE",
        "warning",
        "Evidence is insufficient, conflicting, or the score is in the "
        "experimental inconclusive band. No authenticity claim is made.",
    ),
    AssessmentDecision.SUSPICIOUS: (
        "POTENTIAL MANIPULATION SIGNAL",
        "error",
        "The uncalibrated visual detector returned a strong fake-class "
        "signal (experimental). This is a detector signal, not proof of "
        "manipulation.",
    ),
    AssessmentDecision.SYSTEM_ERROR: (
        "SYSTEM ERROR",
        "error",
        "An analyzer failed to execute. No assessment was produced.",
    ),
    AssessmentDecision.UNSUPPORTED_MEDIA: (
        "UNSUPPORTED MEDIA",
        "error",
        "The uploaded file is not a supported, decodable video container.",
    ),
}


def decision_headline(decision: AssessmentDecision) -> str:
    """Return the human-readable headline for a gate decision."""
    entry = _DECISION_DISPLAY.get(decision)
    return entry[0] if entry else "INCONCLUSIVE"


def render_final_assessment(receipt: EvidenceReceipt) -> None:
    """Render the prominent final assessment produced by the Uncertainty Gate.

    Shows the gate decision (not the raw detector score), the
    gate's plain-language summary, the recommended next action,
    and the mandatory disclosure that no authenticity, liveness,
    or identity claim is made.

    Args:
        receipt: The EvidenceReceipt generated for this analysis.
    """
    decision = receipt.assessment_decision
    entry = _DECISION_DISPLAY.get(decision)
    if entry is None:
        # Defensive: the gate never emits VERIFIED/REJECTED, but if
        # a future build does, surface it honestly as inconclusive.
        headline, color, desc = (
            "INCONCLUSIVE",
            "warning",
            "Unrecognized gate decision; treated as inconclusive.",
        )
    else:
        headline, color, desc = entry

    st.subheader("Final assessment")
    with st.container(border=True):
        getattr(st, color)(f"**{headline}**")
        st.write(desc)
        st.write("**Why:**", receipt.uncertainty_report.summary_reason)
        st.write(
            "**Recommended next action:**",
            receipt.recommended_action.value.replace("_", " ").title(),
        )
        st.caption(receipt.action_reason)

        flags = [f.value for f in receipt.uncertainty_report.flags]
        missing = list(receipt.uncertainty_report.missing_signals)
        if flags or missing:
            with st.expander("Active uncertainty flags & missing signals", expanded=False):
                if flags:
                    st.write("Flags:", ", ".join(flags))
                if missing:
                    st.write("Missing signals:", ", ".join(missing))

        st.caption(
            "This is an experimental identity-verification risk "
            "assessment prototype. It does not verify identity, "
            "liveness, or authenticity, and it must not be used to "
            "approve or reject a real person."
        )


__REPORT_SCHEMA_VERSION = "1.0"
_PROJECT_IDENTIFIER = "AGNI - Adaptive Deepfake & Liveness Intelligence"


def _sanitize_json(obj: Any) -> Any:
    """Recursively replace NaN/Infinity/-Infinity with None.

    Pydantic's ``model_dump_json`` (and ``json.dumps`` with the default
    ``allow_nan=True``) emits non-standards-compliant ``NaN``/``Infinity``
    tokens.  The case report must be valid JSON, so any non-finite float
    is converted to ``None`` (i.e. "missing value") rather than being
    left as a literal token or invented.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, float):
        import math

        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def build_report_dict(receipt: EvidenceReceipt) -> Dict[str, Any]:
    """Build a privacy-conscious, JSON-serializable case report.

    Redacts the on-disk temporary file path (which contains no
    useful information and could leak a username on some hosts)
    while keeping the uploaded filename, never includes video
    frames, biometric embeddings, or per-frame pixel data, and
    guarantees the output is standards-compliant JSON (no
    ``NaN``/``Infinity`` tokens).

    The report is derived entirely from the :class:`EvidenceReceipt`
    produced by the uncertainty gate; no result fields are invented and
    no analysis is reconstructed from display text.  Metadata-only
    enrichment (a schema version and an "unvalidated" label for the
    model's default operating point) is added so consumers of the
    report cannot mistake the threshold for a calibrated boundary.

    Args:
        receipt: The EvidenceReceipt to serialize.

    Returns:
        A plain dict safe to export as JSON.
    """
    data = json.loads(receipt.model_dump_json())

    # --- Project identifier & report schema version (metadata only) ---
    data["project"] = _PROJECT_IDENTIFIER
    data["project_version"] = receipt.system_version
    data["report_schema_version"] = __REPORT_SCHEMA_VERSION

    # --- Redact the temporary processing path; keep only the filename ---
    media = data.get("media_inspection", {})
    if isinstance(media, dict):
        media["file_path"] = "<redacted temporary path>"

    # --- Reference threshold, clearly labelled as UNVALIDATED ------------
    # The detector's default operating point (0.5) is the model's softmax
    # midpoint, NOT a validated decision boundary.  Surface that explicitly
    # in the report so it is never mistaken for a calibrated threshold.
    visual = data.get("visual_analysis", {})
    if isinstance(visual, dict):
        diag = visual.get("diagnostics")
        if isinstance(diag, dict):
            # Drop per-frame score arrays to keep the report concise;
            # aggregate diagnostics are retained.
            diag.pop("per_frame_fake_scores", None)
        threshold = visual.get("decision_threshold")
        visual["threshold_calibration"] = (
            "unvalidated (model default operating point, not a proven boundary)"
        )
        if threshold is not None:
            visual["threshold_note"] = (
                "Unvalidated: this is the model code default (softmax midpoint), "
                "not a threshold derived from labelled validation data."
            )

    # --- SyncNet unavailability disclosure (never concealed) ------------
    sync = data.get("sync_analysis", {})
    if isinstance(sync, dict):
        sync.setdefault(
            "synchronization_status",
            "unavailable (no verified SyncNet architecture or checkpoint is integrated)",
        )

    # --- Honour a privacy request: no temporary path ever leaks ----------
    # (file_path already redacted above; guard against nested copies in
    # diagnostics if a future analyzer ever records one.)
    def _redact_paths(o: Any) -> Any:
        if isinstance(o, dict):
            for k, v in list(o.items()):
                if k == "file_path" and isinstance(v, str) and not v.startswith("<"):
                    o[k] = "<redacted temporary path>"
                else:
                    _redact_paths(v)
        elif isinstance(o, list):
            for v in o:
                _redact_paths(v)
        return o

    _redact_paths(data)

    # --- Guarantee standards-compliant JSON (no NaN/Infinity) -----------
    return _sanitize_json(data)


def render_report_export(receipt: EvidenceReceipt, key: str = "report_download") -> None:
    """Render a JSON case-report download button.

    Args:
        receipt: The EvidenceReceipt to export.
        key: Streamlit widget key.
    """
    try:
        report = build_report_dict(receipt)
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
        st.download_button(
            label="Download case report (JSON)",
            data=payload,
            file_name=f"agni_report_{receipt.receipt_id}.json",
            mime="application/json",
            key=key,
            help="Privacy-conscious report: no video frames, no biometric "
                 "embeddings, temporary paths redacted.",
        )
    except Exception as exc:  # pragma: no cover - defensive
        st.caption(f"Report export unavailable: {type(exc).__name__}: {exc}")