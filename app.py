import json
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

# Allow imports from src/agni in this project layout.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agni.intake.validator import MediaValidator
from agni.intake.processor import MediaProcessor
from agni.intake.upload import temporary_upload
from agni.models.visual.detector import VisualDeepfakeDetector
from agni.models.syncnet.analyzer import SyncAnalyzer
from agni.contracts.enums import AnalyzerStatus, AssessmentDecision, ActionRecommendation
from agni.contracts.receipt import EvidenceReceipt
from agni.uncertainty.gate import UncertaintyGate
from agni.receipt.generator import ReceiptGenerator
from agni.dashboard_helpers import (
    load_detector,
    extract_video_frames,
    run_visual_analysis,
    run_sync_analysis,
    render_frame_slider,
    render_cache_clear_button,
    render_detector_diagnostics,
    render_visual_results,
    render_visual_assessment,
    render_audio_sync_diagnostics,
    render_final_assessment,
    render_report_export,
    render_timing_breakdown,
    DEFAULT_MAX_FRAMES,
)

st.set_page_config(
    page_title="AGNI | Adaptive Deepfake & Liveness Intelligence",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header / problem statement / disclaimer
# ---------------------------------------------------------------------------
st.title("🛡️ AGNI")
st.caption("Adaptive Deepfake & Liveness Intelligence")

st.markdown(
    "**Problem domain:** deepfake-resistant identity verification for digital "
    "onboarding. AGNI analyses an uploaded video, evaluates the visual and "
    "audio evidence that is actually available, exposes uncertainty and "
    "missing evidence, and explains why a case requires further review."
)

st.warning(
    "**Research prototype.** This is an experimental identity-verification "
    "risk-assessment prototype, not a certified banking authentication system. "
    "The visual model score is **uncalibrated** and is not proof of "
    "authenticity, liveness, identity, or fraud. It must not be used to "
    "approve or reject a real person."
)

# ---------------------------------------------------------------------------
# Sidebar: scope, configuration, runtime
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Analysis scope")
    st.write("Media validation: enabled")
    st.write("YuNet face detection: enabled")
    st.write("Xception visual analysis: enabled")
    st.write("Audio-video SyncNet: **not integrated**")
    st.caption("Assessment bands are an experimental, unvalidated heuristic.")
    st.caption("CPU inference; the actual compute device and thread count are "
               "shown after the model loads.")

    max_frames = render_frame_slider()
    render_cache_clear_button()

# ---------------------------------------------------------------------------
# Persistent analysis state
# ---------------------------------------------------------------------------
if "receipt" not in st.session_state:
    st.session_state.receipt = None
if "analyzed_file_name" not in st.session_state:
    st.session_state.analyzed_file_name = None
if "detector_diagnostics" not in st.session_state:
    st.session_state.detector_diagnostics = None


def _reset_analysis() -> None:
    st.session_state.receipt = None
    st.session_state.analyzed_file_name = None
    st.session_state.detector_diagnostics = None


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload a video for analysis",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    help="Supported containers: MP4, MOV, AVI, MKV, WebM (where the installed "
         "decoding stack supports them). Maximum size 250 MB.",
)

# If a different file is uploaded while results from a previous file are
# still present, discard the stale results immediately so the dashboard
# never shows one video's analysis labelled under another's name.
if (
    uploaded is not None
    and st.session_state.receipt is not None
    and uploaded.name != st.session_state.analyzed_file_name
):
    _reset_analysis()

if uploaded is not None:
    st.write(f"**File:** {uploaded.name}")
    st.write(f"**Size:** {uploaded.size / (1024 * 1024):.2f} MB")

    if uploaded.size > 250 * 1024 * 1024:
        st.error("File exceeds the 250 MB upload limit.")
        st.stop()

    if st.button("Run analysis", type="primary"):
        suffix = Path(uploaded.name).suffix.lower()
        try:
            with temporary_upload(uploaded.getvalue(), suffix=suffix) as temp_path:
                stage_ms: dict = {}
                t_total = time.perf_counter()

                # --- Stage 1: media validation ---
                t0 = time.perf_counter()
                inspection = MediaValidator().inspect_media(temp_path)
                stage_ms["media_validation"] = (time.perf_counter() - t0) * 1000.0

                st.subheader("Media validation")
                c = st.columns(4)
                c[0].metric("Valid", inspection.is_valid)
                c[1].metric("Media type", inspection.media_type.value)
                c[2].metric("Size (MB)", f"{inspection.file_size_bytes / (1024*1024):.2f}")
                c[3].metric("Validation (ms)", f"{inspection.inspection_latency_ms:.1f}")

                if inspection.validation_error:
                    st.error(inspection.validation_error)
                if not inspection.is_valid:
                    st.warning("Analysis stopped because media validation failed.")
                    st.stop()

                if inspection.video is not None:
                    v = inspection.video
                    st.caption(
                        f"Video: {v.width}x{v.height}, {v.fps:.2f} fps, "
                        f"{v.frame_count} frames, {v.duration_sec:.2f} s"
                        + (f", codec {v.codec}" if v.codec else "")
                    )
                if inspection.audio is not None:
                    a = inspection.audio
                    st.caption(
                        f"Audio track: present={a.has_audio}"
                        + (f", {a.sample_rate} Hz, {a.channels} ch" if a.has_audio else "")
                        + (", silent" if a.is_silent else "")
                    )

                # --- Stage 2: frame extraction ---
                t0 = time.perf_counter()
                frames = extract_video_frames(temp_path, max_frames=max_frames)
                stage_ms["frame_extraction"] = (time.perf_counter() - t0) * 1000.0

                if not frames:
                    st.error("No decodable video frames were extracted.")
                    st.stop()
                st.write(f"**Frames extracted:** {len(frames)} (requested up to {max_frames})")

                yunet_path = ROOT / "models" / "yunet" / "face_detection_yunet_2023mar.onnx"
                xception_path = ROOT / "models" / "visual_detector" / "xception_best.pth"
                if not yunet_path.is_file() or not xception_path.is_file():
                    st.error("Required model file is missing. Check the model paths.")
                    st.stop()

                # --- Stage 3: model load (cached) ---
                t0 = time.perf_counter()
                detector = load_detector(
                    yunet_path=str(yunet_path),
                    xception_path=str(xception_path),
                )
                stage_ms["model_load"] = (time.perf_counter() - t0) * 1000.0
                st.session_state.detector_diagnostics = {
                    "device": getattr(detector, "device_info", None),
                    "intra_op_threads": getattr(detector, "intra_op_threads", None),
                    "interop_threads": getattr(detector, "interop_threads", None),
                    "thread_calibration": getattr(detector, "thread_calibration", None),
                }
                render_detector_diagnostics(detector)

                # --- Stage 4: visual inference ---
                inference_status = st.empty()
                inference_status.info(
                    f"Analyzing {len(frames)} frames with YuNet + Xception..."
                )
                t0 = time.perf_counter()
                try:
                    visual_result = run_visual_analysis(
                        detector, frames, record_timing=True
                    )
                finally:
                    stage_ms["visual_inference"] = (time.perf_counter() - t0) * 1000.0
                    inference_status.info(
                        f"Analyzed {len(frames)} frames in "
                        f"{stage_ms['visual_inference']:.0f} ms"
                    )

                # --- Stage 5: audio / sync diagnostic ---
                t0 = time.perf_counter()
                try:
                    sync_result = run_sync_analysis(
                        temp_path,
                        media_inspection=inspection,
                        output_wav_dir=Path(tempfile.gettempdir()) / "agni_sync",
                    )
                finally:
                    stage_ms["audio_sync"] = (time.perf_counter() - t0) * 1000.0

                # --- Stage 6: uncertainty gate + receipt ---
                t0 = time.perf_counter()
                gate = UncertaintyGate()
                uncertainty_report, decision, action, reason = gate.evaluate(
                    inspection, visual_result, sync_result
                )
                total_ms = (time.perf_counter() - t_total) * 1000.0
                stage_ms["uncertainty_gate"] = (time.perf_counter() - t0) * 1000.0

                receipt = ReceiptGenerator.generate(
                    media_inspection=inspection,
                    visual_analysis=visual_result,
                    sync_analysis=sync_result,
                    uncertainty_report=uncertainty_report,
                    assessment_decision=decision,
                    recommended_action=action,
                    action_reason=reason,
                    total_latency_ms=total_ms,
                )
                # Attach per-stage timings to the receipt diagnostics for display
                # AND for the exported report ("Stage timings and total elapsed
                # time").  These are pipeline-stage timings, distinct from the
                # detector's internal timing_ms (rendered in render_visual_results).
                receipt.visual_analysis.diagnostics["stage_timings_ms"] = {
                    k: round(v, 1) for k, v in stage_ms.items()
                }
                receipt.visual_analysis.diagnostics["total_pipeline_ms"] = round(
                    total_ms, 1
                )

                st.session_state.receipt = receipt
                st.session_state.analyzed_file_name = uploaded.name
        except Exception as exc:
            st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
            st.caption("Check the terminal for the full traceback.")

# ---------------------------------------------------------------------------
# Results. Rendered from session_state in the SAME pass so the
# expensive inference runs once; later interactions (report
# download, new analysis) reuse the stored receipt instead of
# re-running the model.
# ---------------------------------------------------------------------------
receipt = st.session_state.receipt
if receipt is not None:
    st.divider()
    st.subheader("Analysis results")
    st.write(f"**File:** {st.session_state.analyzed_file_name}")

    # 1. Prominent final assessment from the Uncertainty Gate.
    render_final_assessment(receipt)

    # 2. Visual detector evidence.
    st.subheader("Visual detector evidence")
    render_visual_results(receipt.visual_analysis)
    render_visual_assessment(receipt.visual_analysis)

    # 3. Audio / lip-sync status.
    st.subheader("Audio and lip-sync analysis")
    render_audio_sync_diagnostics(receipt.sync_analysis)

    # 4. Stage timings (pipeline-level), rendered through the shared helper
    #    so we never hand-roll a second timing panel.  The detector's own
    #    sub-stage timings are already shown inside render_visual_results.
    timings = receipt.visual_analysis.diagnostics.get("stage_timings_ms")
    if timings:
        render_timing_breakdown(
            {"timing_ms": {k: float(v) for k, v in timings.items()}},
            expander_label="Stage timings",
        )
        total_pipeline = receipt.visual_analysis.diagnostics.get(
            "total_pipeline_ms"
        )
        if total_pipeline is not None:
            st.metric(
                "Total pipeline",
                f"{float(total_pipeline):,.0f} ms",
                f"{float(total_pipeline) / 1000:.2f} s",
            )

    # 5. Report export.
    st.subheader("Case report")
    st.caption(
        "Reproducible per-analysis report. Contains observed evidence, analyzer "
        "status, the uncertainty-gate outcome, timings, and limitations. "
        "Temporary file paths are redacted; no video frames or biometric "
        "embeddings are included."
    )
    render_report_export(receipt)

    # 6. Reset / new analysis.
    st.divider()
    if st.button("Start new analysis", type="primary"):
        _reset_analysis()
        st.rerun()

    # 7. Technical explanation + limitations.
    with st.expander("How AGNI works (technical)", expanded=False):
        st.markdown(
            "1. **Intake validation** — the container is probed with FFmpeg and "
            "decoded with OpenCV to confirm it is a real, decodable video and to "
            "read resolution, frame rate, duration, and audio-track metadata.\n"
            "2. **Frame sampling** — up to N evenly spaced frames are decoded.\n"
            "3. **Face detection** — OpenCV YuNet locates faces; the "
            "highest-confidence face per frame is cropped with a margin.\n"
            "4. **Visual inference** — a DeepfakeBench Xception backbone "
            "(2-class, FF++ c23) outputs an uncalibrated softmax fake-class "
            "score per face, averaged across frames. Class 0 = real, class 1 = "
            "fake.\n"
            "5. **Audio diagnostic** — if an audio track exists, it is extracted "
            "as 16 kHz mono PCM and SyncNet-compatible MFCC features are "
            "computed. **SyncNet synchronization inference is not integrated** "
            "(no verified architecture or checkpoint is available), so lip-sync "
            "is reported as UNAVAILABLE rather than simulated.\n"
            "6. **Uncertainty gate** — a rule layer arbitrates the available "
            "evidence. Missing audio, no faces, short clips, and analyzer "
            "failures each produce an honest INCONCLUSIVE / ERROR outcome and "
            "can never be converted into a genuine-video verdict. A strong "
            "fake-class signal (with a complete-enough input) is surfaced as an "
            "experimental POTENTIAL MANIPULATION SIGNAL that routes to manual "
            "review.\n"
            "7. **Evidence receipt** — a structured, exportable record of "
            "measurements, analyzer status, gate outcome, timings, and "
            "limitations."
        )
    st.caption(
        "Limitations: uncalibrated score; single-face analysis (the "
        "highest-confidence face per frame is used — it does not verify the "
        "identity of a specific person and does not track multiple people); "
        "CPU-only inference; SyncNet unavailable; not compliant with PCI DSS, "
        "banking regulations, or biometric privacy law."
    )
else:
    if uploaded is None:
        st.info("Upload a supported video to begin.")

st.divider()
st.caption(
    "AGNI research prototype | Visual + audio-feature inference | "
    "No calibrated authenticity probability | SyncNet pending | "
    "Experimental, unvalidated assessment bands"
)
