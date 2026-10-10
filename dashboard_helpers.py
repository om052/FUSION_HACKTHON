"""Root-level dashboard helpers for AGNI Streamlit UI.

This module re-exports helpers from the agni package for
convenience.  The actual implementations are in
src/agni/dashboard_helpers.py.  Nothing in the application
imports this shim directly — app.py and the tests import from
``agni.dashboard_helpers`` — so it exists only as a convenience
for external scripts.
"""

import sys
from pathlib import Path

# Ensure src/ is on the path for agni package imports (same as app.py)
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agni.dashboard_helpers import (  # noqa: F401
    load_detector,
    clear_detector_cache,
    extract_video_frames,
    run_visual_analysis,
    run_sync_analysis,
    sample_frames,
    render_frame_slider,
    render_cache_clear_button,
    render_detector_diagnostics,
    get_detector_device_info,
    render_timing_breakdown,
    render_visual_results,
    render_visual_assessment,
    render_audio_sync_diagnostics,
    render_final_assessment,
    render_report_export,
    build_report_dict,
    decision_headline,
    assess_visual_signal,
    DEFAULT_MAX_FRAMES,
    MIN_FRAMES,
    MAX_FRAMES,
    ASSESSMENT_MARGIN,
)

__all__ = [
    "load_detector",
    "clear_detector_cache",
    "extract_video_frames",
    "run_visual_analysis",
    "run_sync_analysis",
    "sample_frames",
    "render_frame_slider",
    "render_cache_clear_button",
    "render_detector_diagnostics",
    "get_detector_device_info",
    "render_timing_breakdown",
    "render_visual_results",
    "render_visual_assessment",
    "render_audio_sync_diagnostics",
    "render_final_assessment",
    "render_report_export",
    "build_report_dict",
    "decision_headline",
    "assess_visual_signal",
    "DEFAULT_MAX_FRAMES",
    "MIN_FRAMES",
    "MAX_FRAMES",
    "ASSESSMENT_MARGIN",
]
