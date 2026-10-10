"""Unit tests for dashboard helper functions.

Tests the helper functions in agni.dashboard_helpers without loading
real model weights by mocking the detector and frame extraction.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agni.dashboard_helpers import (
    load_detector,
    clear_detector_cache,
    extract_video_frames,
    run_visual_analysis,
    sample_frames,
    render_timing_breakdown,
    DEFAULT_MAX_FRAMES,
    MIN_FRAMES,
    MAX_FRAMES,
)
from agni.contracts.enums import AnalyzerStatus
from agni.contracts.analyzers import VisualAnalysisResult


class TestDashboardHelpers:
    """Tests for dashboard helper functions."""

    def test_constants(self):
        """Verify frame slider constants."""
        assert DEFAULT_MAX_FRAMES == 8
        assert MIN_FRAMES == 1
        assert MAX_FRAMES == 30

    def test_load_detector_caches(self):
        """Verify load_detector is decorated with st.cache_resource (has clear method)."""
        # The load_detector function should have a 'clear' method from cache_resource
        assert hasattr(load_detector, "clear"), "load_detector should be cached with cache_resource"
        assert callable(load_detector.clear)

    def test_clear_detector_cache_calls_clear(self):
        """Verify clear_detector_cache calls the cached function's clear method."""
        with patch("agni.dashboard_helpers.load_detector") as mock_load:
            mock_load.clear = MagicMock()
            clear_detector_cache()
            mock_load.clear.assert_called_once()

    def test_extract_video_frames_delegates_to_processor(self):
        """Verify extract_video_frames calls MediaProcessor.extract_frames."""
        mock_frames = [MagicMock(), MagicMock()]
        mock_processor = MagicMock()
        mock_processor.extract_frames.return_value = mock_frames

        with patch("agni.dashboard_helpers.MediaProcessor", return_value=mock_processor):
            frames = extract_video_frames(Path("dummy.mp4"), max_frames=5)

            assert frames == mock_frames
            mock_processor.extract_frames.assert_called_once_with(
                Path("dummy.mp4"), max_frames=5, rgb=False
            )

    def test_run_visual_analysis_passes_parameters(self):
        """Verify run_visual_analysis calls detector.analyze_frames with correct args."""
        mock_detector = MagicMock()
        mock_result = MagicMock(spec=VisualAnalysisResult)
        mock_detector.analyze_frames.return_value = mock_result
        mock_frames = [MagicMock(), MagicMock()]

        result = run_visual_analysis(mock_detector, mock_frames, record_timing=True)

        assert result == mock_result
        mock_detector.analyze_frames.assert_called_once_with(
            mock_frames, record_timing=True
        )

    def test_run_visual_analysis_defaults_record_timing_true(self):
        """Verify run_visual_analysis defaults record_timing to True."""
        mock_detector = MagicMock()
        mock_result = MagicMock(spec=VisualAnalysisResult)
        mock_detector.analyze_frames.return_value = mock_result
        mock_frames = [MagicMock()]

        run_visual_analysis(mock_detector, mock_frames)

        mock_detector.analyze_frames.assert_called_once_with(
            mock_frames, record_timing=True
        )


class TestSampleFrames:
    """Tests for the sample_frames helper function."""

    def test_empty_frame_list(self):
        """Empty input returns empty list."""
        assert sample_frames([], 8) == []

    def test_single_frame_requested_one_available(self):
        """One frame available, one requested returns that frame."""
        frames = ["frame1"]
        result = sample_frames(frames, 1)
        assert result == ["frame1"]
        assert result is not frames  # New list, not mutated

    def test_fewer_frames_than_requested(self):
        """When fewer frames available than requested, returns all frames."""
        frames = ["f1", "f2"]
        result = sample_frames(frames, 8)
        assert result == ["f1", "f2"]
        assert result is not frames

    def test_even_sampling_across_larger_list(self):
        """Even sampling across a larger frame list."""
        frames = list(range(100))  # 100 frames
        result = sample_frames(frames, 10)
        assert len(result) == 10
        assert result[0] == 0
        assert result[-1] == 99
        # Check roughly even spacing
        assert result[4] == 44 or result[4] == 45  # ~middle

    def test_request_above_available_frame_count(self):
        """Request clamped to available frames when max_frames > len(frames)."""
        frames = ["f1", "f2", "f3"]
        result = sample_frames(frames, 30)
        assert result == ["f1", "f2", "f3"]

    def test_max_frames_clamped_to_bounds(self):
        """max_frames is clamped to [1, 30] range."""
        frames = list(range(100))
        # Below minimum
        result = sample_frames(frames, 0)
        assert len(result) == 1
        # Above maximum
        result = sample_frames(frames, 100)
        assert len(result) == 30

    def test_does_not_mutate_original(self):
        """Original frame list is not modified."""
        frames = ["f1", "f2", "f3", "f4", "f5"]
        original = frames.copy()
        sample_frames(frames, 3)
        assert frames == original

    def test_two_frames_requested(self):
        """Two frames requested returns first and last."""
        frames = list(range(10))
        result = sample_frames(frames, 2)
        assert result == [0, 9]

    def test_three_frames_requested(self):
        """Three frames requested returns first, middle, last."""
        frames = list(range(10))
        result = sample_frames(frames, 3)
        # Python round(4.5) = 4 (banker's rounding), so middle index is 4
        assert result == [0, 4, 9]


class TestTimingFormatting:
    """Tests for render_timing_breakdown timing formatter."""

    def test_complete_timing_data(self):
        """Complete timing data renders correctly."""
        with patch("agni.dashboard_helpers.st") as mock_st:
            mock_expander = MagicMock()
            mock_expander.__enter__ = MagicMock(return_value=None)
            mock_expander.__exit__ = MagicMock(return_value=None)
            mock_st.expander = MagicMock(return_value=mock_expander)
            mock_cols = [MagicMock() for _ in range(4)]
            mock_st.columns = MagicMock(return_value=mock_cols)

            diagnostics = {
                "timing_ms": {
                    "face_detection_ms": 123.45,
                    "preprocessing_ms": 67.89,
                    "inference_ms": 456.78,
                    "total_ms": 648.12,
                }
            }

            render_timing_breakdown(diagnostics)

            mock_st.expander.assert_called_once()
            mock_st.columns.assert_called_once_with(4)
            for col in mock_cols:
                col.metric.assert_called_once()

    def test_partial_timing_data(self):
        """Partial timing data (missing some keys) renders available metrics."""
        with patch("agni.dashboard_helpers.st") as mock_st:
            mock_expander = MagicMock()
            mock_expander.__enter__ = MagicMock(return_value=None)
            mock_expander.__exit__ = MagicMock(return_value=None)
            mock_st.expander = MagicMock(return_value=mock_expander)
            mock_cols = [MagicMock() for _ in range(2)]
            mock_st.columns = MagicMock(return_value=mock_cols)

            diagnostics = {
                "timing_ms": {
                    "face_detection_ms": 100.0,
                    "inference_ms": 500.0,
                    # preprocessing_ms and total_ms missing
                }
            }

            render_timing_breakdown(diagnostics)

            mock_st.columns.assert_called_once_with(2)

    def test_empty_timing_data(self):
        """Empty or missing timing_ms does nothing."""
        with patch("agni.dashboard_helpers.st") as mock_st:
            mock_st.expander = MagicMock()
            mock_st.columns = MagicMock()

            render_timing_breakdown({})
            render_timing_breakdown({"timing_ms": {}})
            render_timing_breakdown({"timing_ms": None})
            render_timing_breakdown({"other_key": "value"})

            mock_st.expander.assert_not_called()
            mock_st.columns.assert_not_called()

    def test_malformed_timing_values(self):
        """Non-numeric timing values are skipped gracefully."""
        with patch("agni.dashboard_helpers.st") as mock_st:
            mock_expander = MagicMock()
            mock_expander.__enter__ = MagicMock(return_value=None)
            mock_expander.__exit__ = MagicMock(return_value=None)
            mock_st.expander = MagicMock(return_value=mock_expander)
            mock_cols = [MagicMock() for _ in range(2)]
            mock_st.columns = MagicMock(return_value=mock_cols)

            diagnostics = {
                "timing_ms": {
                    "face_detection_ms": "not_a_number",
                    "preprocessing_ms": 50.0,
                    "inference_ms": None,
                    "total_ms": 150.0,
                }
            }

            render_timing_breakdown(diagnostics)

            # Only 2 valid numeric values should render
            mock_st.columns.assert_called_once_with(2)

    def test_timing_labels_show_ms_and_seconds(self):
        """Metric labels and values show both ms and seconds."""
        with patch("agni.dashboard_helpers.st") as mock_st:
            mock_expander = MagicMock()
            mock_expander.__enter__ = MagicMock(return_value=None)
            mock_expander.__exit__ = MagicMock(return_value=None)
            mock_st.expander = MagicMock(return_value=mock_expander)
            mock_col = MagicMock()
            mock_st.columns = MagicMock(return_value=[mock_col])

            diagnostics = {
                "timing_ms": {
                    "face_detection_ms": 1000.0,  # 1 second
                }
            }

            render_timing_breakdown(diagnostics)

            mock_col.metric.assert_called_once()
            args, kwargs = mock_col.metric.call_args
            # Check that value shows ms and delta shows seconds
            assert "1000.0 ms" in args[1]
            assert "1.000 s" in args[2]


class TestDashboardHelperUIFunctions:
    """Tests for UI rendering helper functions."""

    def test_render_frame_slider_returns_selected_value(self):
        """Verify render_frame_slider returns the slider value."""
        with patch("agni.dashboard_helpers.st") as mock_st:
            mock_slider = MagicMock(return_value=12)
            mock_st.slider = mock_slider
            mock_st.header = MagicMock()
            # sidebar is a context manager, need to mock it properly
            mock_sidebar = MagicMock()
            mock_sidebar.__enter__ = MagicMock(return_value=mock_sidebar)
            mock_sidebar.__exit__ = MagicMock(return_value=None)
            mock_st.sidebar = mock_sidebar

            from agni.dashboard_helpers import render_frame_slider

            result = render_frame_slider(key="test_slider", default=5)

            assert result == 12
            mock_st.header.assert_called_with("Frame sampling")
            mock_slider.assert_called_once()
            args, kwargs = mock_slider.call_args
            assert kwargs["min_value"] == MIN_FRAMES
            assert kwargs["max_value"] == MAX_FRAMES
            assert kwargs["value"] == 5
            assert kwargs["key"] == "test_slider"

    def test_render_cache_clear_button_returns_true_when_clicked(self):
        """Verify render_cache_clear_button returns True when button clicked."""
        with patch("agni.dashboard_helpers.st") as mock_st, \
             patch("agni.dashboard_helpers.clear_detector_cache") as mock_clear:
            mock_button = MagicMock(return_value=True)
            mock_st.button = mock_button
            mock_st.header = MagicMock()
            mock_st.success = MagicMock()
            # Mock sidebar context manager
            mock_sidebar = MagicMock()
            mock_sidebar.__enter__ = MagicMock(return_value=mock_sidebar)
            mock_sidebar.__exit__ = MagicMock(return_value=None)
            mock_st.sidebar = mock_sidebar

            from agni.dashboard_helpers import render_cache_clear_button

            result = render_cache_clear_button(key="test_clear")

            assert result is True
            mock_st.header.assert_called_with("Development")
            mock_button.assert_called_once()
            mock_clear.assert_called_once()
            mock_st.success.assert_called_once()

    def test_render_cache_clear_button_returns_false_when_not_clicked(self):
        """Verify render_cache_clear_button returns False when button not clicked."""
        with patch("agni.dashboard_helpers.st") as mock_st, \
             patch("agni.dashboard_helpers.clear_detector_cache") as mock_clear:
            mock_button = MagicMock(return_value=False)
            mock_st.button = mock_button
            mock_st.header = MagicMock()
            mock_st.success = MagicMock()
            mock_sidebar = MagicMock()
            mock_sidebar.__enter__ = MagicMock(return_value=mock_sidebar)
            mock_sidebar.__exit__ = MagicMock(return_value=None)
            mock_st.sidebar = mock_sidebar

            from agni.dashboard_helpers import render_cache_clear_button

            result = render_cache_clear_button(key="test_clear")

            assert result is False
            mock_clear.assert_not_called()
            mock_st.success.assert_not_called()

    def test_render_timing_breakdown_skips_when_no_timing(self):
        """Verify render_timing_breakdown does nothing when timing_ms absent."""
        with patch("agni.dashboard_helpers.st") as mock_st:
            mock_st.expander = MagicMock()
            mock_st.columns = MagicMock()

            render_timing_breakdown({"other_key": "value"})

            mock_st.expander.assert_not_called()
            mock_st.columns.assert_not_called()

    def test_render_timing_breakdown_renders_when_timing_present(self):
        """Verify render_timing_breakdown renders metrics when timing_ms present."""
        with patch("agni.dashboard_helpers.st") as mock_st:
            mock_expander = MagicMock()
            mock_expander.__enter__ = MagicMock(return_value=None)
            mock_expander.__exit__ = MagicMock(return_value=None)
            mock_st.expander = MagicMock(return_value=mock_expander)

            mock_cols = [MagicMock() for _ in range(4)]
            mock_st.columns = MagicMock(return_value=mock_cols)

            diagnostics = {
                "timing_ms": {
                    "face_detection_ms": 123.45,
                    "preprocessing_ms": 67.89,
                    "inference_ms": 456.78,
                    "total_ms": 648.12,
                }
            }

            render_timing_breakdown(diagnostics, expander_label="Test Timing")

            mock_st.expander.assert_called_once_with("Test Timing", expanded=False)
            mock_st.columns.assert_called_once_with(4)
            for idx, col in enumerate(mock_cols):
                col.metric.assert_called_once()

    def test_render_visual_results_completed(self):
        """Verify render_visual_results renders COMPLETED status correctly."""
        with patch("agni.dashboard_helpers.st") as mock_st, \
             patch("agni.dashboard_helpers.render_timing_breakdown") as mock_timing:
            mock_cols = [MagicMock() for _ in range(3)]
            mock_st.columns = MagicMock(return_value=mock_cols)
            mock_st.expander = MagicMock()
            mock_st.metric = MagicMock()
            mock_st.write = MagicMock()
            mock_st.warning = MagicMock()

            mock_result = MagicMock(spec=VisualAnalysisResult)
            mock_result.status = AnalyzerStatus.COMPLETED
            mock_result.frames_evaluated_count = 10
            mock_result.faces_detected_count = 8
            mock_result.execution_latency_ms = 1234.5
            mock_result.model_architecture = "Xception"
            mock_result.model_version = "v1.0"
            mock_result.raw_score = 0.75
            mock_result.is_score_calibrated = False
            mock_result.decision_threshold = 0.5
            mock_result.diagnostics = {"timing_ms": {"total_ms": 1234.5}}

            from agni.dashboard_helpers import render_visual_results

            render_visual_results(mock_result, show_frame_diagnostics=True)

            # 3 column metrics + 1 st.metric for raw score = 4 total
            assert mock_st.metric.call_count >= 1
            # Column metrics are called on column objects, not st.metric
            total_metrics = mock_st.metric.call_count
            for col in mock_cols:
                total_metrics += col.metric.call_count
            assert total_metrics >= 4
            mock_st.write.assert_any_call("Analyzer status:", AnalyzerStatus.COMPLETED.value)
            mock_st.warning.assert_called()
            mock_timing.assert_called_once()

    def test_render_visual_results_not_available(self):
        """Verify render_visual_results renders NOT_AVAILABLE status correctly."""
        with patch("agni.dashboard_helpers.st") as mock_st, \
             patch("agni.dashboard_helpers.render_timing_breakdown") as mock_timing:
            mock_cols = [MagicMock() for _ in range(3)]
            mock_st.columns = MagicMock(return_value=mock_cols)
            mock_st.write = MagicMock()
            mock_st.warning = MagicMock()

            mock_result = MagicMock(spec=VisualAnalysisResult)
            mock_result.status = AnalyzerStatus.NOT_AVAILABLE
            mock_result.frames_evaluated_count = 10
            mock_result.faces_detected_count = 0
            mock_result.execution_latency_ms = 100.0
            mock_result.model_architecture = "Xception"
            mock_result.model_version = "v1.0"
            mock_result.raw_score = None
            mock_result.diagnostics = {"timing_ms": {"total_ms": 100.0}}

            from agni.dashboard_helpers import render_visual_results

            render_visual_results(mock_result)

            mock_st.warning.assert_called()
            mock_timing.assert_called_once()

    def test_render_visual_results_error(self):
        """Verify render_visual_results renders ERROR status correctly."""
        with patch("agni.dashboard_helpers.st") as mock_st:
            mock_cols = [MagicMock() for _ in range(3)]
            mock_st.columns = MagicMock(return_value=mock_cols)
            mock_st.write = MagicMock()
            mock_st.error = MagicMock()

            mock_result = MagicMock(spec=VisualAnalysisResult)
            mock_result.status = AnalyzerStatus.ERROR
            mock_result.frames_evaluated_count = 10
            mock_result.faces_detected_count = 5
            mock_result.execution_latency_ms = 500.0
            mock_result.model_architecture = "Xception"
            mock_result.model_version = "v1.0"
            mock_result.raw_score = None
            mock_result.error_message = "Model failed"
            mock_result.diagnostics = {}

            from agni.dashboard_helpers import render_visual_results

            render_visual_results(mock_result)

            mock_st.error.assert_called()
            args, _ = mock_st.error.call_args
            assert "Model failed" in args[0]