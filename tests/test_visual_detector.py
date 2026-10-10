"""Unit and smoke tests for the real Xception and YuNet visual deepfake detector."""

from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

from agni.contracts.enums import AnalyzerStatus
from agni.models.visual.xception import Xception, load_xception_detector
from agni.models.visual.detector import VisualDeepfakeDetector


YUNET_PATH = Path("models/yunet/face_detection_yunet_2023mar.onnx")
XCEPTION_PATH = Path("models/visual_detector/xception_best.pth")
REAL_FACE_PATH = Path("tests/data/sample_face.jpg")


@pytest.fixture(scope="session")
def real_face_image() -> np.ndarray:
    """Loads a real face image for smoke testing."""
    assert REAL_FACE_PATH.exists(), f"Real face sample missing at {REAL_FACE_PATH}"
    img = cv2.imread(str(REAL_FACE_PATH))
    assert img is not None and img.size > 0
    return img


@pytest.fixture(scope="session")
def detector() -> VisualDeepfakeDetector:
    """Instantiates the real detector with verified model artifacts."""
    return VisualDeepfakeDetector(
        yunet_model_path=YUNET_PATH,
        xception_checkpoint_path=XCEPTION_PATH,
    )


def test_checkpoint_strict_load():
    """Invariant: Checkpoint must load with ZERO missing and ZERO unexpected keys."""
    model = load_xception_detector(XCEPTION_PATH)
    assert isinstance(model, Xception)
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 21861162  # Exact DeepfakeBench Xception parameter count


def test_xception_output_shape_and_classes():
    """Verifies output logit dimensions match (batch_size, 2)."""
    model = load_xception_detector(XCEPTION_PATH)
    dummy_input = torch.zeros((2, 3, 256, 256), dtype=torch.float32)
    with torch.no_grad():
        logits, feats = model(dummy_input)

    assert logits.shape == (2, 2)
    assert feats.shape == (2, 2048, 8, 8)
    assert not torch.isnan(logits).any()


def test_detector_preprocessing(detector: VisualDeepfakeDetector, real_face_image: np.ndarray):
    """Verifies image normalization to [-1.0, 1.0] and 256x256 shape."""
    tensor = detector.preprocess_face(real_face_image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 256, 256)
    assert tensor.min() >= -1.01
    assert tensor.max() <= 1.01


def test_face_detection_on_real_face(detector: VisualDeepfakeDetector, real_face_image: np.ndarray):
    """Verifies YuNet accurately detects and crops a real face."""
    crop = detector.detect_and_crop_face(real_face_image)

    assert crop is not None
    assert isinstance(crop, np.ndarray)
    assert crop.shape[0] > 50
    assert crop.shape[1] > 50


def test_face_detection_on_blank_frame_returns_none(detector: VisualDeepfakeDetector):
    """Verifies blank/noise frame yields None (no false positive face)."""
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    crop = detector.detect_and_crop_face(blank)

    assert crop is None


def test_analyze_frames_with_no_face_returns_not_available(detector: VisualDeepfakeDetector):
    """CRITICAL RULE: No face must yield NOT_AVAILABLE with raw_score=None."""
    blank_frames = [np.zeros((300, 300, 3), dtype=np.uint8) for _ in range(3)]
    result = detector.analyze_frames(blank_frames)

    assert result.status == AnalyzerStatus.NOT_AVAILABLE
    assert result.faces_detected_count == 0
    assert result.frames_evaluated_count == 3
    assert result.raw_score is None
    assert result.is_score_calibrated is False
    assert result.execution_latency_ms > 0.0


def test_real_inference_smoke_test(detector: VisualDeepfakeDetector, real_face_image: np.ndarray):
    """Real inference smoke test on an actual photograph face."""
    frames = [real_face_image, real_face_image]
    result = detector.analyze_frames(frames)

    assert result.status == AnalyzerStatus.COMPLETED
    assert result.faces_detected_count == 2
    assert result.frames_evaluated_count == 2
    assert result.raw_score is not None
    assert 0.0 <= result.raw_score <= 1.0
    assert result.is_score_calibrated is False
    assert result.execution_latency_ms > 0.0

    # Diagnostic checks
    assert "per_frame_fake_scores" in result.diagnostics
    assert len(result.diagnostics["per_frame_fake_scores"]) == 2
    assert result.diagnostics["face_detection_rate"] == 1.0


def test_analyzer_error_handling_on_invalid_files():
    """Verifies clean FileNotFoundError when model path is non-existent."""
    with pytest.raises(FileNotFoundError):
        VisualDeepfakeDetector(
            yunet_model_path=Path("non_existent_yunet.onnx"),
            xception_checkpoint_path=XCEPTION_PATH,
        )
