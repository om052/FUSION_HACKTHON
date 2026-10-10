"""Visual models exports."""

from agni.models.visual.xception import Xception, load_xception_detector
from agni.models.visual.detector import VisualDeepfakeDetector

__all__ = ["Xception", "load_xception_detector", "VisualDeepfakeDetector"]
