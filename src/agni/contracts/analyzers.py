"""Data contracts for independent visual and audio-video synchronization analyzers.

Strictly isolates model metrics from presentation layers and enforces disclosure
of non-calibrated model scores.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from agni.contracts.enums import AnalyzerStatus


class VisualAnalysisResult(BaseModel):
    """Outputs and telemetry from the visual deepfake detector component."""

    analyzer_name: str = Field("visual_deepfake_detector", description="Identifier of the visual analyzer")
    model_architecture: Optional[str] = Field(None, description="Pretrained backbone (e.g. Xception, EfficientNet-B4)")
    model_version: Optional[str] = Field(None, description="Model weights tag or commit hash")
    status: AnalyzerStatus = Field(AnalyzerStatus.IDLE, description="Execution status")
    faces_detected_count: int = Field(0, ge=0, description="Number of frames where faces were successfully detected")
    frames_evaluated_count: int = Field(0, ge=0, description="Total sampled frames passed to model inference")
    raw_score: Optional[float] = Field(None, description="Raw model logit or uncalibrated score")
    is_score_calibrated: bool = Field(False, description="Flag confirming whether raw_score represents a true probability")
    decision_threshold: Optional[float] = Field(None, description="Operating threshold used for decision boundary")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic telemetry (e.g. per-frame scores)")
    execution_latency_ms: float = Field(0.0, ge=0.0, description="Inference runtime in milliseconds")
    error_message: Optional[str] = Field(None, description="Diagnostic error string if status is ERROR or UNSUPPORTED")


class SyncAnalysisResult(BaseModel):
    """Outputs and telemetry from the audio-video synchronization analyzer component."""

    analyzer_name: str = Field("av_sync_detector", description="Identifier of the synchronization analyzer")
    model_architecture: Optional[str] = Field(None, description="Pretrained model name (e.g. SyncNet-v2)")
    model_version: Optional[str] = Field(None, description="Model weights checkpoint version")
    status: AnalyzerStatus = Field(AnalyzerStatus.IDLE, description="Execution status")
    has_audio_track: bool = Field(False, description="Whether intake media provided an extractable audio stream")
    mouth_rois_extracted_count: int = Field(0, ge=0, description="Number of mouth crops successfully tracked")
    audio_speech_detected: bool = Field(False, description="Whether non-silent vocal speech features were extracted")
    temporal_offset_frames: Optional[float] = Field(None, description="Estimated audio lead/lag in frame intervals")
    temporal_offset_ms: Optional[float] = Field(None, description="Estimated synchronization offset in milliseconds")
    min_distance_score: Optional[float] = Field(None, description="Minimum euclidean distance between audio & visual embeddings")
    confidence_score: Optional[float] = Field(None, description="Uncalibrated synchronization confidence metric")
    is_score_calibrated: bool = Field(False, description="Flag confirming whether confidence is an empirical probability")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic telemetry (e.g. offset curve, valley depth)")
    execution_latency_ms: float = Field(0.0, ge=0.0, description="Synchronization analysis runtime in milliseconds")
    error_message: Optional[str] = Field(None, description="Diagnostic error string if status is ERROR, NOT_AVAILABLE, etc.")
