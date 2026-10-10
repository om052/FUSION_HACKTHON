"""Data contracts for media intake, container inspection, and stream metadata."""

from typing import Optional
from pydantic import BaseModel, Field

from agni.contracts.enums import MediaType


class VideoStreamMetadata(BaseModel):
    """Extracted technical parameters of the visual video stream."""

    width: int = Field(..., gt=0, description="Horizontal frame resolution in pixels")
    height: int = Field(..., gt=0, description="Vertical frame resolution in pixels")
    fps: float = Field(..., gt=0.0, description="Video playback frames per second")
    frame_count: int = Field(..., ge=0, description="Total frame count in the stream")
    duration_sec: float = Field(..., ge=0.0, description="Stream duration in seconds")
    codec: Optional[str] = Field(None, description="Reported video codec (e.g. h264, mpeg4)")


class AudioStreamMetadata(BaseModel):
    """Extracted technical parameters of the embedded audio stream."""

    has_audio: bool = Field(..., description="Whether an audio stream was detected in the container")
    sample_rate: Optional[int] = Field(None, description="Audio sampling rate in Hertz (e.g. 16000, 44100)")
    channels: Optional[int] = Field(None, description="Number of audio channels (e.g. 1 for mono, 2 for stereo)")
    duration_sec: Optional[float] = Field(None, ge=0.0, description="Audio duration in seconds")
    is_silent: bool = Field(False, description="True if stream contains only digital silence or near-zero RMS")
    codec: Optional[str] = Field(None, description="Reported audio codec (e.g. aac, pcm_s16le)")


class MediaInspectionResult(BaseModel):
    """Comprehensive validation record produced by the intake preprocessor."""

    file_path: str = Field(..., description="Absolute or relative file path to the intake media")
    file_name: str = Field(..., description="Base filename of the uploaded artifact")
    file_size_bytes: int = Field(..., ge=0, description="File size on disk in bytes")
    media_type: MediaType = Field(MediaType.UNKNOWN, description="Classified container type")
    is_valid: bool = Field(..., description="True if media passes container integrity and decodability checks")
    validation_error: Optional[str] = Field(None, description="Detailed failure description if is_valid is False")
    video: Optional[VideoStreamMetadata] = Field(None, description="Video stream metadata if present and valid")
    audio: Optional[AudioStreamMetadata] = Field(None, description="Audio stream metadata if present")
    inspection_latency_ms: float = Field(0.0, ge=0.0, description="Time spent inspecting media in milliseconds")
