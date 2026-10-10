"""Typed enumeration contracts for AGNI.

Ensures strict categorization of analyzer outputs, uncertainty flags,
decisions, and recommended onboarding actions.
"""

from enum import Enum


class AnalyzerStatus(str, Enum):
    """Execution status of an individual analyzer component."""

    IDLE = "IDLE"
    COMPLETED = "COMPLETED"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNSUPPORTED = "UNSUPPORTED"
    ERROR = "ERROR"


class AssessmentDecision(str, Enum):
    """Deterministic final assessment output by AGNI's uncertainty gate."""

    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    SUSPICIOUS = "SUSPICIOUS"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNSUPPORTED_MEDIA = "UNSUPPORTED_MEDIA"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class ActionRecommendation(str, Enum):
    """Recommended operational next step for the onboarding pipeline."""

    PROCEED_ONBOARDING = "PROCEED_ONBOARDING"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
    RECAPTURE_WITH_AUDIO = "RECAPTURE_WITH_AUDIO"
    RECAPTURE_STABLE_LIGHTING = "RECAPTURE_STABLE_LIGHTING"
    RECAPTURE_CLEAR_FACE = "RECAPTURE_CLEAR_FACE"
    REJECT_APPLICATION = "REJECT_APPLICATION"
    ABORT_UNSUPPORTED_MEDIA = "ABORT_UNSUPPORTED_MEDIA"


class UncertaintyFlag(str, Enum):
    """Identifiable anomalies, missing signals, or boundary conditions."""

    NO_AUDIO_TRACK = "NO_AUDIO_TRACK"
    AUDIO_SILENT = "AUDIO_SILENT"
    NO_FACES_DETECTED = "NO_FACES_DETECTED"
    PARTIAL_FACE_EVIDENCE = "PARTIAL_FACE_EVIDENCE"
    LOW_RESOLUTION = "LOW_RESOLUTION"
    SHORT_DURATION = "SHORT_DURATION"
    CORRUPTED_MEDIA = "CORRUPTED_MEDIA"
    UNSUPPORTED_CONTAINER = "UNSUPPORTED_CONTAINER"
    CONFLICTING_SIGNALS = "CONFLICTING_SIGNALS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    ANALYZER_FAILURE = "ANALYZER_FAILURE"


class MediaType(str, Enum):
    """Classified media container type."""

    VIDEO = "VIDEO"
    AUDIO_ONLY = "AUDIO_ONLY"
    IMAGE_ONLY = "IMAGE_ONLY"
    UNKNOWN = "UNKNOWN"
