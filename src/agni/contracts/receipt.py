"""Data contracts for the AGNI Evidence Receipt.

Produces an immutable, auditable, and transparent cryptographic/structural
record of evidence, uncertainty, analyzer measurements, and operational recommendations.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from agni.contracts.enums import ActionRecommendation, AssessmentDecision
from agni.contracts.media import MediaInspectionResult
from agni.contracts.analyzers import SyncAnalysisResult, VisualAnalysisResult
from agni.contracts.uncertainty import UncertaintyReport


class EvidenceReceipt(BaseModel):
    """The structured, auditable Evidence Receipt mandated by AGNI innovation #3."""

    receipt_id: str = Field(..., description="Unique UUIDv4 identifying this verification transaction")
    created_at_utc: str = Field(..., description="ISO 8601 UTC timestamp of receipt creation")
    session_id: Optional[str] = Field(None, description="Optional onboarding session identifier")
    system_version: str = Field("AGNI-0.1.0", description="Version of the AGNI verification engine")

    # Analyzer and media telemetry
    media_inspection: MediaInspectionResult = Field(..., description="Inspection results of intake media container")
    visual_analysis: VisualAnalysisResult = Field(..., description="Telemetry from visual detector")
    sync_analysis: SyncAnalysisResult = Field(..., description="Telemetry from audio-video synchronization analyzer")
    uncertainty_report: UncertaintyReport = Field(..., description="Active uncertainty flags and missing signals")

    # Decisioning & Operations
    assessment_decision: AssessmentDecision = Field(..., description="Final assessment verdict from uncertainty gate")
    recommended_action: ActionRecommendation = Field(..., description="Prescribed next step for onboarding flow")
    action_reason: str = Field(..., description="Plain-language justification for the recommended action")

    # Mandatory Scientific Disclosures & Limitations
    limitations_and_disclosures: List[str] = Field(
        default_factory=lambda: [
            "Model scores are uncalibrated raw model indicators and do not represent mathematical probabilities.",
            "Absence of audio or lip sync does not constitute evidence of visual authenticity or manipulation.",
            "Face similarity, blinking, or synchronization alone cannot prove real-world identity authenticity.",
            "Prototype implementation is not calibrated for zero-shot generalization against unseen generative pipelines.",
            "This prototype is designed for evaluation and auditability, not unsupervised production clearance."
        ],
        description="Explicit technical boundaries and operational caveats"
    )

    total_processing_latency_ms: float = Field(0.0, ge=0.0, description="End-to-end processing duration in milliseconds")
