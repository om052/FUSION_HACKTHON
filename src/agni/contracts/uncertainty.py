"""Data contracts for uncertainty detection, missing signal tracking, and gate status."""

from typing import Any, Dict, List
from pydantic import BaseModel, Field

from agni.contracts.enums import UncertaintyFlag


class UncertaintyReport(BaseModel):
    """Structured diagnostic record tracking all uncertainty conditions identified across analyzers."""

    flags: List[UncertaintyFlag] = Field(
        default_factory=list,
        description="Active uncertainty flags raised during media intake and analyzer execution"
    )
    missing_signals: List[str] = Field(
        default_factory=list,
        description="Explicit signals that could not be computed (e.g. 'audio_track', 'lip_motion')"
    )
    conflicting_signals: bool = Field(
        False,
        description="True if independent analyzers produced contradictory authentic/manipulated indications"
    )
    is_gate_closed: bool = Field(
        False,
        description="True if uncertainty prevents reaching an automated definitive decision"
    )
    summary_reason: str = Field(
        ...,
        description="Human-readable explanation of why uncertainty was triggered or passed"
    )
    rule_evaluations: Dict[str, Any] = Field(
        default_factory=dict,
        description="Trace of specific uncertainty heuristics evaluated"
    )
