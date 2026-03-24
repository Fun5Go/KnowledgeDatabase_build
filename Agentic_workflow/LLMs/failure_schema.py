from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any

Confidence = Literal["high", "medium", "low"]
Support_type = Literal ["complete_entity", "composed_from_multiple" ,"partial_pattern", "no_direct_gt"]


class Cause_Modes(BaseModel):
    failure_cause: str = Field(..., description="Failure cause")
    failure_mode: str = Field(..., description="Failure mode")
    # failure_mode: List[str] = Field(..., description="Failure modes")

class Effect_Mode(BaseModel):
    failure_mode: str = Field(..., description="Failure mode")
    failure_effect: str = Field(..., description="Failure effect")
    # failure_effect: List[str] = Field(..., description="Failure effect")
