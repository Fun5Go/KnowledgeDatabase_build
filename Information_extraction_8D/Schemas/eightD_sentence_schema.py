from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class SelectedSentence(BaseModel):
    sentence: str = Field(..., description="Verbatim sentence from the document")
    source: Literal["D2", "D3", "D4"]
    signal_type: Literal["symptom", "failure", "cause", "action", "evidence"]
    confidence: float = Field(..., ge=0.0, le=1.0)

class Iteration1Output(BaseModel):
    selected_sentences: List[SelectedSentence]
