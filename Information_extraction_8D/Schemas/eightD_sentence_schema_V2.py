from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class AnnotationItem(BaseModel):
    entity_type: Literal[
        "symptom",
        "condition",
        "occurrence",
        "investigation",
        "root_cause_evidence"
    ] = Field(
        ...,
        description="Type of factual signal represented by the sentence"
    )
    assertion_level: Literal[
        "observed",
        "confirmed",
        "ruled_out",
        "suspected"
    ] = Field(
        ...,
        description="How strongly the fact is asserted in the source text"
    )
    faithful_score: Optional[int] = None
    faithful_type: Literal["exact","partial","fuzzy","hallucinated"] = None

class SelectedSentence(BaseModel):
    id: Optional[str] = Field(
        default=None,
        description="Optional unique identifier for traceability"
    )
    text: str = Field(
        ...,
        description="Atomic factual sentence extracted from the document; light rephrasing allowed for clarity"
    )

    source_section: Literal["D2", "D3", "D4"] = Field(
        ...,
        description="Origin section of the sentence in the 8D report"
    )
    annotations: AnnotationItem = Field(        
        ...,
        description="Description of the sentence en"
    )




class Iteration1Output(BaseModel):
    selected_sentences: List[SelectedSentence]
