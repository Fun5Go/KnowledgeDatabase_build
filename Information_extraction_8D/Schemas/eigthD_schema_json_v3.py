
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from typing import Any, Dict, Optional, Literal




class TextEntity(BaseModel):

    text: Optional[str]                     # exact copied text span
    source_section: Optional[str]            # D2 / D3 / D4 / D5 / D6
    entity_type: Optional[str]                # symptom | cause | action | observation | context


    
class FailureItem(BaseModel):   
    failure_ID: Optional[str]
    failure_level: Literal["system","sub_system"]    
    system_element: Optional[str]
    failure_effect: Optional[str]
    failure_mode: Optional[str]                  
    discipline_type: Optional[Literal["HW", "ESW", "MCH","Other"]]       
    supporting_entities: List[TextEntity] = Field(
                                                    default_factory=list,
                                                    description="Evidence sentences or text spans supporting this failure"
                                                )  
    inferred_insight: Optional[str]
    confidence: float = Field(..., ge=0.0, le=1.0)



class DocumentInfo(BaseModel):
    file_name: str
    product_name: Optional[str] = None
    date: Optional[str] = None

class MaintenaceTag(BaseModel):
    review_status: Literal["validated", "pending", "rejected"] = "pending"
    Version: Optional[str] = None


class D2Section(BaseModel):
    raw_context: str                    # whole D2 section text


class D4Section(BaseModel):
    raw_context: str                    # whole D4 section text

class D3Section(BaseModel):
    raw_context: str                    # whole D4 section text


class D5Section(BaseModel):
    # solutions: List[SolutionItem]    # <-- multiple solutions]
    raw_context: str                    # whole D5 section text

class D6Section(BaseModel):
    raw_context: str

class EightDSections(BaseModel):
    D2: Optional[D2Section]
    D3: Optional[D3Section]
    D4: Optional[D4Section]
    D5: Optional[D5Section] 
    D6: Optional[D6Section]


class EightDCase(BaseModel):
    documents: List[DocumentInfo]
    maintenance_tag: Optional[MaintenaceTag] = None
    system_name: Optional[str] = None
    failures: List[FailureItem]
    sections: EightDSections

    selected_sentences: Optional[Dict[str, Any]] = None

