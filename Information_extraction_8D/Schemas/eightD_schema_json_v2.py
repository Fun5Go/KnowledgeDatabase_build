from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class DisciplineType(str, Enum):
    HW = "HW"
    ESW = "ESW"
    MCH = "MCH"
    OTHER = "Other"
class TextEntity(BaseModel):
    text: Optional[str]                     # exact copied text span
    source_section: Optional[str]            # D2 / D3 / D4 / D5 / D6
    entity_type: Optional[str]                # symptom | cause | action | observation | context
    
class FailureItem(BaseModel):        
    system_element: Optional[str]
    failure_effect: Optional[str]
    failure_mode: Optional[str]                  
    discipline_type: Optional[DisciplineType] = None        
    supporting_entities: List[TextEntity]   
    inferred_insight: Optional[str]              



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
    d8_id: str
    product_name: Optional[str] = None
    system_name: Optional[str] = None
    failures: List[FailureItem]
    sections: EightDSections
