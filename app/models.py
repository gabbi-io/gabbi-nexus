from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    chunk_id: str
    file_name: str
    file_type: str
    page_or_sheet: Optional[str] = None
    text: str
    score: float = 0.0


class FileArtifact(BaseModel):
    file_id: str
    file_name: str
    file_type: str
    size_bytes: int
    extracted_text_chars: int = 0
    tables_detected: int = 0
    columns_detected: List[str] = Field(default_factory=list)
    quality_flags: List[str] = Field(default_factory=list)
    summary: str = ''
    top_entities: List[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    executive_summary: str
    key_findings: List[str] = Field(default_factory=list)
    relevant_fields: List[str] = Field(default_factory=list)
    detected_rules: List[str] = Field(default_factory=list)
    exceptions: List[str] = Field(default_factory=list)
    quality_issues: List[str] = Field(default_factory=list)


class Diagnosis(BaseModel):
    process_objective: str
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    primary_entities: List[str] = Field(default_factory=list)
    decision_rules: List[str] = Field(default_factory=list)
    exceptions: List[str] = Field(default_factory=list)
    bottlenecks: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    automation_suggestions: List[str] = Field(default_factory=list)
    recommended_approach: List[str] = Field(default_factory=list)
    rag_seed: List[str] = Field(default_factory=list)
    workflow_blueprint: List[str] = Field(default_factory=list)
    human_validation_checklist: List[str] = Field(default_factory=list)


class CaseRecord(BaseModel):
    case_id: str
    title: str
    created_at: str
    updated_at: str
    files: List[FileArtifact] = Field(default_factory=list)
    analysis: Optional[AnalysisResult] = None
    diagnosis: Optional[Diagnosis] = None
    chunk_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateCaseRequest(BaseModel):
    title: str


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk] = Field(default_factory=list)
    mode: str


class ParseOutput(BaseModel):
    file_name: str
    file_type: str
    text: str
    table_headers: List[str] = Field(default_factory=list)
    table_rows_preview: List[Dict[str, Any]] = Field(default_factory=list)
    tables_detected: int = 0
    quality_flags: List[str] = Field(default_factory=list)
    top_entities: List[str] = Field(default_factory=list)
    sheets_or_sections: List[str] = Field(default_factory=list)
