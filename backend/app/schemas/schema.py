import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict

class CommentBase(BaseModel):
    comment_text: str
    commented_by: Optional[str] = None
    created_at: Optional[datetime] = None

class CommentRead(CommentBase):
    id: uuid.UUID
    ticket_id: uuid.UUID
    
    model_config = ConfigDict(from_attributes=True)


class AIAnalysisBase(BaseModel):
    category_prediction: Optional[str] = None
    similarity_score: Optional[float] = None
    confidence_score: Optional[float] = None
    source_used: Optional[str] = None
    decision_reason: Optional[str] = None
    created_at: Optional[datetime] = None
    runbook_score: Optional[float] = None
    runbook_resolution: Optional[str] = None
    rag_complaint: Optional[str] = None
    rag_resolution: Optional[str] = None

class AIAnalysisRead(AIAnalysisBase):
    id: uuid.UUID
    ticket_id: uuid.UUID
    
    model_config = ConfigDict(from_attributes=True)


class TicketBase(BaseModel):
    jira_issue_key: str
    title: str
    description: str
    status: Optional[str] = None
    priority: Optional[str] = None
    assignee: Optional[str] = None
    category: Optional[str] = None
    customer_name: Optional[str] = None
    application_name: Optional[str] = None
    resolution: Optional[str] = None
    resolution_time: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    jira_resolution_comment_synced_at: Optional[datetime] = None

class TicketRead(TicketBase):
    id: uuid.UUID
    
    model_config = ConfigDict(from_attributes=True)

class TicketDetailRead(TicketRead):
    comments: List[CommentRead] = []
    ai_analysis: Optional[AIAnalysisRead] = None
    
    model_config = ConfigDict(from_attributes=True)


# Qdrant Knowledge Schemas
class HistoricalTicketRead(BaseModel):
    id: str
    complaint_id: str
    complaint: str
    resolution: str
    metadata: Optional[dict] = None


class RunbookRead(BaseModel):
    runbook_id: str
    ticket_level: Optional[str] = None
    domain: Optional[str] = None
    symptoms: List[str] = []
    root_cause: Optional[str] = None
    steps: List[str] = []
    owner: Optional[str] = None
