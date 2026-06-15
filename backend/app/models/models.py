import uuid
from sqlalchemy import Column, String, Text, Integer, Numeric, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from app.db.database import Base

class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    jira_issue_key = Column(String(100), nullable=False)
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String(50))
    priority = Column(String(20))
    assignee = Column(String(255))
    category = Column(String(100))
    customer_name = Column(String(255))
    application_name = Column(String(255))
    resolution = Column(Text)
    resolution_time = Column(Integer)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    resolved_by = Column(String(100))
    jira_resolution_comment_synced_at = Column(DateTime)

    # Relationships
    comments = relationship("Comment", back_populates="ticket", cascade="all, delete-orphan")
    ai_analysis = relationship("AIAnalysis", back_populates="ticket", uselist=False, cascade="all, delete-orphan")


class Comment(Base):
    __tablename__ = "comments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticket_id = Column(UUID(as_uuid=True), ForeignKey("tickets.id"), nullable=False)
    comment_text = Column(Text, nullable=False)
    commented_by = Column(String(255))
    created_at = Column(DateTime)

    # Relationships
    ticket = relationship("Ticket", back_populates="comments")


class AIAnalysis(Base):
    __tablename__ = "ai_analysis"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticket_id = Column(UUID(as_uuid=True), ForeignKey("tickets.id"), nullable=False)
    category_prediction = Column(String(100))
    similarity_score = Column(Numeric(5, 2))
    confidence_score = Column(Numeric(5, 2))
    source_used = Column(String(50))
    decision_reason = Column(Text)
    created_at = Column(DateTime)
    runbook_score = Column(Numeric(5, 2))
    runbook_resolution = Column(Text)
    rag_complaint = Column(Text)
    rag_resolution = Column(Text)

    # Relationships
    ticket = relationship("Ticket", back_populates="ai_analysis")
