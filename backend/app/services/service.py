import logging
import json
import uuid
import httpx
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.models.models import Ticket, Comment, AIAnalysis
from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Neon DB services ---

def get_tickets(db: Session, skip: int = 0, limit: int = 100) -> List[Ticket]:
    try:
        return db.query(Ticket).offset(skip).limit(limit).all()
    except Exception as e:
        logger.error(f"Error querying tickets from database: {e}", exc_info=True)
        raise

def get_ticket_by_id(db: Session, ticket_id: uuid.UUID) -> Optional[Ticket]:
    try:
        return db.query(Ticket).filter(Ticket.id == ticket_id).first()
    except Exception as e:
        logger.error(f"Error querying ticket by id {ticket_id} from database: {e}", exc_info=True)
        raise

def get_ticket_comments(db: Session, ticket_id: uuid.UUID) -> List[Comment]:
    try:
        return db.query(Comment).filter(Comment.ticket_id == ticket_id).all()
    except Exception as e:
        logger.error(f"Error querying comments for ticket {ticket_id} from database: {e}", exc_info=True)
        raise

def get_ticket_analysis(db: Session, ticket_id: uuid.UUID) -> Optional[AIAnalysis]:
    try:
        return db.query(AIAnalysis).filter(AIAnalysis.ticket_id == ticket_id).first()
    except Exception as e:
        logger.error(f"Error querying AI analysis for ticket {ticket_id} from database: {e}", exc_info=True)
        raise


# --- Qdrant Vector DB services ---

def get_qdrant_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.QDRANT_API_KEY:
        headers["api-key"] = settings.QDRANT_API_KEY
    return headers

async def get_historical_tickets() -> List[Dict[str, Any]]:
    if not settings.QDRANT_URL or not settings.VECTORRAG_COLLECTION:
        logger.error("Qdrant URL or collection name is not configured.")
        return []

    url = f"{settings.QDRANT_URL}/collections/{settings.VECTORRAG_COLLECTION}/points/scroll"
    headers = get_qdrant_headers()
    body = {
        "limit": 100,
        "with_payload": True,
        "with_vector": False
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=body, timeout=10.0)
            if response.status_code != 200:
                logger.error(f"Qdrant vectorrag scroll query failed: status={response.status_code}, response={response.text}")
                return []

            data = response.json()
            points = data.get("result", {}).get("points", [])
            tickets = []
            for p in points:
                payload = p.get("payload", {})
                text_str = payload.get("text", "")

                complaint_id = ""
                complaint = ""
                resolution = ""

                if text_str:
                    try:
                        inner = json.loads(text_str)
                        complaint_id = inner.get("complaint_id", "")
                        complaint = inner.get("complaint", "")
                        resolution = inner.get("resolution", "")
                    except Exception as parse_err:
                        logger.warning(f"Could not parse inner json text for point {p.get('id')}: {parse_err}")

                tickets.append({
                    "id": p.get("id"),
                    "complaint_id": complaint_id or payload.get("complaint_id") or "",
                    "complaint": complaint or payload.get("complaint") or text_str,
                    "resolution": resolution or payload.get("resolution") or "",
                    "metadata": payload
                })
            return tickets
    except Exception as e:
        logger.error(f"Error fetching historical tickets from Qdrant: {e}", exc_info=True)
        return []

def parse_json_objects(text: str) -> List[Dict[str, Any]]:
    objs = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        start = text.find('{', pos)
        if start == -1:
            break
        try:
            obj, idx = decoder.raw_decode(text[start:])
            objs.append(obj)
            pos = start + idx
        except json.JSONDecodeError:
            pos = start + 1
    return objs

async def get_runbooks() -> List[Dict[str, Any]]:
    if not settings.QDRANT_URL or not settings.INCIDENT_RUNBOOK_COLLECTION:
        logger.error("Qdrant URL or incident collection name is not configured.")
        return []

    url = f"{settings.QDRANT_URL}/collections/{settings.INCIDENT_RUNBOOK_COLLECTION}/points/scroll"
    headers = get_qdrant_headers()
    body = {
        "limit": 100,
        "with_payload": True,
        "with_vector": False
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=body, timeout=10.0)
            if response.status_code != 200:
                logger.error(f"Qdrant incidentrunbook scroll query failed: status={response.status_code}, response={response.text}")
                return []

            data = response.json()
            points = data.get("result", {}).get("points", [])
            runbooks_map = {}

            for p in points:
                payload = p.get("payload", {})
                text_str = payload.get("text", "")

                objs = parse_json_objects(text_str)
                for obj in objs:
                    runbook_id = obj.get("runbook_id")
                    if runbook_id:
                        symptoms = obj.get("symptoms", [])
                        if isinstance(symptoms, str):
                            symptoms = [symptoms]
                        steps = obj.get("steps", [])
                        if isinstance(steps, str):
                            steps = [steps]

                        runbooks_map[runbook_id] = {
                            "runbook_id": runbook_id,
                            "ticket_level": obj.get("ticket_level"),
                            "domain": obj.get("domain"),
                            "symptoms": symptoms,
                            "root_cause": obj.get("root_cause"),
                            "steps": steps,
                            "owner": obj.get("owner")
                        }

            return list(runbooks_map.values())
    except Exception as e:
        logger.error(f"Error fetching runbooks from Qdrant: {e}", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Ticket Understanding Service
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set:
    """
    Produce a set of lowercase tokens (length >= 3) from free text.
    Used ONLY to score relevance of Qdrant documents for context selection.
    The resulting scores are never returned in the API response.
    """
    return {word for word in text.lower().split() if len(word) >= 3}


def _score_historical_ticket(ticket_tokens: set, item: dict) -> int:
    """Count token overlaps for a single historical ticket (context selection only)."""
    candidate_text = " ".join([
        item.get("complaint") or "",
        item.get("resolution") or "",
    ])
    candidate_tokens = _tokenize(candidate_text)
    return len(ticket_tokens & candidate_tokens)


def _score_runbook(ticket_tokens: set, runbook: dict) -> int:
    """Count token overlaps for a single runbook (context selection only)."""
    symptoms = runbook.get("symptoms", [])
    if isinstance(symptoms, list):
        symptoms_text = " ".join(str(s) for s in symptoms)
    else:
        symptoms_text = str(symptoms)

    candidate_text = " ".join([
        symptoms_text,
        runbook.get("domain") or "",
        runbook.get("root_cause") or "",
    ])
    candidate_tokens = _tokenize(candidate_text)
    return len(ticket_tokens & candidate_tokens)


async def get_ticket_understanding(db: Session, ticket_id: uuid.UUID) -> dict:
    """
    Orchestrate all four layers of the Ticket Understanding Service.

    Layer 1 – Ticket Data Loader:
        Fetch ticket and comments from Neon PostgreSQL.

    Layer 2 – Knowledge Context Retriever:
        Scroll Qdrant collections, score by token overlap (selection only),
        select top-5 historical tickets and top-3 runbooks.

    Layer 3 – Gemini Understanding Engine:
        Delegate to ai.ticket_understanding.run_gemini_understanding().
        Gemini generates all output fields dynamically.

    Layer 4 – Response Builder:
        Construct TicketUnderstandingResponse from Gemini output.
    """
    from app.ai.ticket_understanding import run_gemini_understanding
    from app.schemas.schema import TicketUnderstandingResponse

    # ------------------------------------------------------------------
    # Layer 1: Ticket Data Loader
    # ------------------------------------------------------------------
    ticket = get_ticket_by_id(db, ticket_id=ticket_id)
    if not ticket:
        return None

    comments = get_ticket_comments(db, ticket_id=ticket_id)

    # Serialise ORM objects into plain dicts for downstream layers
    ticket_dict = {
        "id": str(ticket.id),
        "title": ticket.title,
        "description": ticket.description,
        "priority": ticket.priority,
        "status": ticket.status,
        "customer_name": ticket.customer_name,
        "application_name": ticket.application_name,
        "created_at": str(ticket.created_at) if ticket.created_at else None,
    }

    comment_dicts = [
        {
            "comment_text": c.comment_text,
            "commented_by": c.commented_by,
        }
        for c in comments
    ]

    # ------------------------------------------------------------------
    # Layer 2: Knowledge Context Retriever
    # ------------------------------------------------------------------

    # Build search text from ticket + comments — used for token scoring only
    comment_texts = " ".join(c.comment_text for c in comments if c.comment_text)
    search_text = f"{ticket.title} {ticket.description} {comment_texts}"
    ticket_tokens = _tokenize(search_text)

    # Fetch all points from both Qdrant collections (reuse existing pattern)
    all_historical = await get_historical_tickets()
    all_runbooks = await get_runbooks()

    # Score and select top-5 historical tickets (score used for selection only)
    scored_historical = sorted(
        all_historical,
        key=lambda item: _score_historical_ticket(ticket_tokens, item),
        reverse=True,
    )
    selected_historical = scored_historical[:5]

    # Score and select top-3 runbooks (score used for selection only)
    scored_runbooks = sorted(
        all_runbooks,
        key=lambda rb: _score_runbook(ticket_tokens, rb),
        reverse=True,
    )
    selected_runbooks = scored_runbooks[:3]

    logger.info(
        "Context selected for ticket '%s': %d historical tickets, %d runbooks",
        ticket_id,
        len(selected_historical),
        len(selected_runbooks),
    )

    # ------------------------------------------------------------------
    # Layer 3: Gemini Understanding Engine
    # ------------------------------------------------------------------
    gemini_output = await run_gemini_understanding(
        ticket=ticket_dict,
        comments=comment_dicts,
        historical_tickets=selected_historical,
        runbooks=selected_runbooks,
    )

    # ------------------------------------------------------------------
    # Layer 4: Response Builder
    # ------------------------------------------------------------------
    response = TicketUnderstandingResponse(
        ticket_id=str(ticket_id),
        issue_category=gemini_output.get("issue_category", ""),
        issue_subcategory=gemini_output.get("issue_subcategory", ""),
        sentiment=gemini_output.get("sentiment", ""),
        sentiment_score=float(gemini_output.get("sentiment_score", 0)),
        impact_level=gemini_output.get("impact_level", ""),
        impact_reason=gemini_output.get("impact_reason", ""),
        severity_score=float(gemini_output.get("severity_score", 0)),
        keywords=gemini_output.get("keywords") or [],
        short_summary=gemini_output.get("short_summary", ""),
        tags=gemini_output.get("tags") or [],
        related_ticket_ids=gemini_output.get("related_ticket_ids") or [],
        similar_issue_count=int(gemini_output.get("similar_issue_count", 0)),
        recommended_runbooks=gemini_output.get("recommended_runbooks") or [],
    )
    return response


# ---------------------------------------------------------------------------
# Ticket Summary Service
# ---------------------------------------------------------------------------

async def get_ticket_summary(db: Session, ticket_id: uuid.UUID) -> Optional[dict]:
    """
    Orchestrate all four layers of the Ticket Summary Service.

    This function is fully standalone — it does NOT call get_ticket_understanding()
    and therefore triggers exactly ONE Gemini call per request.

    Layer 1 – Ticket Data Loader:
        Fetch ticket and comments from Neon PostgreSQL.

    Layer 2 – Qdrant Context Retriever:
        Scroll Qdrant collections, score by token overlap (selection only),
        select top-5 historical tickets and top-3 runbooks.

    Layer 3 – Gemini Summary Engine:
        Delegate to ai.ticket_summarization.run_gemini_summary().
        Gemini generates all output fields dynamically from runtime context.

    Layer 4 – Response Builder:
        Construct TicketSummaryResponse from Gemini output.
    """
    from app.ai.ticket_summarization import run_gemini_summary
    from app.schemas.schema import TicketSummaryResponse

    # ------------------------------------------------------------------
    # Layer 1: Ticket Data Loader
    # ------------------------------------------------------------------
    ticket = get_ticket_by_id(db, ticket_id=ticket_id)
    if not ticket:
        return None

    comments = get_ticket_comments(db, ticket_id=ticket_id)

    # Serialise ORM objects into plain dicts for downstream layers
    ticket_dict = {
        "id": str(ticket.id),
        "title": ticket.title,
        "description": ticket.description,
        "priority": ticket.priority,
        "status": ticket.status,
        "customer_name": ticket.customer_name,
        "application_name": ticket.application_name,
        "created_at": str(ticket.created_at) if ticket.created_at else None,
    }

    comment_dicts = [
        {
            "comment_text": c.comment_text,
            "commented_by": c.commented_by,
        }
        for c in comments
    ]

    # ------------------------------------------------------------------
    # Layer 2: Qdrant Context Retriever
    # ------------------------------------------------------------------

    # Build search text from ticket + comments — used for token scoring only
    comment_texts = " ".join(c.comment_text for c in comments if c.comment_text)
    search_text = f"{ticket.title} {ticket.description} {comment_texts}"
    ticket_tokens = _tokenize(search_text)

    # Fetch all points from both Qdrant collections (reuse existing functions)
    all_historical = await get_historical_tickets()
    all_runbooks = await get_runbooks()

    # Score and select top-5 historical tickets (score used for selection only)
    scored_historical = sorted(
        all_historical,
        key=lambda item: _score_historical_ticket(ticket_tokens, item),
        reverse=True,
    )
    selected_historical = scored_historical[:5]

    # Score and select top-3 runbooks (score used for selection only)
    scored_runbooks = sorted(
        all_runbooks,
        key=lambda rb: _score_runbook(ticket_tokens, rb),
        reverse=True,
    )
    selected_runbooks = scored_runbooks[:3]

    logger.info(
        "Summary context selected for ticket '%s': %d historical tickets, %d runbooks",
        ticket_id,
        len(selected_historical),
        len(selected_runbooks),
    )

    # ------------------------------------------------------------------
    # Layer 3: Gemini Summary Engine (single Gemini call)
    # ------------------------------------------------------------------
    gemini_output = await run_gemini_summary(
        ticket=ticket_dict,
        comments=comment_dicts,
        historical_tickets=selected_historical,
        runbooks=selected_runbooks,
    )

    # ------------------------------------------------------------------
    # Layer 4: Response Builder
    # ------------------------------------------------------------------
    response = TicketSummaryResponse(
        ticket_id=str(ticket_id),
        ticket_title=ticket.title,                                       # from Neon, not Gemini
        summary=gemini_output.get("summary", ""),
        customer_problem=gemini_output.get("customer_problem", ""),
        current_status=gemini_output.get("current_status", ""),
        business_impact=gemini_output.get("business_impact", ""),
        sentiment_summary=gemini_output.get("sentiment_summary", ""),
        key_events=gemini_output.get("key_events") or [],
        actions_taken=gemini_output.get("actions_taken") or [],
        pending_actions=gemini_output.get("pending_actions") or [],
        risk_level=gemini_output.get("risk_level", ""),
        recommended_next_action=gemini_output.get("recommended_next_action", ""),
        related_ticket_ids=gemini_output.get("related_ticket_ids") or [],
        recommended_runbooks=gemini_output.get("recommended_runbooks") or [],
    )
    return response

