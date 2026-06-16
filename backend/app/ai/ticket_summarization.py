"""
Gemini Summary Engine.

Receives pre-assembled ticket context, pre-selected historical tickets,
and pre-selected runbooks. Builds a fully dynamic prompt and calls Gemini
to produce structured JSON summary output.

No business logic, risk levels, status messages, or mappings are hardcoded
in this module. All summary fields are derived by Gemini from the runtime
context provided.

One Gemini call per summary request. This module is intentionally independent
of the Ticket Understanding Service to avoid chained Gemini calls.
"""
import json
import logging
from typing import Any, Dict, List

from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builder helpers
# ---------------------------------------------------------------------------

def _format_comments(comments: List[Dict[str, Any]]) -> str:
    """Render comment list into a numbered plain-text block."""
    if not comments:
        return "No comments."
    lines = []
    for i, c in enumerate(comments, start=1):
        author = c.get("commented_by") or "Unknown"
        text = c.get("comment_text", "").strip()
        lines.append(f"[{i}] {author}: {text}")
    return "\n".join(lines)


def _format_historical_tickets(tickets: List[Dict[str, Any]]) -> str:
    """Render selected historical tickets into a compact block."""
    if not tickets:
        return "No historical context available."
    lines = []
    for t in tickets:
        cid = t.get("complaint_id") or t.get("id") or "N/A"
        complaint = (t.get("complaint") or "").strip()
        resolution = (t.get("resolution") or "").strip()
        lines.append(f"ID: {cid} | Issue: {complaint} | Resolution: {resolution}")
    return "\n".join(lines)


def _format_runbooks(runbooks: List[Dict[str, Any]]) -> str:
    """Render selected runbooks into a compact block (IDs + domain + symptoms only)."""
    if not runbooks:
        return "No runbook context available."
    lines = []
    for rb in runbooks:
        rid = rb.get("runbook_id") or "N/A"
        domain = rb.get("domain") or "N/A"
        symptoms = rb.get("symptoms", [])
        if isinstance(symptoms, list):
            symptoms_text = "; ".join(str(s) for s in symptoms)
        else:
            symptoms_text = str(symptoms)
        lines.append(f"ID: {rid} | Domain: {domain} | Symptoms: {symptoms_text}")
    return "\n".join(lines)


def _build_summary_prompt(
    ticket: Dict[str, Any],
    comments: List[Dict[str, Any]],
    historical_tickets: List[Dict[str, Any]],
    runbooks: List[Dict[str, Any]],
) -> str:
    """
    Construct the full Gemini summary prompt from runtime data only.
    No hardcoded categories, risk levels, sentiments, or mappings.
    """
    ticket_id   = str(ticket.get("id", ""))
    title       = ticket.get("title", "")
    description = ticket.get("description", "")
    priority    = ticket.get("priority", "Not specified")
    status      = ticket.get("status", "Not specified")
    customer    = ticket.get("customer_name", "Not specified")
    application = ticket.get("application_name", "Not specified")
    created_at  = str(ticket.get("created_at", "Not specified"))

    comments_block   = _format_comments(comments)
    historical_block = _format_historical_tickets(historical_tickets)
    runbooks_block   = _format_runbooks(runbooks)

    return f"""You are an expert customer service intelligence analyst.

Generate a BUSINESS-FRIENDLY summary of the following support ticket for use
by support teams, managers, and leadership.

Return ONLY a valid JSON object. Do not include markdown, code fences, or any
text outside the JSON.

=== TICKET CONTEXT ===
Ticket ID   : {ticket_id}
Title       : {title}
Description : {description}
Priority    : {priority}
Status      : {status}
Customer    : {customer}
Application : {application}
Created At  : {created_at}

=== CUSTOMER COMMENTS ===
{comments_block}

=== SIMILAR HISTORICAL TICKETS ===
{historical_block}

=== RELEVANT RUNBOOKS ===
{runbooks_block}

=== INSTRUCTIONS ===
Using ONLY the information provided above, produce a JSON object with exactly
these fields. Do not invent data that is not supported by the context.

{{
  "summary":                 "<2–4 sentence plain-English overview of the ticket>",
  "customer_problem":        "<what the customer is actually experiencing>",
  "current_status":          "<current condition derived from ticket status and comment thread>",
  "business_impact":         "<business impact in plain English>",
  "sentiment_summary":       "<customer tone, urgency, and emotional state>",
  "key_events":              ["<timeline event extracted from ticket details or comments>", ...],
  "actions_taken":           ["<completed action mentioned in the ticket or comments>", ...],
  "pending_actions":         ["<outstanding item not yet resolved>", ...],
  "risk_level":              "<infer from priority, sentiment, business impact, and historical patterns>",
  "recommended_next_action": "<specific actionable next step for the support team>",
  "related_ticket_ids":      ["<complaint_id from the historical tickets listed above>", ...],
  "recommended_runbooks":    ["<runbook_id from the runbooks listed above>", ...]
}}"""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def run_gemini_summary(
    ticket: Dict[str, Any],
    comments: List[Dict[str, Any]],
    historical_tickets: List[Dict[str, Any]],
    runbooks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Call Gemini with structured JSON response mode and return the parsed summary dict.

    Parameters
    ----------
    ticket:             Ticket data dict from Neon PostgreSQL.
    comments:           List of comment dicts for the ticket.
    historical_tickets: Pre-selected top-K historical tickets from Qdrant
                        (selected upstream by token overlap scoring).
    runbooks:           Pre-selected top-K runbooks from Qdrant
                        (selected upstream by token overlap scoring).

    Returns
    -------
    dict: Parsed Gemini JSON output containing all summary fields.

    Raises
    ------
    RuntimeError: If GEMINI_API_KEY is missing, the API call fails, or the
                  response cannot be parsed as JSON.
    """
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    prompt = _build_summary_prompt(ticket, comments, historical_tickets, runbooks)
    logger.info(
        "Invoking Gemini model '%s' for ticket summary — ticket_id='%s'",
        settings.GEMINI_MODEL,
        ticket.get("id"),
    )

    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,        # low temperature → consistent structured output
            ),
        )

        raw_text = response.text.strip()
        logger.debug(
            "Gemini summary raw response for ticket '%s': %s",
            ticket.get("id"),
            raw_text,
        )

        result = json.loads(raw_text)
        return result

    except json.JSONDecodeError as e:
        logger.error(
            "Gemini returned non-JSON summary response for ticket '%s': %s",
            ticket.get("id"),
            raw_text[:500],
        )
        raise RuntimeError(f"Gemini summary response could not be parsed as JSON: {e}") from e

    except Exception as e:
        logger.error(
            "Gemini summary API call failed for ticket '%s': %s",
            ticket.get("id"),
            str(e),
            exc_info=True,
        )
        raise RuntimeError(f"Gemini summary API call failed: {e}") from e
