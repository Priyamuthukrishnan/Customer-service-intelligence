"""
Gemini Understanding Engine.

Receives pre-assembled ticket context, pre-selected historical tickets,
and pre-selected runbooks. Builds a fully dynamic prompt and calls Gemini
to produce structured JSON understanding output.

No business logic, categories, scores, labels, or mappings are hardcoded
in this module. All understanding is derived by Gemini from the runtime
context provided.
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
    """Render selected runbooks into a compact block (IDs + domain + symptoms only — no steps)."""
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


def _build_prompt(
    ticket: Dict[str, Any],
    comments: List[Dict[str, Any]],
    historical_tickets: List[Dict[str, Any]],
    runbooks: List[Dict[str, Any]],
) -> str:
    """
    Construct the full Gemini prompt from runtime data only.
    No hardcoded categories, sentiments, severities, or mappings.
    """
    ticket_id    = str(ticket.get("id", ""))
    title        = ticket.get("title", "")
    description  = ticket.get("description", "")
    priority     = ticket.get("priority", "Not specified")
    status       = ticket.get("status", "Not specified")
    customer     = ticket.get("customer_name", "Not specified")
    application  = ticket.get("application_name", "Not specified")
    created_at   = str(ticket.get("created_at", "Not specified"))

    comments_block    = _format_comments(comments)
    historical_block  = _format_historical_tickets(historical_tickets)
    runbooks_block    = _format_runbooks(runbooks)

    return f"""You are an expert customer service intelligence analyst.

Analyze the following support ticket and return ONLY a valid JSON object.
Do not include markdown, code fences, or any text outside the JSON.

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
Using only the information provided above, produce a JSON object with
exactly these fields. Do not invent data that is not supported by the context.

{{
  "issue_category":       "<infer the primary issue category from the ticket content>",
  "issue_subcategory":    "<infer a specific sub-topic within the category>",
  "sentiment":            "<descriptive label for the overall customer sentiment expressed in the ticket and comments>",
  "sentiment_score":      <integer 0 to 100; 0 = most negative / hostile, 100 = most positive / satisfied>,
  "impact_level":         "<business impact level: Low | Medium | High | Critical>",
  "impact_reason":        "<one concise sentence explaining why this impact level was assigned>",
  "severity_score":       <integer 0 to 100; derive from priority, sentiment, impact level, and historical patterns together>,
  "keywords":             ["<key term extracted from ticket content>", ...],
  "short_summary":        "<2–3 sentence plain-English summary of the issue>",
  "tags":                 ["<tag characterizing this ticket>", ...],
  "related_ticket_ids":   ["<complaint_id from the historical tickets listed above that are related>", ...],
  "similar_issue_count":  <integer count of historical tickets above that you consider similar to this ticket>,
  "recommended_runbooks": ["<runbook_id from the runbooks listed above that are relevant>", ...]
}}"""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def run_gemini_understanding(
    ticket: Dict[str, Any],
    comments: List[Dict[str, Any]],
    historical_tickets: List[Dict[str, Any]],
    runbooks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Call Gemini with structured JSON response mode and return the parsed dict.

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
    dict: Parsed Gemini JSON output.

    Raises
    ------
    RuntimeError: If Gemini API key is missing, the call fails, or the
                  response cannot be parsed as JSON.
    """
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    prompt = _build_prompt(ticket, comments, historical_tickets, runbooks)
    logger.info(
        "Invoking Gemini model '%s' for ticket_id='%s'",
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
        logger.debug("Gemini raw response for ticket '%s': %s", ticket.get("id"), raw_text)

        result = json.loads(raw_text)
        return result

    except json.JSONDecodeError as e:
        logger.error(
            "Gemini returned non-JSON response for ticket '%s': %s",
            ticket.get("id"),
            raw_text[:500],
        )
        raise RuntimeError(f"Gemini response could not be parsed as JSON: {e}") from e

    except Exception as e:
        logger.error(
            "Gemini API call failed for ticket '%s': %s",
            ticket.get("id"),
            str(e),
            exc_info=True,
        )
        raise RuntimeError(f"Gemini API call failed: {e}") from e
