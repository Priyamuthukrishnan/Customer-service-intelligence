import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Customer Service & Escalation Intelligence Platform"

    NEON_DATABASE_URL: str = os.getenv("NEON_DATABASE_URL", "")

    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    VECTORRAG_COLLECTION: str = os.getenv("VECTORRAG_COLLECTION", "vectorrag")
    INCIDENT_RUNBOOK_COLLECTION: str = os.getenv("INCIDENT_RUNBOOK_COLLECTION", "incidentrunbook")

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

settings = Settings()
