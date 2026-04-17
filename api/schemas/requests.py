"""Pydantic models for API requests."""
from __future__ import annotations

from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class GenerateProjectRequest(BaseModel):
    """Request model for project generation.

    Attributes:
        text: Project description/requirements as raw text.
        user_id: Optional user identifier. Auto-generated as UUID if not provided.
        session_id: Optional session identifier for tracking conversation.
        output_format: Output format - "zip" or "folder". Defaults to "zip".
        metadata: Optional metadata dictionary for additional context.
    """

    text: str = Field(..., min_length=1, description="Project description or requirements")
    user_id: Optional[str] = Field(
        None, description="Optional user identifier"
    )
    session_id: Optional[str] = Field(
        None, description="Optional session identifier"
    )
    output_format: Literal["zip", "folder"] = Field(
        "zip", description="Output format"
    )
    metadata: Optional[Dict[str, object]] = Field(
        None, description="Optional metadata dictionary"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Build a CRM system with FastAPI backend and React frontend",
                "user_id": "user_123",
                "session_id": "session_456",
                "output_format": "zip",
                "metadata": {"client": "acme_corp"},
            }
        }
    )
