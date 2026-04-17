"""Session & conversation history API."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.dependencies import get_store
from persistence.base import BaseStore
from persistence.models import StoredSession

router = APIRouter(prefix="/api/v1", tags=["sessions"])


class CreateSessionRequest(BaseModel):
    user_id: str | None = None
    metadata: dict = Field(default_factory=dict)


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: str


class SessionSummary(BaseModel):
    session_id: str
    user_id: str | None
    message_count: int
    metadata: dict
    created_at: str
    updated_at: str


class MessageSchema(BaseModel):
    role: str
    content: str
    timestamp: str | None = None


class SessionDetail(BaseModel):
    session_id: str
    user_id: str | None
    messages: List[MessageSchema]
    metadata: dict
    created_at: str
    updated_at: str


class AppendMessageRequest(BaseModel):
    role: str = Field(..., pattern=r"^(system|user|assistant)$")
    content: str = Field(..., min_length=1, max_length=50000)


class PaginatedMessages(BaseModel):
    session_id: str
    messages: List[MessageSchema]
    total: int
    limit: int
    offset: int


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    body: CreateSessionRequest,
    store: BaseStore = Depends(get_store),
) -> CreateSessionResponse:
    session = StoredSession(
        session_id=str(uuid.uuid4()),
        user_id=body.user_id,
        messages=[],
        metadata=body.metadata,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    store.save_session(session)
    return CreateSessionResponse(
        session_id=session.session_id,
        created_at=session.created_at.isoformat(),
    )


@router.get("/sessions", response_model=List[SessionSummary])
async def list_sessions(
    user_id: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    store: BaseStore = Depends(get_store),
) -> List[SessionSummary]:
    sessions = store.list_sessions(user_id=user_id, limit=limit, offset=offset)
    return [
        SessionSummary(
            session_id=s.session_id,
            user_id=s.user_id,
            message_count=len(s.messages),
            metadata=s.metadata,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat(),
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: str,
    store: BaseStore = Depends(get_store),
) -> SessionDetail:
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionDetail(
        session_id=session.session_id,
        user_id=session.user_id,
        messages=[MessageSchema(**m) for m in session.messages],
        metadata=session.metadata,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
    )


@router.get("/sessions/{session_id}/messages", response_model=PaginatedMessages)
async def get_session_messages(
    session_id: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    store: BaseStore = Depends(get_store),
) -> PaginatedMessages:
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    total = len(session.messages)
    messages = store.get_session_messages(session_id, limit=limit, offset=offset)
    return PaginatedMessages(
        session_id=session_id,
        messages=[MessageSchema(**m) for m in messages],
        total=total, limit=limit, offset=offset,
    )


@router.post("/sessions/{session_id}/messages")
async def append_message(
    session_id: str,
    body: AppendMessageRequest,
    store: BaseStore = Depends(get_store),
) -> dict:
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    store.append_message(session_id, body.role, body.content)
    return {"status": "ok", "session_id": session_id}


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    store: BaseStore = Depends(get_store),
) -> dict:
    deleted = store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}
