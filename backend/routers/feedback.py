"""
Label feedback router — the human-in-the-loop relabel loop (#16).

Endpoints:
  POST /api/v1/feedback          submit one operator correction
  GET  /api/v1/feedback          list recent corrections (filter by status)
  POST /api/v1/feedback/apply    apply all pending corrections to the labels
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.schemas.api import APIResponse, LabelFeedbackRequest
from backend.services.label_feedback_service import LabelFeedbackService

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


@router.post("", response_model=APIResponse)
async def submit_feedback(
    request: LabelFeedbackRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Submit an operator correction for a detection result.

    The correction is stored as pending; call POST /apply to write it back to
    the YOLO labels. Rejects unknown feedback types or out-of-range class ids.
    """
    try:
        service = LabelFeedbackService(session)
        fb = await service.create_feedback(request)
        return APIResponse(code=0, message="success", data=fb.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.get("", response_model=APIResponse)
async def list_feedback(
    status: Optional[str] = None,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
):
    """List recent corrections, optionally filtered by status."""
    service = LabelFeedbackService(session)
    rows = await service.list_feedback(status=status, limit=limit)
    return APIResponse(code=0, message="success", data=[r.to_dict() for r in rows])


@router.post("/apply", response_model=APIResponse)
async def apply_feedback(
    session: AsyncSession = Depends(get_session),
):
    """
    Apply all pending corrections to the YOLO label files and mark them applied.

    Returns a summary of how many were applied/skipped and whether the
    incremental fine-tune threshold has been reached.
    """
    service = LabelFeedbackService(session)
    summary = await service.apply_pending()
    return APIResponse(code=0, message="success", data=summary)
