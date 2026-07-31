"""
Model update router — POST /api/v1/model/update (Section 3.2.2).
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.schemas.api import (
    APIResponse,
    ModelUpdateRequest,
)
from backend.services.model_service import ModelService

router = APIRouter(prefix="/api/v1/model", tags=["model"])


@router.post("/update", response_model=APIResponse)
async def update_model(
    request: ModelUpdateRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Register a model update for deployment.

    Accepts model metadata (version, type, URL, checksum) and
    schedules it for deployment to edge devices.
    The current active model is deactivated.
    """
    try:
        service = ModelService(session)
        result = await service.register_model_update(request)
        return APIResponse(
            code=0,
            message="model update accepted",
            data=result.model_dump(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register model update: {e!s}",
        ) from e


@router.get("/current", response_model=APIResponse)
async def get_current_model(
    session: AsyncSession = Depends(get_session),
):
    """Get the currently active model version."""
    service = ModelService(session)
    model = await service.get_current_model()
    if model is None:
        return APIResponse(code=0, message="no model deployed", data=None)
    return APIResponse(
        code=0,
        message="success",
        data=model.to_dict(),
    )


@router.get("/history", response_model=APIResponse)
async def get_model_history(
    limit: int = 10,
    session: AsyncSession = Depends(get_session),
):
    """Get model version history."""
    service = ModelService(session)
    models = await service.get_model_history(limit=limit)
    return APIResponse(
        code=0,
        message="success",
        data=[m.to_dict() for m in models],
    )
