"""
Model version management service.

Handles model update requests with A/B testing support.
"""

from datetime import datetime, timezone
from typing import Optional
import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.model_registry import ModelVersion
from backend.schemas.api import ModelUpdateRequest, ModelUpdateResponseData


class ModelService:
    """Service for managing model versions and updates."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def register_model_update(
        self, request: ModelUpdateRequest
    ) -> ModelUpdateResponseData:
        """
        Register a new model version for deployment.

        1. Deactivate current active model
        2. Register the new model version
        3. Return deployment estimate
        """
        update_id = f"UPD-{uuid.uuid4().hex[:6].upper()}"

        # Deactivate current model
        await self.session.execute(
            update(ModelVersion)
            .where(ModelVersion.is_current)
            .values(is_current=False, status="retired")
        )

        # Create new model version entry
        model = ModelVersion(
            model_id=request.model_id,
            model_type=request.model_type,
            model_url=request.model_url,
            checksum=request.checksum,
            version_info={
                "release_date": request.version_info.release_date,
                "improvements": request.version_info.improvements,
            },
            accuracy=request.version_info.accuracy,
            status="pending",
            is_current=True,
            release_date=datetime.now(timezone.utc),
        )
        self.session.add(model)
        await self.session.commit()

        return ModelUpdateResponseData(
            update_id=update_id,
            estimated_deploy_time=120,  # seconds
        )

    async def get_current_model(self) -> Optional[ModelVersion]:
        """Get the currently active model version."""
        result = await self.session.execute(
            select(ModelVersion).filter_by(is_current=True)
        )
        return result.scalar_one_or_none()

    async def get_model_history(self, limit: int = 10) -> list[ModelVersion]:
        """Get recent model version history."""
        result = await self.session.execute(
            select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())
