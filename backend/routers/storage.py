"""Image storage API router."""

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from backend.schemas.api import APIResponse
from backend.storage import get_storage

router = APIRouter(prefix="/api/v1/storage", tags=["storage"])


@router.post("/upload", response_model=APIResponse)
async def upload_image(
    file: UploadFile = File(...),
    device_id: str = Query("unknown"),
    batch_id: str = Query(""),
):
    """Upload a fabric defect image to storage."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "Empty file")

    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    object_name = f"images/{device_id}/{batch_id or 'unknown'}/{ts}_{file.filename}"

    storage = get_storage()
    storage.upload(object_name, data, file.content_type)

    return APIResponse(
        code=0,
        message="uploaded",
        data={"object_name": object_name, "size": len(data)},
    )


@router.get("/download/{object_name:path}", response_model=None)
async def download_image(object_name: str):
    """Download an image from storage."""
    storage = get_storage()
    data = storage.download(object_name)
    if data is None:
        raise HTTPException(404, "Image not found")
    return Response(content=data, media_type="image/jpeg")


@router.get("/list", response_model=APIResponse)
async def list_images(
    device_id: str = Query(""),
    limit: int = Query(50, le=200),
):
    """List stored images, optionally filtered by device."""
    storage = get_storage()
    prefix = f"images/{device_id}/" if device_id else "images/"
    objects = storage.list_objects(prefix=prefix, limit=limit)
    return APIResponse(code=0, message="success", data=objects)


@router.delete("/{object_name:path}", response_model=APIResponse)
async def delete_image(object_name: str):
    """Delete an image from storage."""
    storage = get_storage()
    if not storage.exists(object_name):
        raise HTTPException(404, "Image not found")
    storage.delete(object_name)
    return APIResponse(code=0, message="deleted")
