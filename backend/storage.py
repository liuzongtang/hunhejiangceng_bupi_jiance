"""
MinIO object storage client for fabric defect images.

Supports upload, download, list, and presigned URL generation.
Works with MinIO, AWS S3, and compatible object stores.
"""

from __future__ import annotations

from datetime import timedelta
import io
import logging
import os
from typing import Dict, List, Optional

from backend.config import MinioConfig, get_config

logger = logging.getLogger(__name__)


class ImageStorage:
    """
    MinIO/S3-compatible image storage client.

    Usage:
        storage = ImageStorage()
        storage.upload("images/defect_001.jpg", image_bytes)
        url = storage.get_presigned_url("images/defect_001.jpg")
    """

    def __init__(self, config: Optional[MinioConfig] = None):
        self.config = config or get_config().minio
        self._client = None

    @property
    def client(self):
        """Lazy-initialize MinIO client."""
        if self._client is None:
            try:
                from minio import Minio

                self._client = Minio(
                    self.config.endpoint,
                    access_key=self.config.access_key,
                    secret_key=self.config.secret_key,
                    secure=self.config.secure,
                )
                self._ensure_bucket()
            except ImportError:
                logger.warning("minio not installed — using local file storage")
                self._client = None
        return self._client

    def _ensure_bucket(self):
        """Create bucket if it doesn't exist."""
        if self._client and not self._client.bucket_exists(self.config.bucket):
            self._client.make_bucket(self.config.bucket)
            logger.info(f"Created bucket: {self.config.bucket}")

    # ------------------------------------------------------------------
    # Local fallback (no MinIO dependency)
    # ------------------------------------------------------------------
    _local_dir: Optional[str] = None

    @property
    def local_dir(self) -> str:
        if self._local_dir is None:
            self._local_dir = os.path.join(
                os.path.dirname(__file__), "..", "data", "images"
            )
            os.makedirs(self._local_dir, exist_ok=True)
        return self._local_dir

    def _local_path(self, object_name: str) -> str:
        return os.path.join(self.local_dir, object_name.replace("/", os.sep))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload(
        self, object_name: str, data: bytes, content_type: str = "image/jpeg"
    ) -> str:
        """
        Upload an image to storage.

        Args:
            object_name: Object key (e.g., "images/defect_001.jpg").
            data: Raw image bytes.
            content_type: MIME type.

        Returns:
            Object path/URL.
        """
        if self._client:
            self.client.put_object(
                self.config.bucket,
                object_name,
                io.BytesIO(data),
                length=len(data),
                content_type=content_type,
            )
            logger.info(f"Uploaded: {object_name} ({len(data)} bytes)")
        else:
            # Local fallback
            path = self._local_path(object_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            logger.info(f"Saved locally: {path} ({len(data)} bytes)")
        return object_name

    def download(self, object_name: str) -> Optional[bytes]:
        """Download an image from storage."""
        if self._client:
            try:
                response = self.client.get_object(self.config.bucket, object_name)
                data = response.read()
                response.close()
                response.release_conn()
                return data
            except Exception as e:
                logger.error(f"Download failed: {object_name} — {e}")
                return None
        else:
            path = self._local_path(object_name)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return f.read()
            return None

    def list_objects(self, prefix: str = "", limit: int = 100) -> List[Dict]:
        """List objects with optional prefix filter."""
        if self._client:
            objects = self.client.list_objects(
                self.config.bucket, prefix=prefix, recursive=True
            )
            return [
                {
                    "name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified.isoformat()
                    if obj.last_modified
                    else None,
                }
                for obj in list(objects)[:limit]
            ]
        else:
            import glob

            pattern = os.path.join(self.local_dir, prefix.replace("/", os.sep) + "*")
            files = glob.glob(pattern, recursive=True)[:limit]
            return [
                {"name": os.path.relpath(f, self.local_dir), "size": os.path.getsize(f)}
                for f in files
            ]

    def delete(self, object_name: str) -> bool:
        """Delete an object from storage."""
        if self._client:
            self.client.remove_object(self.config.bucket, object_name)
        else:
            path = self._local_path(object_name)
            if os.path.exists(path):
                os.remove(path)
        return True

    def get_presigned_url(
        self, object_name: str, expires_hours: int = 24
    ) -> Optional[str]:
        """Generate a presigned download URL."""
        if self._client:
            return self.client.presigned_get_object(
                self.config.bucket, object_name, expires=timedelta(hours=expires_hours)
            )
        return f"/api/v1/storage/download/{object_name}"

    def exists(self, object_name: str) -> bool:
        """Check if an object exists."""
        if self._client:
            try:
                self.client.stat_object(self.config.bucket, object_name)
                return True
            except Exception:
                return False
        return os.path.exists(self._local_path(object_name))


# Singleton
_storage: Optional[ImageStorage] = None


def get_storage() -> ImageStorage:
    global _storage
    if _storage is None:
        _storage = ImageStorage()
    return _storage
