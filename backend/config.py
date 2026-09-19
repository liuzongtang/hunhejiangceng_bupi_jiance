"""
Backend configuration for textile defect detection system.

Uses dataclass-based config with YAML file loading.
Matches the tech stack defined in Section 1.3 of the project document.
"""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, Optional

import yaml


@dataclass
class DatabaseConfig:
    """PostgreSQL connection settings."""

    host: str = "localhost"
    port: int = 5432
    name: str = "fabric_defect"
    user: str = "postgres"
    password: str = "postgres"
    echo: bool = False  # SQLAlchemy query logging

    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def sync_url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def sqlite_url(self) -> str:
        """SQLite fallback for local development."""
        return "sqlite+aiosqlite:///./fabric_defect.db"


@dataclass
class MinioConfig:
    """MinIO object storage for images."""

    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "fabric-images"
    secure: bool = False


@dataclass
class ModelConfig:
    """Model deployment settings."""

    current_version: str = "v1.0.0"
    model_type: str = "yolov8_defect"
    model_path: str = "./models/yolov8_defect_v1.0.0.onnx"
    backend: str = "dummy"  # dummy / onnx / pytorch / rtdetr
    input_size: int = 640
    confidence_threshold: float = 0.5
    nms_iou_threshold: float = 0.45
    device: str = "cuda"  # cuda / cpu / tensorrt


@dataclass
class AlertConfig:
    """Alert and alarm settings (severity-driven actions)."""

    critical_severity_action: str = "stop_machine"
    major_severity_action: str = "mark_roll"
    medium_severity_action: str = "log_only"
    minor_severity_action: str = "log_only"
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic: str = "fabric/defect/alert"


@dataclass
class ServerConfig:
    """FastAPI server settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = True
    log_level: str = "info"


@dataclass
class BackendConfig:
    """Master backend configuration."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    minio: MinioConfig = field(default_factory=MinioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    debug: bool = True
    environment: str = "development"

    @classmethod
    def from_yaml(cls, path: str) -> "BackendConfig":
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        def _nested_update(dataclass_type, section: dict):
            return dataclass_type(
                **{
                    k: v
                    for k, v in section.items()
                    if k in dataclass_type.__dataclass_fields__
                }
            )

        config = cls()
        if "database" in data:
            config.database = _nested_update(DatabaseConfig, data["database"])
        if "minio" in data:
            config.minio = _nested_update(MinioConfig, data["minio"])
        if "model" in data:
            config.model = _nested_update(ModelConfig, data["model"])
        if "alert" in data:
            config.alert = _nested_update(AlertConfig, data["alert"])
        if "server" in data:
            config.server = _nested_update(ServerConfig, data["server"])
        if "debug" in data:
            config.debug = data["debug"]
        if "environment" in data:
            config.environment = data["environment"]

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        from dataclasses import asdict

        return asdict(self)


# Singleton config instance
_config: Optional[BackendConfig] = None


def get_config(config_path: Optional[str] = None) -> BackendConfig:
    """Get or create the backend configuration instance."""
    global _config
    if _config is None:
        if config_path and os.path.exists(config_path):
            _config = BackendConfig.from_yaml(config_path)
        else:
            # Try default paths
            default_paths = [
                os.path.join(os.path.dirname(__file__), "config.yaml"),
                "config.yaml",
                "backend/config.yaml",
            ]
            for p in default_paths:
                if os.path.exists(p):
                    _config = BackendConfig.from_yaml(p)
                    break
            if _config is None:
                _config = BackendConfig()
    return _config
