"""
Structured JSON logging for fabric defect detection system.

Produces logs in the format:
{
  "timestamp": "2026-07-12T10:30:15.700+08:00",
  "level": "CRITICAL",
  "service": "inference",
  "trace_id": "req-7f9a1b2c",
  "module": "alert_engine",
  "message": "Critical defect detected! Machine stop signal sent.",
  "data": { ... }
}
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from typing import Dict, Optional
import uuid

# ============================================================================
# JSON Formatter
# ============================================================================


class StructuredFormatter(logging.Formatter):
    """JSON log formatter matching the specified format."""

    def __init__(self, service: str = "backend"):
        super().__init__()
        self.service = service
        # Try to get local timezone
        try:
            self._tz_offset = datetime.now(timezone.utc).astimezone().strftime("%z")
        except Exception:
            self._tz_offset = "+08:00"

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self._format_timestamp(record),
            "level": record.levelname,
            "service": getattr(record, "service", self.service),
            "trace_id": getattr(record, "trace_id", self._generate_trace_id()),
            "module": record.name,
            "message": record.getMessage(),
        }

        # Attach structured data if present
        if hasattr(record, "data") and record.data:
            log_entry["data"] = record.data

        # Attach exception info
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, ensure_ascii=False, default=str)

    def _format_timestamp(self, record: logging.LogRecord) -> str:
        """Format timestamp with local timezone offset."""
        created = datetime.fromtimestamp(record.created, tz=timezone.utc)
        # Format with timezone
        ts = (
            created.strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{created.microsecond // 1000:03d}"
        )
        return f"{ts}{self._tz_offset}"

    def _generate_trace_id(self) -> str:
        return f"req-{uuid.uuid4().hex[:8]}"


# ============================================================================
# Log Manager
# ============================================================================


class LogManager:
    """
    Central log manager for structured detection logging.

    Usage:
        log = LogManager(service="inference")
        log.critical("Critical defect detected!", data={
            "alert_id": "ALT-001",
            "trigger_type": "broken_warp",
            "severity": "critical",
            "action_taken": "immediate_stop",
            "location": {"loom_id": "LOOM-007", "yarn_id": "WARP-023"},
            "operator_notified": True,
        })
    """

    def __init__(
        self,
        service: str = "inference",
        log_dir: str = "./logs",
        log_file: str = "detection.log",
        console: bool = True,
    ):
        self.service = service
        self.trace_id = f"req-{uuid.uuid4().hex[:8]}"

        # Create logger
        self.logger = logging.getLogger(f"fabric.{service}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler (JSON)
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, log_file), encoding="utf-8"
        )
        file_handler.setFormatter(StructuredFormatter(service=service))
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

        # Console handler (readable)
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)

        self._log_file = os.path.join(log_dir, log_file)

    def _log(self, level: int, message: str, data: Optional[Dict] = None, **kwargs):
        """Internal: log with structured data."""
        extra = {
            "service": self.service,
            "trace_id": self.trace_id,
        }
        if data:
            extra["data"] = data
        self.logger.log(level, message, extra=extra, **kwargs)

    def debug(self, message: str, data: Optional[Dict] = None):
        self._log(logging.DEBUG, message, data)

    def info(self, message: str, data: Optional[Dict] = None):
        self._log(logging.INFO, message, data)

    def warning(self, message: str, data: Optional[Dict] = None):
        self._log(logging.WARNING, message, data)

    def error(self, message: str, data: Optional[Dict] = None):
        self._log(logging.ERROR, message, data)

    def critical(self, message: str, data: Optional[Dict] = None):
        self._log(logging.CRITICAL, message, data)

    def new_trace(self) -> str:
        """Generate a new trace ID for a new request/detection cycle."""
        self.trace_id = f"req-{uuid.uuid4().hex[:8]}"
        return self.trace_id

    @property
    def log_file(self) -> str:
        return self._log_file

    def get_recent_logs(self, lines: int = 100) -> list:
        """Read recent log entries from file."""
        try:
            with open(self._log_file, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                return [json.loads(line) for line in all_lines[-lines:] if line.strip()]
        except (FileNotFoundError, json.JSONDecodeError):
            return []


# ============================================================================
# Detection-specific log helpers
# ============================================================================


def log_detection_result(
    log: LogManager,
    defect_type: str,
    severity: str,
    device_id: str,
    batch_id: str,
    bbox: list,
    confidence: float,
    action: str = "log_only",
    location: Optional[dict] = None,
):
    """Log a detection result in structured format."""
    alert_id = f"ALT-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:3].upper()}"

    data = {
        "alert_id": alert_id,
        "trigger_type": defect_type,
        "severity": severity,
        "action_taken": action,
        "location": location
        or {
            "device_id": device_id,
            "batch_id": batch_id,
            "image_coords": bbox,
        },
        "operator_notified": severity in ("critical", "major"),
        "confidence": confidence,
        "device_id": device_id,
        "batch_id": batch_id,
    }

    level = (
        "CRITICAL"
        if severity == "critical"
        else ("WARNING" if severity == "major" else "INFO")
    )
    getattr(log, level.lower())(
        f"{defect_type} detected on {device_id} — action: {action}",
        data=data,
    )
    return alert_id


def log_training_metrics(
    log: LogManager,
    epoch: int,
    metrics: dict,
):
    """Log training epoch metrics in structured format."""
    data = {
        "epoch": epoch,
        **metrics,
    }
    log.info(
        f"Epoch {epoch} complete — loss: {metrics.get('total_loss', 'N/A')}", data=data
    )


# ============================================================================
# Singleton loggers
# ============================================================================

_inference_log: Optional[LogManager] = None
_training_log: Optional[LogManager] = None


def get_inference_log() -> LogManager:
    global _inference_log
    if _inference_log is None:
        _inference_log = LogManager(service="inference", log_file="inference.log")
    return _inference_log


def get_training_log() -> LogManager:
    global _training_log
    if _training_log is None:
        _training_log = LogManager(service="training", log_file="training.log")
    return _training_log
