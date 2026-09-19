"""
Alert and alarm service for defect notification.

Handles critical defect alerts with configurable actions.
Integrates with MQTT for edge device notification.
"""

from dataclasses import dataclass
import logging
from typing import List, Optional

from backend.config import AlertConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Alert signal for a detected defect."""

    alert_id: str
    type: str
    severity: str
    device_id: str
    batch_id: str
    position: List[float]
    confidence: float
    recommended_action: str
    context: dict


class AlertService:
    """
    Alert service for triggering alarms on critical defects.

    Actions (Section 3.1.4, 3.1.5):
    - critical: stop_machine (immediate production halt)
    - major: mark_roll (flag roll for inspection)
    - medium/minor: log_only (record for reporting)
    """

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or get_config().alert
        self._alert_history: List[Alert] = []

    def evaluate_defect(
        self,
        defect_type: str,
        severity: str,
        device_id: str,
        batch_id: str,
        bbox: List[float],
        confidence: float,
        context: Optional[dict] = None,
    ) -> Optional[Alert]:
        """
        Evaluate a defect and return an Alert if action is needed.

        Returns None for minor defects (log_only).
        Returns Alert for critical/major defects requiring action.
        """
        action = self._get_action(severity)

        if action == "log_only":
            return None

        import uuid

        alert = Alert(
            alert_id=f"AL-{uuid.uuid4().hex[:8].upper()}",
            type=defect_type,
            severity=severity,
            device_id=device_id,
            batch_id=batch_id,
            position=bbox,
            confidence=confidence,
            recommended_action=action,
            context=context or {},
        )

        self._alert_history.append(alert)
        self._send_alert(alert)
        return alert

    def _get_action(self, severity: str) -> str:
        """Map severity to action."""
        actions = {
            "critical": self.config.critical_severity_action,
            "major": self.config.major_severity_action,
            "medium": self.config.medium_severity_action,
            "minor": self.config.minor_severity_action,
        }
        return actions.get(severity, "log_only")

    def _send_alert(self, alert: Alert):
        """
        Send alert via configured channel (MQTT / WebSocket / log).
        """
        # Structured JSON logging
        try:
            from backend.logging_config import get_inference_log, log_detection_result

            ilog = get_inference_log()
            log_detection_result(
                log=ilog,
                defect_type=alert.type,
                severity=alert.severity,
                device_id=alert.device_id,
                batch_id=alert.batch_id,
                bbox=alert.position,
                confidence=alert.confidence,
                action=alert.recommended_action,
            )
        except Exception:
            pass

        # Legacy console log
        logger.warning(
            f"[ALERT] {alert.severity.upper()} | "
            f"Type: {alert.type} | "
            f"Device: {alert.device_id} | "
            f"Batch: {alert.batch_id} | "
            f"Action: {alert.recommended_action} | "
            f"Confidence: {alert.confidence:.2f}"
        )

        # Broadcast via WebSocket for real-time dashboard updates
        try:
            import asyncio

            from backend.websocket import get_ws_manager

            manager = get_ws_manager()
            if manager.active_connections > 0:
                alert_data = {
                    "alert_id": alert.alert_id,
                    "type": alert.type,
                    "severity": alert.severity,
                    "device_id": alert.device_id,
                    "batch_id": alert.batch_id,
                    "confidence": alert.confidence,
                    "action": alert.recommended_action,
                    "position": alert.position,
                }
                # Check if running in async context
                try:
                    loop = asyncio.get_running_loop()
                    _task = loop.create_task(manager.broadcast_alert(alert_data))  # noqa: RUF006
                except RuntimeError:
                    # No running event loop — fire and forget via new thread
                    import threading

                    def _fire():
                        try:
                            loop2 = asyncio.new_event_loop()
                            loop2.run_until_complete(
                                manager.broadcast_alert(alert_data)
                            )
                            loop2.close()
                        except Exception:
                            pass

                    threading.Thread(target=_fire, daemon=True).start()
        except Exception:
            pass  # WebSocket broadcast is best-effort

    def get_alert_history(self, limit: int = 50) -> List[Alert]:
        """Get recent alert history."""
        return self._alert_history[-limit:]

    def get_alert_count(self, severity: Optional[str] = None) -> int:
        """Count alerts, optionally filtered by severity."""
        if severity:
            return sum(1 for a in self._alert_history if a.severity == severity)
        return len(self._alert_history)

    def clear_history(self):
        """Clear alert history."""
        self._alert_history.clear()


# Singleton alert service
_alert_service: Optional[AlertService] = None


def get_alert_service() -> AlertService:
    """Get or create the singleton AlertService."""
    global _alert_service
    if _alert_service is None:
        _alert_service = AlertService()
    return _alert_service
