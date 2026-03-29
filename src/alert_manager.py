"""Alert manager: evaluates threat records and dispatches notifications."""

from __future__ import annotations

import logging
import smtplib
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Callable

from src.config import AlertConfig
from src.models import AlertRecord, ThreatRecord

logger = logging.getLogger(__name__)

# Threat levels that trigger alerts
_ALERTABLE_LEVELS = {"Potentially_Hazardous", "Dangerous"}

# Severity mapping
_SEVERITY_MAP = {
    "Potentially_Hazardous": "medium",
    "Dangerous": "high",
}


class AlertManager:
    def __init__(self, config: AlertConfig) -> None:
        self.config = config
        # {(track_id, threat_level): last_alert_utc}
        self._dedup: dict[tuple[str, str], datetime] = {}
        # All generated alerts (for inspection/testing)
        self._alerts: list[AlertRecord] = []
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="alert_delivery")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, threat: ThreatRecord) -> None:
        """Generate and dispatch alerts for hazardous/dangerous threats."""
        if threat.threat_level not in _ALERTABLE_LEVELS:
            return

        key = (threat.track_id, threat.threat_level)
        now = datetime.now(tz=timezone.utc)

        # Deduplication check
        last = self._dedup.get(key)
        if last is not None:
            elapsed = (now - last).total_seconds()
            if elapsed < self.config.deduplication_window_seconds:
                logger.debug(
                    "Suppressing duplicate alert for %s/%s (%.1fs < %ds window)",
                    threat.track_id,
                    threat.threat_level,
                    elapsed,
                    self.config.deduplication_window_seconds,
                )
                return

        # Record dedup timestamp before dispatching
        self._dedup[key] = now

        # Determine channels
        channels: list[str] = ["visual"]
        if self.config.email.enabled:
            channels.append("email")
        if self.config.sms.enabled and threat.threat_level == "Dangerous":
            channels.append("sms")

        alert = AlertRecord(
            alert_id=str(uuid.uuid4()),
            track_id=threat.track_id,
            threat_level=threat.threat_level,
            closest_approach_au=threat.closest_approach_au,
            velocity_kms=threat.velocity_kms,
            created_at=now,
            channels=channels,
            delivery_status={"visual": "sent"},
        )
        self._alerts.append(alert)

        logger.info(
            "Alert generated: id=%s track=%s level=%s severity=%s",
            alert.alert_id,
            alert.track_id,
            alert.threat_level,
            self.severity(threat.threat_level),
        )

        # Dispatch non-visual channels in background
        if "email" in channels:
            self._executor.submit(self._deliver_with_retry, alert, "email", self._send_email)
        if "sms" in channels:
            self._executor.submit(self._deliver_with_retry, alert, "sms", self._send_sms)

    def get_alerts(self) -> list[AlertRecord]:
        """Return all generated alerts (for testing and dashboard use)."""
        return list(self._alerts)

    @staticmethod
    def severity(threat_level: str) -> str:
        """Map a threat level string to its severity label."""
        return _SEVERITY_MAP.get(threat_level, "unknown")

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the background thread pool."""
        self._executor.shutdown(wait=wait)

    # ------------------------------------------------------------------
    # Delivery helpers
    # ------------------------------------------------------------------

    def _deliver_with_retry(
        self,
        alert: AlertRecord,
        channel: str,
        deliver_fn: Callable[[AlertRecord], None],
    ) -> None:
        """Attempt delivery up to retry_attempts times with retry_interval_seconds delay."""
        attempts = self.config.retry_attempts
        for attempt in range(1, attempts + 1):
            try:
                deliver_fn(alert)
                alert.delivery_status[channel] = "sent"
                logger.info("Alert %s delivered via %s (attempt %d)", alert.alert_id, channel, attempt)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Alert %s delivery via %s failed (attempt %d/%d): %s",
                    alert.alert_id,
                    channel,
                    attempt,
                    attempts,
                    exc,
                )
                if attempt < attempts:
                    time.sleep(self.config.retry_interval_seconds)

        alert.delivery_status[channel] = "failed"
        logger.error(
            "Alert %s delivery via %s failed after %d attempts",
            alert.alert_id,
            channel,
            attempts,
        )

    def _send_email(self, alert: AlertRecord) -> None:
        """Send email notification via SMTP with STARTTLS."""
        cfg = self.config.email
        severity = self.severity(alert.threat_level)
        subject = f"[{severity.upper()}] Asteroid Threat Alert — Track {alert.track_id}"
        body = (
            f"Threat Level: {alert.threat_level}\n"
            f"Severity: {severity}\n"
            f"Track ID: {alert.track_id}\n"
            f"Closest Approach: {alert.closest_approach_au:.6f} AU\n"
            f"Velocity: {alert.velocity_kms} km/s\n"
            f"Alert ID: {alert.alert_id}\n"
            f"Generated At: {alert.created_at.isoformat()}\n"
        )

        msg = MIMEMultipart()
        msg["From"] = cfg.sender
        msg["To"] = ", ".join(cfg.recipients)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as server:
            server.ehlo()
            server.starttls()
            if cfg.username:
                server.login(cfg.username, cfg.password)
            server.sendmail(cfg.sender, cfg.recipients, msg.as_string())

    def _send_sms(self, alert: AlertRecord) -> None:
        """Send SMS notification via Twilio (Dangerous only)."""
        cfg = self.config.sms
        if alert.threat_level != "Dangerous":
            # SMS is only for Dangerous; mark suppressed
            alert.delivery_status["sms"] = "suppressed"
            return

        try:
            from twilio.rest import Client  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("twilio package is required for SMS delivery") from exc

        client = Client(cfg.account_sid, cfg.auth_token)
        body = (
            f"ASTEROID ALERT [HIGH]: Track {alert.track_id} — "
            f"Closest approach {alert.closest_approach_au:.6f} AU, "
            f"velocity {alert.velocity_kms} km/s"
        )
        for number in cfg.to_numbers:
            client.messages.create(body=body, from_=cfg.from_number, to=number)
