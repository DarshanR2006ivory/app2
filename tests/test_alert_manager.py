"""Tests for AlertManager — unit tests and property-based tests.

Feature: asteroid-threat-monitor
Property 9: Alert deduplication
Property 10: Alert severity mapping
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.alert_manager import AlertManager
from src.config import AlertConfig, EmailConfig, SmsConfig
from src.models import ThreatRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_threat(
    track_id: str = "TRK-001",
    threat_level: str = "Potentially_Hazardous",
    closest_approach_au: float = 0.03,
    velocity_kms: float | None = 5.0,
    evaluated_at: datetime | None = None,
) -> ThreatRecord:
    return ThreatRecord(
        track_id=track_id,
        threat_level=threat_level,
        previous_threat_level=None,
        closest_approach_au=closest_approach_au,
        velocity_kms=velocity_kms,
        changed_at=None,
        evaluated_at=evaluated_at or _utcnow(),
    )


def _default_config(
    dedup_window: int = 300,
    retry_attempts: int = 3,
    retry_interval: int = 0,
    email_enabled: bool = False,
    sms_enabled: bool = False,
) -> AlertConfig:
    return AlertConfig(
        deduplication_window_seconds=dedup_window,
        retry_attempts=retry_attempts,
        retry_interval_seconds=retry_interval,
        email=EmailConfig(
            enabled=email_enabled,
            smtp_host="smtp.test.com",
            smtp_port=587,
            sender="test@test.com",
            recipients=["recipient@test.com"],
            username="user",
            password="pass",
        ),
        sms=SmsConfig(
            enabled=sms_enabled,
            provider="twilio",
            account_sid="ACtest",
            auth_token="token",
            from_number="+10000000000",
            to_numbers=["+19999999999"],
        ),
    )


# ---------------------------------------------------------------------------
# Unit tests — basic alert generation
# ---------------------------------------------------------------------------

class TestAlertGeneration:
    def test_safe_threat_generates_no_alert(self):
        manager = AlertManager(_default_config())
        manager.evaluate(_make_threat(threat_level="Safe"))
        assert len(manager.get_alerts()) == 0

    def test_potentially_hazardous_generates_alert(self):
        manager = AlertManager(_default_config())
        manager.evaluate(_make_threat(threat_level="Potentially_Hazardous"))
        alerts = manager.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].threat_level == "Potentially_Hazardous"

    def test_dangerous_generates_alert(self):
        manager = AlertManager(_default_config())
        manager.evaluate(_make_threat(threat_level="Dangerous", closest_approach_au=0.001, velocity_kms=15.0))
        alerts = manager.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].threat_level == "Dangerous"

    def test_alert_has_uuid_alert_id(self):
        import re
        manager = AlertManager(_default_config())
        manager.evaluate(_make_threat(threat_level="Potentially_Hazardous"))
        alert = manager.get_alerts()[0]
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        assert uuid_pattern.match(alert.alert_id), f"alert_id is not a UUID: {alert.alert_id}"

    def test_alert_visual_channel_always_present(self):
        manager = AlertManager(_default_config())
        manager.evaluate(_make_threat(threat_level="Potentially_Hazardous"))
        alert = manager.get_alerts()[0]
        assert "visual" in alert.channels
        assert alert.delivery_status["visual"] == "sent"

    def test_alert_fields_match_threat(self):
        manager = AlertManager(_default_config())
        threat = _make_threat(
            track_id="TRK-XYZ",
            threat_level="Dangerous",
            closest_approach_au=0.001,
            velocity_kms=20.0,
        )
        manager.evaluate(threat)
        alert = manager.get_alerts()[0]
        assert alert.track_id == "TRK-XYZ"
        assert alert.threat_level == "Dangerous"
        assert alert.closest_approach_au == 0.001
        assert alert.velocity_kms == 20.0


# ---------------------------------------------------------------------------
# Unit tests — deduplication window boundary
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_duplicate_within_window_suppressed(self):
        manager = AlertManager(_default_config(dedup_window=300))
        manager.evaluate(_make_threat(threat_level="Potentially_Hazardous"))
        manager.evaluate(_make_threat(threat_level="Potentially_Hazardous"))
        assert len(manager.get_alerts()) == 1

    def test_different_track_ids_not_deduplicated(self):
        manager = AlertManager(_default_config(dedup_window=300))
        manager.evaluate(_make_threat(track_id="TRK-001", threat_level="Potentially_Hazardous"))
        manager.evaluate(_make_threat(track_id="TRK-002", threat_level="Potentially_Hazardous"))
        assert len(manager.get_alerts()) == 2

    def test_different_threat_levels_not_deduplicated(self):
        manager = AlertManager(_default_config(dedup_window=300))
        manager.evaluate(_make_threat(track_id="TRK-001", threat_level="Potentially_Hazardous"))
        manager.evaluate(_make_threat(track_id="TRK-001", threat_level="Dangerous", closest_approach_au=0.001, velocity_kms=15.0))
        assert len(manager.get_alerts()) == 2

    def test_alert_allowed_after_window_expires(self):
        """Simulate window expiry by directly manipulating the dedup dict."""
        manager = AlertManager(_default_config(dedup_window=300))
        manager.evaluate(_make_threat(threat_level="Potentially_Hazardous"))
        assert len(manager.get_alerts()) == 1

        # Manually expire the dedup entry
        key = ("TRK-001", "Potentially_Hazardous")
        manager._dedup[key] = _utcnow() - timedelta(seconds=301)

        manager.evaluate(_make_threat(threat_level="Potentially_Hazardous"))
        assert len(manager.get_alerts()) == 2

    def test_just_inside_window_suppressed(self):
        """Alert arriving 1 second before window expiry should be suppressed."""
        manager = AlertManager(_default_config(dedup_window=300))
        manager.evaluate(_make_threat(threat_level="Potentially_Hazardous"))

        key = ("TRK-001", "Potentially_Hazardous")
        # 299 seconds ago — still inside the 300s window
        manager._dedup[key] = _utcnow() - timedelta(seconds=299)

        manager.evaluate(_make_threat(threat_level="Potentially_Hazardous"))
        assert len(manager.get_alerts()) == 1

    def test_just_outside_window_allowed(self):
        """Alert arriving 1 second after window expiry should be allowed."""
        manager = AlertManager(_default_config(dedup_window=300))
        manager.evaluate(_make_threat(threat_level="Potentially_Hazardous"))

        key = ("TRK-001", "Potentially_Hazardous")
        # 301 seconds ago — just outside the 300s window
        manager._dedup[key] = _utcnow() - timedelta(seconds=301)

        manager.evaluate(_make_threat(threat_level="Potentially_Hazardous"))
        assert len(manager.get_alerts()) == 2


# ---------------------------------------------------------------------------
# Unit tests — retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    def test_email_retried_up_to_retry_attempts(self):
        """Mock email delivery failure; assert it is retried up to retry_attempts times."""
        config = _default_config(email_enabled=True, retry_attempts=3, retry_interval=0)
        manager = AlertManager(config)

        call_count = 0

        def failing_email(alert):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("SMTP unavailable")

        manager._send_email = failing_email  # type: ignore[method-assign]

        threat = _make_threat(threat_level="Potentially_Hazardous")
        manager.evaluate(threat)

        # Wait for background thread to finish
        manager.shutdown(wait=True)

        assert call_count == 3, f"Expected 3 attempts, got {call_count}"
        alert = manager.get_alerts()[0]
        assert alert.delivery_status.get("email") == "failed"

    def test_email_succeeds_on_second_attempt(self):
        """Mock email failing once then succeeding; assert status is 'sent'."""
        config = _default_config(email_enabled=True, retry_attempts=3, retry_interval=0)
        manager = AlertManager(config)

        call_count = 0

        def flaky_email(alert):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Transient failure")

        manager._send_email = flaky_email  # type: ignore[method-assign]

        manager.evaluate(_make_threat(threat_level="Potentially_Hazardous"))
        manager.shutdown(wait=True)

        assert call_count == 2
        alert = manager.get_alerts()[0]
        assert alert.delivery_status.get("email") == "sent"


# ---------------------------------------------------------------------------
# Unit tests — SMS only for Dangerous
# ---------------------------------------------------------------------------

class TestSmsChannel:
    def test_sms_not_sent_for_potentially_hazardous(self):
        config = _default_config(sms_enabled=True, retry_attempts=1, retry_interval=0)
        manager = AlertManager(config)

        sms_called = False

        def mock_sms(alert):
            nonlocal sms_called
            sms_called = True

        manager._send_sms = mock_sms  # type: ignore[method-assign]

        manager.evaluate(_make_threat(threat_level="Potentially_Hazardous"))
        manager.shutdown(wait=True)

        assert not sms_called, "SMS should not be sent for Potentially_Hazardous"
        alert = manager.get_alerts()[0]
        assert "sms" not in alert.channels

    def test_sms_sent_for_dangerous(self):
        config = _default_config(sms_enabled=True, retry_attempts=1, retry_interval=0)
        manager = AlertManager(config)

        sms_called = False

        def mock_sms(alert):
            nonlocal sms_called
            sms_called = True

        manager._send_sms = mock_sms  # type: ignore[method-assign]

        manager.evaluate(_make_threat(
            threat_level="Dangerous",
            closest_approach_au=0.001,
            velocity_kms=15.0,
        ))
        manager.shutdown(wait=True)

        assert sms_called, "SMS should be sent for Dangerous"
        alert = manager.get_alerts()[0]
        assert "sms" in alert.channels

    def test_email_channel_included_when_enabled(self):
        config = _default_config(email_enabled=True, retry_attempts=1, retry_interval=0)
        manager = AlertManager(config)
        manager._send_email = lambda a: None  # type: ignore[method-assign]

        manager.evaluate(_make_threat(threat_level="Potentially_Hazardous"))
        manager.shutdown(wait=True)

        alert = manager.get_alerts()[0]
        assert "email" in alert.channels

    def test_email_not_included_when_disabled(self):
        config = _default_config(email_enabled=False)
        manager = AlertManager(config)

        manager.evaluate(_make_threat(threat_level="Potentially_Hazardous"))
        alert = manager.get_alerts()[0]
        assert "email" not in alert.channels


# ---------------------------------------------------------------------------
# Property 9: Alert deduplication
# Validates: Requirements 7.7
# ---------------------------------------------------------------------------

@st.composite
def dedup_scenario(draw):
    """Generate a sequence of ThreatRecords for the same track/level within a window."""
    track_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_")))
    threat_level = draw(st.sampled_from(["Potentially_Hazardous", "Dangerous"]))
    count = draw(st.integers(min_value=2, max_value=10))
    closest_au = 0.001 if threat_level == "Dangerous" else 0.03
    velocity = 15.0 if threat_level == "Dangerous" else 5.0
    threats = [
        _make_threat(
            track_id=track_id,
            threat_level=threat_level,
            closest_approach_au=closest_au,
            velocity_kms=velocity,
        )
        for _ in range(count)
    ]
    return track_id, threat_level, threats


@given(scenario=dedup_scenario())
@settings(max_examples=100)
def test_property9_alert_deduplication(scenario):
    """
    Property 9: Alert deduplication
    For any sequence of ThreatRecords for the same track ID and same threat level
    arriving within the deduplication window, the alert manager must generate at
    most one alert for that (track_id, threat_level) pair within the window.
    Validates: Requirements 7.7
    """
    track_id, threat_level, threats = scenario
    # Use a large dedup window to ensure all threats fall within it
    config = _default_config(dedup_window=86400)  # 24 hours
    manager = AlertManager(config)

    for threat in threats:
        manager.evaluate(threat)

    alerts = manager.get_alerts()
    matching = [a for a in alerts if a.track_id == track_id and a.threat_level == threat_level]
    assert len(matching) <= 1, (
        f"Expected at most 1 alert for ({track_id}, {threat_level}), got {len(matching)}"
    )


# ---------------------------------------------------------------------------
# Property 10: Alert severity mapping
# Validates: Requirements 7.3
# ---------------------------------------------------------------------------

@given(
    threat_level=st.sampled_from(["Potentially_Hazardous", "Dangerous"])
)
@settings(max_examples=100)
def test_property10_alert_severity_mapping(threat_level: str):
    """
    Property 10: Alert severity mapping
    For any ThreatRecord with level Potentially_Hazardous, the generated AlertRecord
    must record severity "medium"; for any ThreatRecord with level Dangerous,
    severity must be "high".
    Validates: Requirements 7.3
    """
    closest_au = 0.001 if threat_level == "Dangerous" else 0.03
    velocity = 15.0 if threat_level == "Dangerous" else 5.0

    config = _default_config(dedup_window=0)  # no dedup so every call generates an alert
    manager = AlertManager(config)

    threat = _make_threat(
        threat_level=threat_level,
        closest_approach_au=closest_au,
        velocity_kms=velocity,
    )
    manager.evaluate(threat)

    alerts = manager.get_alerts()
    assert len(alerts) == 1

    alert = alerts[0]
    severity = AlertManager.severity(alert.threat_level)

    if threat_level == "Potentially_Hazardous":
        assert severity == "medium", f"Expected 'medium' for Potentially_Hazardous, got '{severity}'"
    elif threat_level == "Dangerous":
        assert severity == "high", f"Expected 'high' for Dangerous, got '{severity}'"
