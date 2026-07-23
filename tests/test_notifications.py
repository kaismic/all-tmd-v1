from urllib import request

import pytest

from all_tmd.notifications import (
    configured_notification_events,
    publish_notification,
)


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("all-trials", {"all-trials"}),
        ("run", {"all-trials"}),
        ("train", {"train"}),
        (
            "ingest, features",
            {"ingest-train-dataset", "ingest-collector", "features"},
        ),
        (
            "steps",
            {"ingest-train-dataset", "ingest-collector", "features", "train"},
        ),
        (
            "all",
            {
                "all-trials",
                "ingest-train-dataset",
                "ingest-collector",
                "features",
                "train",
            },
        ),
        ("none", set()),
    ],
)
def test_notification_event_aliases(configured, expected):
    assert configured_notification_events(configured) == expected


def test_unknown_notification_event_configuration_is_rejected():
    with pytest.raises(ValueError, match="Unknown NTFY_EVENTS"):
        configured_notification_events("trane")


def test_publish_only_sends_selected_event(monkeypatch):
    sent: list[request.Request] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setenv("NTFY_TOPIC", "all tmd")
    monkeypatch.setenv("NTFY_SERVER", "https://notify.example/")
    monkeypatch.setenv("NTFY_TOKEN", "secret")
    monkeypatch.setenv("NTFY_EVENTS", "train")
    monkeypatch.setattr(
        "all_tmd.notifications.request.urlopen",
        lambda notification, timeout: sent.append(notification) or Response(),
    )

    assert not publish_notification(
        "features-0", 0, 10, event="features"
    )
    assert publish_notification("train-0", 0, 65, event="train")

    assert len(sent) == 1
    notification = sent[0]
    assert notification.full_url == "https://notify.example/all%20tmd"
    assert notification.headers["Authorization"] == "Bearer secret"
    assert notification.headers["Title"] == "All-TMD train-0 completed"
    assert b"after 1m 5s" in notification.data


def test_empty_topic_disables_notifications(monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    monkeypatch.setenv("NTFY_EVENTS", "all")

    assert not publish_notification("run-trials", 0, 1, event="all-trials")
