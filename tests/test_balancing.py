from __future__ import annotations

import numpy as np
import pandas as pd

from all_tmd.balancing import select_training_indices, training_sample_weights


def test_hierarchical_weights_equalize_classes_participants_and_sessions():
    frame = pd.DataFrame(
        [
            _row("source", 0, "source-a", "source-a1", 0),
            _row("source", 0, "source-a", "source-a1", 1),
            _row("source", 1, "source-b", "source-b1", 0),
            _row("collector", 0, "p1", "bus-long", 0),
            _row("collector", 0, "p1", "bus-long", 1),
            _row("collector", 0, "p1", "bus-short", 0),
            _row("collector", 0, "p2", "bus-other", 0),
            _row("collector", 1, "p3", "car-one", 0),
        ]
    )

    weights = training_sample_weights(
        frame,
        frame.index,
        strategy="hierarchical",
        collector_domain_weight=2.0,
    )
    weighted = frame.assign(weight=weights)

    domain_class_totals = weighted.groupby(["domain", "label"])["weight"].sum()
    assert np.isclose(
        domain_class_totals["collector", 0],
        domain_class_totals["collector", 1],
    )
    assert np.isclose(
        domain_class_totals["collector", 0],
        2 * domain_class_totals["source", 0],
    )
    bus = weighted[(weighted["domain"] == "collector") & (weighted["label"] == 0)]
    participant_totals = bus.groupby("participant_id")["weight"].sum()
    assert np.isclose(participant_totals["p1"], participant_totals["p2"])
    p1_sessions = bus[bus["participant_id"] == "p1"].groupby("session_id")["weight"].sum()
    assert np.isclose(p1_sessions["bus-long"], p1_sessions["bus-short"])


def test_duration_balancing_keeps_source_and_whole_diverse_collector_sessions():
    rows = [
        _row("source", label, f"source-{label}", f"source-{label}", 0)
        for label in range(2)
    ]
    for session, participant, label, seconds in (
        ("bus-a", "p1", 0, 100),
        ("bus-b", "p2", 0, 100),
        ("car-a", "p3", 1, 100),
        ("car-b", "p4", 1, 100),
        ("car-c", "p3", 1, 100),
        ("car-d", "p4", 1, 100),
    ):
        rows.extend(
            [
                _row("collector", label, participant, session, 0, seconds),
                _row("collector", label, participant, session, 1, seconds),
            ]
        )
    frame = pd.DataFrame(rows)

    selected, report = select_training_indices(
        frame,
        frame.index,
        strategy="smallest_mode",
        random_seed=42,
    )
    selected_frame = frame.loc[selected]

    assert set(frame.index[frame["domain"] == "source"]).issubset(selected)
    selected_sessions = selected_frame[selected_frame["domain"] == "collector"]
    assert selected_sessions.groupby("session_id").size().eq(2).all()
    assert selected_sessions.groupby("label")["session_id"].nunique().to_dict() == {
        0: 2,
        1: 2,
    }
    assert selected_sessions[selected_sessions["label"] == 1][
        "participant_id"
    ].nunique() == 2
    assert report["target_duration_seconds"] == 200.0


def _row(
    domain: str,
    label: int,
    participant: str,
    session: str,
    window: int,
    session_seconds: int = 10,
) -> dict:
    step = session_seconds * 1000 // 2
    return {
        "domain": domain,
        "label": label,
        "participant_id": participant,
        "session_id": session,
        "group_id": f"{participant}#{session}",
        "window_start_ms": window * step,
        "window_end_ms": window * step + step,
    }
