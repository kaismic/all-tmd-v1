from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight


def select_training_indices(
    frame: pd.DataFrame,
    indices: Sequence[int],
    *,
    strategy: str,
    random_seed: int,
) -> tuple[list[int], dict[str, Any]]:
    """Optionally cap collector modes at the smallest usable window duration.

    Source-domain rows are always retained. Collector rows are selected as whole
    sessions in participant round-robin order, so no evaluation data is touched
    and a long participant cannot fill the target before other participants are
    represented.
    """
    selected = [int(index) for index in indices]
    if strategy == "none":
        return selected, {"strategy": "none", "selected_rows": len(selected)}
    if strategy != "smallest_mode":
        raise ValueError(f"Unsupported duration balancing strategy: {strategy}")

    subset = frame.loc[selected]
    source_indices = subset.index[
        subset["domain"].astype(str) != "collector"
    ].astype(int).tolist()
    collector = subset.loc[subset["domain"].astype(str) == "collector"]
    if collector.empty:
        return selected, {
            "strategy": strategy,
            "target_duration_seconds": 0.0,
            "selected_rows": len(selected),
            "selected_collector_sessions_by_label": {},
            "selected_collector_duration_seconds_by_label": {},
        }

    sessions = _session_table(collector)
    totals = sessions.groupby("label")["duration_seconds"].sum()
    target = float(totals.min())
    chosen_session_ids: set[str] = set()
    selected_durations: dict[str, float] = {}
    selected_counts: dict[str, int] = {}

    for label, label_sessions in sessions.groupby("label", sort=True):
        queues: dict[str, list[tuple[str, float]]] = {}
        for participant, participant_sessions in label_sessions.groupby(
            "participant_id",
            sort=True,
        ):
            records = [
                (str(row.session_id), float(row.duration_seconds))
                for row in participant_sessions.itertuples(index=False)
            ]
            rng = np.random.default_rng(
                _stable_seed(random_seed, str(label), str(participant))
            )
            queues[str(participant)] = [records[index] for index in rng.permutation(len(records))]

        duration = 0.0
        count = 0
        while queues and duration < target:
            made_progress = False
            for participant in sorted(list(queues)):
                queue = queues[participant]
                if not queue:
                    del queues[participant]
                    continue
                session_id, session_duration = queue.pop(0)
                chosen_session_ids.add(session_id)
                duration += session_duration
                count += 1
                made_progress = True
                if not queue:
                    del queues[participant]
                if duration >= target:
                    break
            if not made_progress:
                break
        selected_durations[str(int(label))] = duration
        selected_counts[str(int(label))] = count

    collector_indices = collector.index[
        collector["session_id"].astype(str).isin(chosen_session_ids)
    ].astype(int).tolist()
    result = source_indices + collector_indices
    result.sort()
    return result, {
        "strategy": strategy,
        "target_duration_seconds": target,
        "selected_rows": len(result),
        "selected_collector_sessions_by_label": selected_counts,
        "selected_collector_duration_seconds_by_label": selected_durations,
    }


def training_sample_weights(
    frame: pd.DataFrame,
    indices: Sequence[int],
    *,
    strategy: str,
    collector_domain_weight: float,
    apply_collector_domain_weight: bool = True,
) -> np.ndarray:
    subset = frame.loc[list(indices)]
    if strategy == "class_balanced":
        weights = compute_sample_weight(
            "balanced",
            subset["label"].to_numpy(dtype=np.int64),
        ).astype(np.float64)
        if apply_collector_domain_weight:
            collector = subset["domain"].astype(str).to_numpy() == "collector"
            weights[collector] *= collector_domain_weight
        return weights
    if strategy != "hierarchical":
        raise ValueError(f"Unsupported weighting strategy: {strategy}")

    weights = pd.Series(0.0, index=subset.index, dtype=np.float64)
    for (domain, label), class_rows in subset.groupby(
        ["domain", "label"],
        sort=True,
    ):
        domain_mass = (
            collector_domain_weight
            if apply_collector_domain_weight and str(domain) == "collector"
            else 1.0
        )
        participants = class_rows["participant_id"].astype(str).unique()
        participant_mass = domain_mass / len(participants)
        for participant in participants:
            participant_rows = class_rows.loc[
                class_rows["participant_id"].astype(str) == participant
            ]
            sessions = participant_rows["session_id"].astype(str).unique()
            session_mass = participant_mass / len(sessions)
            for session_id in sessions:
                session_indices = participant_rows.index[
                    participant_rows["session_id"].astype(str) == session_id
                ]
                weights.loc[session_indices] = session_mass / len(session_indices)

    values = weights.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Training weights must be finite and positive")
    values *= len(values) / values.sum()
    return values


def _session_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session_id, session in frame.groupby("session_id", sort=True):
        labels = session["label"].dropna().astype(int).unique()
        participants = session["participant_id"].dropna().astype(str).unique()
        if len(labels) != 1 or len(participants) != 1:
            raise ValueError(
                "Each session must contain exactly one label and participant"
            )
        rows.append(
            {
                "session_id": str(session_id),
                "label": int(labels[0]),
                "participant_id": participants[0],
                "duration_seconds": _covered_duration_seconds(session),
            }
        )
    return pd.DataFrame(rows)


def _covered_duration_seconds(session: pd.DataFrame) -> float:
    intervals = sorted(
        (
            int(row.window_start_ms),
            int(row.window_end_ms),
        )
        for row in session[["window_start_ms", "window_end_ms"]].itertuples(
            index=False
        )
    )
    if not intervals:
        return 0.0
    covered = 0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            covered += end - start
            start, end = next_start, next_end
    covered += end - start
    return covered / 1000.0


def _stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:8], "big")) % (2**32)
