#!/usr/bin/env bash

set -uo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
trials_path="$project_root/trials.json"

if [[ ! -f "$trials_path" ]]; then
    printf '%s\n' \
        "trials.json was not found. Copy trials.json.example to trials.json and edit it before running trials." \
        >&2
    exit 1
fi

count_trials() {
    if command -v python3 >/dev/null 2>&1; then
        python3 - "$trials_path" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as trials_file:
        trials = json.load(trials_file)
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"Could not read {path}: {error}")

if (
    not isinstance(trials, list)
    or not trials
    or not all(isinstance(trial, dict) for trial in trials)
):
    raise SystemExit("trials.json must contain a non-empty array of trial objects.")

print(len(trials))
PY
        return
    fi

    if command -v jq >/dev/null 2>&1; then
        jq -er '
            if type == "array" and length > 0 and all(.[]; type == "object")
            then length
            else error("trials.json must contain a non-empty array of trial objects.")
            end
        ' "$trials_path"
        return
    fi

    printf '%s\n' \
        "python3 or jq is required to read trials.json." \
        >&2
    return 1
}

if ! trial_count=$(count_trials); then
    exit 1
fi

if [[ ${ALL_TMD_TRIAL_INDEX+x} ]]; then
    previous_trial_index=$ALL_TMD_TRIAL_INDEX
    trial_index_was_set=1
else
    previous_trial_index=
    trial_index_was_set=0
fi

run_trials() {
    local index
    local status

    docker compose build
    status=$?
    if ((status != 0)); then
        printf 'docker compose build failed with exit code %d.\n' "$status" >&2
        return "$status"
    fi

    docker compose --profile mlflow up -d --wait mlflow
    status=$?
    if ((status != 0)); then
        printf 'MLflow failed to start with exit code %d.\n' "$status" >&2
        return "$status"
    fi

    for ((index = 0; index < trial_count; index++)); do
        export ALL_TMD_TRIAL_INDEX=$index
        printf 'Running All-TMD trial %d/%d (index %d)\n' \
            "$((index + 1))" "$trial_count" "$index"

        docker compose --profile ingest run --rm ingest-train-dataset
        status=$?
        if ((status != 0)); then
            printf 'Training dataset ingestion failed for trial index %d.\n' "$index" >&2
            return "$status"
        fi

        docker compose --profile ingest run --rm ingest-collector
        status=$?
        if ((status != 0)); then
            printf 'Collector ingestion failed for trial index %d.\n' "$index" >&2
            return "$status"
        fi

        docker compose --profile features run --rm features
        status=$?
        if ((status != 0)); then
            printf 'Feature extraction failed for trial index %d.\n' "$index" >&2
            return "$status"
        fi

        docker compose --profile train run --rm train
        status=$?
        if ((status != 0)); then
            printf 'Training failed for trial index %d.\n' "$index" >&2
            return "$status"
        fi
    done
}

original_directory=$PWD
if ! cd "$project_root"; then
    printf 'Could not change to project directory: %s\n' "$project_root" >&2
    exit 1
fi

start_seconds=$SECONDS
run_trials
run_exit_code=$?
duration_seconds=$((SECONDS - start_seconds))

if ! docker compose --profile notifications run --rm --no-deps notify \
    run-trials \
    --event all-trials \
    --exit-code "$run_exit_code" \
    --duration-seconds "$duration_seconds"; then
    printf '%s\n' "Warning: the run-level ntfy notification command failed." >&2
fi

if ((trial_index_was_set)); then
    export ALL_TMD_TRIAL_INDEX=$previous_trial_index
else
    unset ALL_TMD_TRIAL_INDEX
fi

cd "$original_directory" || true
exit "$run_exit_code"
