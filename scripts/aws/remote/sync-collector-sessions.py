#!/usr/bin/env python3
"""Incrementally download confirmed collector sessions using the AWS CLI."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any


CHECKPOINT_VERSION = 1
INDEX_NAME = "received-sync-index"
PARTICIPANT_PATTERN = re.compile(r"^participant_\d{3}$")
SYNC_PARTITION = "received"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download newly confirmed collector sessions from S3."
    )
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def aws_json(arguments: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        ["aws", *arguments, "--output", "json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def query_sessions(table: str, after_sync_key: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    exclusive_start_key: dict[str, Any] | None = None
    while True:
        key_condition = "sync_partition = :received"
        values: dict[str, Any] = {":received": {"S": SYNC_PARTITION}}
        if after_sync_key:
            key_condition += " AND sync_key > :after"
            values[":after"] = {"S": after_sync_key}
        arguments = [
            "dynamodb",
            "query",
            "--no-paginate",
            "--table-name",
            table,
            "--index-name",
            INDEX_NAME,
            "--key-condition-expression",
            key_condition,
            "--expression-attribute-values",
            json.dumps(values, separators=(",", ":")),
        ]
        if exclusive_start_key:
            arguments.extend(
                [
                    "--exclusive-start-key",
                    json.dumps(exclusive_start_key, separators=(",", ":")),
                ]
            )
        page = aws_json(arguments)
        items.extend(deserialize_item(item) for item in page.get("Items", []))
        exclusive_start_key = page.get("LastEvaluatedKey")
        if not exclusive_start_key:
            return items


def deserialize_item(item: dict[str, Any]) -> dict[str, Any]:
    return {key: deserialize_value(value) for key, value in item.items()}


def deserialize_value(value: dict[str, Any]) -> Any:
    if "S" in value:
        return value["S"]
    if "N" in value:
        number = value["N"]
        if any(character in number for character in ".eE"):
            return float(number)
        return int(number)
    if "BOOL" in value:
        return value["BOOL"]
    if "NULL" in value:
        return None
    if "L" in value:
        return [deserialize_value(entry) for entry in value["L"]]
    if "M" in value:
        return deserialize_item(value["M"])
    if "SS" in value:
        return value["SS"]
    if "NS" in value:
        return [deserialize_value({"N": entry}) for entry in value["NS"]]
    raise ValueError(f"Unsupported DynamoDB value: {value}")


def is_eligible(item: dict[str, Any]) -> bool:
    participant_id = item.get("participant_id")
    s3_key = item.get("s3_key")
    if not isinstance(participant_id, str) or not PARTICIPANT_PATTERN.fullmatch(
        participant_id
    ):
        return False
    if not isinstance(s3_key, str):
        return False
    parts = PurePosixPath(s3_key).parts
    return len(parts) >= 3 and parts[0] == "raw" and parts[1] == participant_id


def destination_for(output_dir: Path, s3_key: str) -> Path:
    key = PurePosixPath(s3_key)
    if key.is_absolute() or any(part in {"", ".", ".."} for part in key.parts):
        raise ValueError(f"Unsafe collector S3 key: {s3_key}")
    return output_dir.joinpath(*key.parts)


def read_checkpoint(path: Path, bucket: str, table: str) -> str:
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("version") != CHECKPOINT_VERSION
        or data.get("source_bucket") != bucket
        or data.get("source_table") != table
        or data.get("source_index") != INDEX_NAME
    ):
        return ""
    value = data.get("last_sync_key", "")
    if not isinstance(value, str):
        raise ValueError(f"Invalid collector download checkpoint: {path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def download_session(bucket: str, output_dir: Path, item: dict[str, Any]) -> bool:
    s3_key = str(item["s3_key"])
    destination = destination_for(output_dir, s3_key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = False
    if not destination.exists():
        temporary = destination.with_name(f"{destination.name}.tmp")
        try:
            subprocess.run(
                [
                    "aws",
                    "s3",
                    "cp",
                    f"s3://{bucket}/{s3_key}",
                    str(temporary),
                    "--only-show-errors",
                ],
                check=True,
            )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        downloaded = True
    metadata_path = destination.with_suffix(f"{destination.suffix}.metadata.json")
    write_json_atomic(metadata_path, item)
    return downloaded


def sync(bucket: str, table: str, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / ".download_checkpoint.json"
    last_sync_key = read_checkpoint(checkpoint_path, bucket, table)
    discovered = query_sessions(table, last_sync_key)
    eligible = [item for item in discovered if is_eligible(item)]
    downloaded_count = sum(
        download_session(bucket, output_dir, item) for item in eligible
    )
    if discovered:
        write_json_atomic(
            checkpoint_path,
            {
                "version": CHECKPOINT_VERSION,
                "last_sync_key": discovered[-1]["sync_key"],
                "source_bucket": bucket,
                "source_table": table,
                "source_index": INDEX_NAME,
            },
        )
    return {
        "discovered_count": len(discovered),
        "eligible_count": len(eligible),
        "downloaded_count": downloaded_count,
    }


def main() -> None:
    args = parse_args()
    result = sync(args.bucket, args.table, args.output_dir)
    print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
