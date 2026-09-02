"""Export the selected e821edc model and verified mobile fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from all_tmd.deployment import export_e821edc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--collector-run-log", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fixtures-per-class", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = export_e821edc(
        model_path=args.model,
        metrics_path=args.metrics,
        split_manifest_path=args.split_manifest,
        feature_cache=args.feature_cache,
        collector_run_log=args.collector_run_log,
        output_dir=args.output_dir,
        fixtures_per_class=args.fixtures_per_class,
    )
    print(f"ONNX model: {paths.model}")
    print(f"Metadata: {paths.metadata}")
    print(f"Fixtures: {paths.fixtures}")


if __name__ == "__main__":
    main()
