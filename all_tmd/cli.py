from __future__ import annotations

import argparse
import json

from all_tmd.config import PipelineConfig
from all_tmd.ingest import ingest_collector, ingest_training_dataset
from all_tmd.progress import progress
from all_tmd.train import train
from all_tmd.windowing import build_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="All-TMD training pipeline")
    parser.add_argument("--config", default="model.config.yaml")
    parser.add_argument("--trials", default="trials.json")
    parser.add_argument("--trial-index", type=int, default=0)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest-train-dataset")
    subparsers.add_parser("ingest-collector")
    subparsers.add_parser("features")
    subparsers.add_parser("train")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig.from_files(
        args.config,
        args.trials,
        args.trial_index,
    )
    run_name = (
        f", run_name={config.trial.run_name}"
        if config.trial.run_name is not None
        else ""
    )
    progress(
        f"Command starting: command={args.command}, trial={args.trial_index}"
        f"{run_name}, config_hash={config.config_hash}"
    )
    if args.command == "ingest-train-dataset":
        result = {"events_path": str(ingest_training_dataset(config))}
    elif args.command == "ingest-collector":
        result = {"events_path": str(ingest_collector(config))}
    elif args.command == "features":
        result = {
            name: str(path)
            for name, path in build_features(config).items()
        }
    elif args.command == "train":
        result = train(config)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
