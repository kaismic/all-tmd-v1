from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence


class TrialParametersError(ValueError):
    """Raised when a trial-parameters document is invalid."""


def generate_trials(document: Any) -> list[dict[str, Any]]:
    """Expand a trial-parameters document into its Cartesian product."""
    if not isinstance(document, dict):
        raise TrialParametersError("The document root must be a JSON object")

    unknown_root_fields = sorted(set(document) - {"default", "dimensions"})
    if unknown_root_fields:
        raise TrialParametersError(
            "Unknown root field(s): " + ", ".join(unknown_root_fields)
        )

    default = document.get("default")
    if not isinstance(default, dict):
        raise TrialParametersError("'default' must be a JSON object")

    dimensions = document.get("dimensions")
    if not isinstance(dimensions, list):
        raise TrialParametersError("'dimensions' must be a JSON array")

    parsed_dimensions: list[list[dict[str, dict[str, Any]]]] = []
    dimension_names: set[str] = set()
    paths_by_dimension: list[tuple[str, set[str]]] = []

    for dimension_index, dimension in enumerate(dimensions):
        location = f"dimensions[{dimension_index}]"
        if not isinstance(dimension, dict):
            raise TrialParametersError(f"{location} must be a JSON object")

        unknown_fields = sorted(set(dimension) - {"name", "options"})
        if unknown_fields:
            raise TrialParametersError(
                f"{location} has unknown field(s): " + ", ".join(unknown_fields)
            )

        name = dimension.get("name")
        if not isinstance(name, str) or not name.strip():
            raise TrialParametersError(f"{location}.name must be a non-empty string")
        if name in dimension_names:
            raise TrialParametersError(f"Dimension name '{name}' is duplicated")
        dimension_names.add(name)

        options = dimension.get("options")
        if not isinstance(options, list) or not options:
            raise TrialParametersError(
                f"Dimension '{name}' must contain a non-empty options array"
            )

        parsed_options: list[dict[str, dict[str, Any]]] = []
        expected_operations: set[tuple[str, str]] | None = None
        for option_index, option in enumerate(options):
            option_location = f"Dimension '{name}' option {option_index}"
            parsed_option = _validate_option(default, option, option_location)
            operations = {
                (operation, path)
                for operation, assignments in parsed_option.items()
                for path in assignments
            }
            if expected_operations is None:
                expected_operations = operations
            elif operations != expected_operations:
                raise TrialParametersError(
                    f"{option_location} must use the same operations and paths "
                    "as the other options in its dimension"
                )
            parsed_options.append(parsed_option)

        assert expected_operations is not None
        dimension_paths = {path for _, path in expected_operations}
        for previous_name, previous_paths in paths_by_dimension:
            overlap = _find_overlapping_paths(dimension_paths, previous_paths)
            if overlap is not None:
                current_path, previous_path = overlap
                raise TrialParametersError(
                    f"Dimensions '{previous_name}' and '{name}' both modify "
                    f"overlapping paths '{previous_path}' and '{current_path}'"
                )
        paths_by_dimension.append((name, dimension_paths))
        parsed_dimensions.append(parsed_options)

    combinations = product(*parsed_dimensions) if parsed_dimensions else [()]
    trials: list[dict[str, Any]] = []
    for combination in combinations:
        trial = deepcopy(default)
        for option in combination:
            _apply_option(trial, option)
        trials.append(trial)
    return trials


def write_trials(trials: list[dict[str, Any]], output_path: str | Path) -> None:
    """Atomically write generated trials as formatted JSON."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(trials, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate trials.json from Cartesian trial parameters"
    )
    parser.add_argument(
        "--parameters",
        default="trial-parameters.json",
        help="input parameters file (default: trial-parameters.json)",
    )
    parser.add_argument(
        "--output",
        default="trials.json",
        help="generated trials file (default: trials.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    parameters_path = Path(args.parameters)
    output_path = Path(args.output)
    if parameters_path.resolve() == output_path.resolve():
        parser.error("the parameters and output paths must be different")

    try:
        with parameters_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        trials = generate_trials(document)
        write_trials(trials, output_path)
    except (OSError, json.JSONDecodeError, TrialParametersError) as error:
        parser.error(str(error))

    print(f"Generated {len(trials)} trial(s) in {output_path}")
    return 0


def _validate_option(
    default: dict[str, Any],
    option: Any,
    location: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(option, dict):
        raise TrialParametersError(f"{location} must be a JSON object")

    unknown_operations = sorted(set(option) - {"set", "pick"})
    if unknown_operations:
        raise TrialParametersError(
            f"{location} has unknown operation(s): " + ", ".join(unknown_operations)
        )
    if not option:
        raise TrialParametersError(f"{location} must contain 'set' or 'pick'")

    parsed: dict[str, dict[str, Any]] = {}
    all_paths: list[str] = []
    for operation in ("set", "pick"):
        if operation not in option:
            continue
        assignments = option[operation]
        if not isinstance(assignments, dict) or not assignments:
            raise TrialParametersError(
                f"{location}.{operation} must be a non-empty JSON object"
            )

        parsed_assignments: dict[str, Any] = {}
        for path, value in assignments.items():
            _validate_path(path, location)
            current_value = _get_path(default, path, location)
            if operation == "pick":
                value = _validate_pick(value, current_value, path, location)
            parsed_assignments[path] = value
            all_paths.append(path)
        parsed[operation] = parsed_assignments

    if len(all_paths) != len(set(all_paths)):
        raise TrialParametersError(
            f"{location} cannot apply multiple operations to the same path"
        )
    option_paths = set(all_paths)
    overlap = _find_overlapping_paths(option_paths, option_paths)
    if overlap is not None:
        raise TrialParametersError(
            f"{location} modifies overlapping paths '{overlap[0]}' and "
            f"'{overlap[1]}'"
        )
    return parsed


def _validate_path(path: Any, location: str) -> None:
    if (
        not isinstance(path, str)
        or not path
        or any(not component for component in path.split("."))
    ):
        raise TrialParametersError(
            f"{location} contains invalid dotted path {path!r}"
        )


def _get_path(root: dict[str, Any], path: str, location: str) -> Any:
    current: Any = root
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            raise TrialParametersError(
                f"{location} path '{path}' does not exist in 'default'"
            )
        current = current[component]
    return current


def _validate_pick(
    keys: Any,
    current_value: Any,
    path: str,
    location: str,
) -> list[str]:
    if not isinstance(current_value, dict):
        raise TrialParametersError(
            f"{location} can only pick keys from an object; '{path}' is not one"
        )
    if (
        not isinstance(keys, list)
        or not keys
        or any(not isinstance(key, str) or not key for key in keys)
    ):
        raise TrialParametersError(
            f"{location}.pick['{path}'] must be a non-empty array of strings"
        )
    if len(keys) != len(set(keys)):
        raise TrialParametersError(
            f"{location}.pick['{path}'] contains duplicate keys"
        )
    missing = [key for key in keys if key not in current_value]
    if missing:
        raise TrialParametersError(
            f"{location}.pick['{path}'] references unknown key(s): "
            + ", ".join(missing)
        )
    return keys


def _apply_option(
    trial: dict[str, Any],
    option: dict[str, dict[str, Any]],
) -> None:
    for path, value in option.get("set", {}).items():
        _set_path(trial, path, deepcopy(value))
    for path, keys in option.get("pick", {}).items():
        current_value = _get_path(trial, path, "Generated trial")
        selected = {key: deepcopy(current_value[key]) for key in keys}
        _set_path(trial, path, selected)


def _set_path(root: dict[str, Any], path: str, value: Any) -> None:
    components = path.split(".")
    parent: dict[str, Any] = root
    for component in components[:-1]:
        parent = parent[component]
    parent[components[-1]] = value


def _find_overlapping_paths(
    left_paths: set[str],
    right_paths: set[str],
) -> tuple[str, str] | None:
    same_collection = left_paths is right_paths
    for left in sorted(left_paths):
        for right in sorted(right_paths):
            if same_collection and left >= right:
                continue
            if (
                left == right
                or left.startswith(f"{right}.")
                or right.startswith(f"{left}.")
            ):
                return left, right
    return None


if __name__ == "__main__":
    raise SystemExit(main())
