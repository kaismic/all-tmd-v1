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
    operations_by_dimension: list[tuple[str, set[tuple[str, str]]]] = []

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
        for previous_name, previous_operations in operations_by_dimension:
            conflict = _find_conflicting_operations(
                expected_operations,
                previous_operations,
            )
            if conflict is not None:
                current_path, previous_path = conflict
                raise TrialParametersError(
                    f"Dimensions '{previous_name}' and '{name}' both modify "
                    f"overlapping paths '{previous_path}' and '{current_path}'"
                )
        operations_by_dimension.append((name, expected_operations))
        parsed_dimensions.append(parsed_options)

    combinations = product(*parsed_dimensions) if parsed_dimensions else [()]
    trials: list[dict[str, Any]] = []
    trial_signatures: set[str] = set()
    for combination in combinations:
        trial = deepcopy(default)
        _apply_options(trial, combination)
        signature = json.dumps(
            trial,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        if signature in trial_signatures:
            continue
        trial_signatures.add(signature)
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
    operations = {
        (operation, path)
        for operation, assignments in parsed.items()
        for path in assignments
    }
    conflict = _find_conflicting_operations(operations, operations)
    if conflict is not None:
        raise TrialParametersError(
            f"{location} modifies overlapping paths '{conflict[0]}' and "
            f"'{conflict[1]}'"
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
    selections: Any,
    current_value: Any,
    path: str,
    location: str,
) -> list[str]:
    if not isinstance(current_value, (dict, list)):
        raise TrialParametersError(
            f"{location} can only pick keys from an object or values from a "
            f"string array; '{path}' is neither"
        )
    if (
        not isinstance(selections, list)
        or not selections
        or any(
            not isinstance(selection, str) or not selection
            for selection in selections
        )
    ):
        raise TrialParametersError(
            f"{location}.pick['{path}'] must be a non-empty array of strings"
        )
    if len(selections) != len(set(selections)):
        raise TrialParametersError(
            f"{location}.pick['{path}'] contains duplicate selections"
        )
    if isinstance(current_value, list) and any(
        not isinstance(value, str) or not value for value in current_value
    ):
        raise TrialParametersError(
            f"{location} can only pick values from a string array; "
            f"'{path}' is not one"
        )
    missing = [
        selection for selection in selections if selection not in current_value
    ]
    if missing:
        item_type = "key(s)" if isinstance(current_value, dict) else "value(s)"
        raise TrialParametersError(
            f"{location}.pick['{path}'] references unknown {item_type}: "
            + ", ".join(missing)
        )
    return selections


def _apply_options(
    trial: dict[str, Any],
    options: Sequence[dict[str, dict[str, Any]]],
) -> None:
    picks: list[tuple[str, list[str]]] = []
    for option in options:
        for path, value in option.get("set", {}).items():
            _set_path(trial, path, deepcopy(value))
        picks.extend(option.get("pick", {}).items())

    for path, selections in sorted(
        picks,
        key=lambda item: len(item[0].split(".")),
        reverse=True,
    ):
        current_value = _get_path(trial, path, "Generated trial")
        if isinstance(current_value, dict):
            selected = {
                selection: deepcopy(current_value[selection])
                for selection in selections
            }
        else:
            selected = [deepcopy(selection) for selection in selections]
        _set_path(trial, path, selected)


def _set_path(root: dict[str, Any], path: str, value: Any) -> None:
    components = path.split(".")
    parent: dict[str, Any] = root
    for component in components[:-1]:
        parent = parent[component]
    parent[components[-1]] = value


def _find_conflicting_operations(
    left_operations: set[tuple[str, str]],
    right_operations: set[tuple[str, str]],
) -> tuple[str, str] | None:
    same_collection = left_operations is right_operations
    for left_index, (left_operation, left_path) in enumerate(
        sorted(left_operations)
    ):
        for right_index, (right_operation, right_path) in enumerate(
            sorted(right_operations)
        ):
            if same_collection and left_index >= right_index:
                continue
            if (
                left_path == right_path
                or left_path.startswith(f"{right_path}.")
                or right_path.startswith(f"{left_path}.")
            ):
                nested_picks = (
                    left_path != right_path
                    and left_operation == "pick"
                    and right_operation == "pick"
                )
                if not nested_picks:
                    return left_path, right_path
    return None


if __name__ == "__main__":
    raise SystemExit(main())
