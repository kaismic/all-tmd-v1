from __future__ import annotations

from typing import Any

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def suggest_model_params(
    trial: optuna.Trial,
    model_families: tuple[str, ...],
) -> dict[str, Any]:
    family = trial.suggest_categorical("family", list(model_families))
    prefix = f"{family}__"
    if family == "random_forest":
        params = {
            "family": family,
            "n_estimators": trial.suggest_int(
                f"{prefix}n_estimators", 200, 800, step=100
            ),
            "max_depth": trial.suggest_categorical(
                f"{prefix}max_depth", [None, 8, 12, 16, 24]
            ),
            "min_samples_split": trial.suggest_categorical(
                f"{prefix}min_samples_split", [2, 4, 8, 16]
            ),
            "min_samples_leaf": trial.suggest_categorical(
                f"{prefix}min_samples_leaf", [1, 2, 4, 8]
            ),
            "max_features": trial.suggest_categorical(
                f"{prefix}max_features", ["sqrt", "log2", 0.5, 0.8]
            ),
            "class_weight": trial.suggest_categorical(
                f"{prefix}class_weight", [None, "balanced", "balanced_subsample"]
            ),
        }
    elif family == "xgboost":
        params = {
            "family": family,
            "n_estimators": trial.suggest_int(
                f"{prefix}n_estimators", 200, 1200, step=100
            ),
            "learning_rate": trial.suggest_float(
                f"{prefix}learning_rate", 0.02, 0.2, log=True
            ),
            "max_depth": trial.suggest_int(f"{prefix}max_depth", 3, 8),
            "min_child_weight": trial.suggest_float(
                f"{prefix}min_child_weight", 1, 10
            ),
            "subsample": trial.suggest_float(f"{prefix}subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                f"{prefix}colsample_bytree", 0.5, 1.0
            ),
            "gamma": trial.suggest_float(f"{prefix}gamma", 0, 5),
            "reg_alpha": trial.suggest_float(f"{prefix}reg_alpha", 0, 2),
            "reg_lambda": trial.suggest_float(f"{prefix}reg_lambda", 0.5, 10),
        }
    elif family == "mlp":
        params = {
            "family": family,
            "hidden_layers": trial.suggest_categorical(
                f"{prefix}hidden_layers", ["64", "128", "64_32", "128_64"]
            ),
            "activation": trial.suggest_categorical(
                f"{prefix}activation", ["relu", "tanh"]
            ),
            "alpha": trial.suggest_float(f"{prefix}alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float(
                f"{prefix}learning_rate_init", 1e-4, 5e-3, log=True
            ),
            "batch_size": trial.suggest_categorical(
                f"{prefix}batch_size", [32, 64, 128, 256]
            ),
            "max_iter": trial.suggest_int(f"{prefix}max_iter", 100, 300, step=50),
            "n_iter_no_change": trial.suggest_int(
                f"{prefix}n_iter_no_change", 10, 20
            ),
            "tol": trial.suggest_float(f"{prefix}tol", 1e-4, 1e-3, log=True),
        }
    else:
        raise ValueError(f"Unknown model family: {family}")
    trial.set_user_attr("model_params", params)
    return params


def model_from_params(
    params: dict[str, Any],
    random_seed: int,
    n_jobs: int,
    xgboost_device: str,
):
    family = params["family"]
    if family == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth"),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight", "balanced_subsample"),
            random_state=random_seed,
            n_jobs=n_jobs,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("model", estimator),
            ]
        )
    if family == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "xgboost is required for the xgboost model family"
            ) from exc
        estimator = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            device=xgboost_device,
            n_estimators=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 5)),
            min_child_weight=float(params.get("min_child_weight", 1)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            gamma=float(params.get("gamma", 0)),
            reg_alpha=float(params.get("reg_alpha", 0)),
            reg_lambda=float(params.get("reg_lambda", 1)),
            random_state=random_seed,
            n_jobs=n_jobs,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("model", estimator),
            ]
        )
    if family == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=_hidden_layers(params.get("hidden_layers", "64")),
            activation=str(params.get("activation", "relu")),
            alpha=float(params.get("alpha", 0.0001)),
            learning_rate_init=float(params.get("learning_rate_init", 0.001)),
            batch_size=int(params.get("batch_size", 64)),
            max_iter=int(params.get("max_iter", 150)),
            early_stopping=True,
            n_iter_no_change=int(params.get("n_iter_no_change", 12)),
            tol=float(params.get("tol", 0.0001)),
            random_state=random_seed,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    raise ValueError(f"Unknown model family: {family}")


def fit_with_optional_sample_weight(model, x, y, sample_weight=None):
    if sample_weight is None:
        return model.fit(x, y)
    try:
        return model.fit(x, y, model__sample_weight=sample_weight)
    except TypeError:
        return model.fit(x, y)


def _hidden_layers(value: Any) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    return tuple(int(part) for part in str(value).replace(",", "_").split("_") if part)
