"""Standalone fitting and validation selection for the Final plug-in."""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch import nn

from .plugin import Plugin, Q_COLUMNS, _quality_raw


AFFINE_C_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)
QUALITY_RIDGE_GRID = (0.0001, 0.001, 0.01, 0.1)
VIEWS = ("full", "duration")
SEEDS = (0, 1, 2)
RANK_WEIGHTS = (0.0, 0.5)
NEURAL = {
    "hidden_dim": 8,
    "dropout": 0.1,
    "rank_temperature": 2.0,
    "rank_pairs_per_batch": 512,
    "movement_penalty": 0.01,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "batch_size": 131072,
    "max_epochs": 160,
    "validation_every": 5,
    "patience_checks": 8,
    "minimum_improvement": 1e-6,
}


def _notify(callback: Callable | None, **event) -> None:
    if callback is not None:
        callback(dict(event))


def _field(split: dict, long_name: str, short_name: str):
    if long_name in split:
        return split[long_name]
    if short_name in split:
        return split[short_name]
    raise ValueError(f"Split requires {long_name!r}")


def _validate_split(split, encoder_names, label: str) -> dict:
    if not isinstance(split, dict):
        raise TypeError(f"{label} must be a mapping")
    scores = np.asarray(_field(split, "scores", "s"), dtype=np.float64)
    context = np.asarray(_field(split, "context", "q"), dtype=np.float64)
    labels = np.asarray(_field(split, "labels", "y"))
    if scores.ndim != 2 or scores.shape[1] != len(encoder_names):
        raise ValueError(f"{label} scores require shape [N,{len(encoder_names)}]")
    if context.shape != (len(scores), len(Q_COLUMNS)):
        raise ValueError(f"{label} context requires shape [N,15]")
    if labels.shape != (len(scores),) or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{label} labels must contain one binary value per trial")
    if len(np.unique(labels)) != 2:
        raise ValueError(f"{label} requires both target and nontarget trials")
    if not np.isfinite(scores).all() or not np.isfinite(context).all():
        raise ValueError(f"{label} contains nonfinite values")
    return {"s": scores, "q": context, "y": labels.astype(np.int8)}


def _check_trial_separation(calibration, validation) -> None:
    for key in ("trial_ids", "trial_keys"):
        if key in calibration and key in validation:
            left = set(map(str, calibration[key]))
            right = set(map(str, validation[key]))
            overlap = left & right
            if overlap:
                raise ValueError(
                    f"Calibration and validation share {len(overlap)} {key}"
                )
            return


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    return np.where(y == 1, 0.5 / np.sum(y == 1), 0.5 / np.sum(y == 0))


def _cllr(y: np.ndarray, scores: np.ndarray) -> float:
    positive = y == 1
    negative = ~positive
    return float(
        (
            np.logaddexp(0.0, -scores[positive]).mean()
            + np.logaddexp(0.0, scores[negative]).mean()
        )
        / (2.0 * math.log(2.0))
    )


def _eer(y: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    end = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(y) - 1]
    target = y[order] == 1
    accepted_target = np.r_[0.0, np.cumsum(target)[end]]
    accepted_negative = np.r_[0.0, np.cumsum(~target)[end]]
    miss = 1.0 - accepted_target / np.sum(target)
    false_alarm = accepted_negative / np.sum(~target)
    difference = false_alarm - miss
    crossing = np.flatnonzero(difference >= 0.0)
    if not len(crossing):
        index = int(np.argmin(np.abs(difference)))
        return float(50.0 * (false_alarm[index] + miss[index]))
    index = int(crossing[0])
    if index == 0:
        return float(50.0 * (false_alarm[index] + miss[index]))
    fraction = -difference[index - 1] / (difference[index] - difference[index - 1])
    value = false_alarm[index - 1] + fraction * (
        false_alarm[index] - false_alarm[index - 1]
    )
    return float(100.0 * value)


def _metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    return {"Cllr": _cllr(y, scores), "eer_pct": _eer(y, scores)}


def _logistic_objective(theta, x, y, regularization):
    weights = _balanced_weights(y)
    logits = x @ theta[:-1] + theta[-1]
    value = float(weights @ (np.logaddexp(0.0, logits) - y * logits))
    error = weights * (expit(logits) - y)
    gradient = np.r_[x.T @ error, error.sum()]
    if regularization is not None:
        value += float(theta[:-1] @ theta[:-1]) / (2.0 * regularization)
        gradient[:-1] += theta[:-1] / regularization
    return value, gradient


def _optimize_logistic(x, y, regularization):
    result = minimize(
        _logistic_objective,
        np.zeros(x.shape[1] + 1),
        args=(x, y, regularization),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Affine fitting failed: {result.message}")
    return result.x


def _fit_positive_affine(scores, labels):
    weights = _balanced_weights(labels)

    def objective(theta):
        scale = float(np.exp(np.clip(theta[0], -8.0, 8.0)))
        logits = scale * scores + theta[1]
        value = float(weights @ (np.logaddexp(0.0, logits) - labels * logits))
        posterior = expit(np.clip(logits, -50.0, 50.0))
        error = weights * (posterior - labels)
        gradient = np.asarray([(error * scores).sum() * scale, error.sum()])
        return value, gradient

    result = minimize(
        objective,
        np.zeros(2),
        jac=True,
        method="L-BFGS-B",
        bounds=((-8.0, 8.0), (None, None)),
        options={"maxiter": 2000},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Single-score affine fitting failed: {result.message}")
    return np.asarray([np.exp(np.clip(result.x[0], -8.0, 8.0)), result.x[1]])


def _fit_source(calibration, validation, encoder_names):
    mean = calibration["s"].mean(axis=0)
    scale = np.maximum(calibration["s"].std(axis=0), 1e-6)
    x_cal = (calibration["s"] - mean) / scale
    x_val = (validation["s"] - mean) / scale
    if len(encoder_names) == 1:
        theta = _fit_positive_affine(x_cal[:, 0], calibration["y"])
        selected_c = None
    else:
        candidates = []
        for value in AFFINE_C_GRID:
            theta = _optimize_logistic(x_cal, calibration["y"], value)
            candidates.append((value, theta, _cllr(validation["y"], x_val @ theta[:-1] + theta[-1])))
        best = min(item[2] for item in candidates)
        selected_c, theta, _ = next(item for item in candidates if item[2] <= best + 1e-8)
    model = {
        "score_columns": list(encoder_names),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": theta[:-1].tolist(),
        "bias": float(theta[-1]),
        "selected_C": selected_c,
    }
    return model


def _source_predict(model, scores):
    normalized = (scores - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return normalized @ np.asarray(model["weights"]) + model["bias"]


def _context_raw(scores, context, base):
    return np.column_stack(
        [
            base,
            context,
            scores.std(axis=1),
            np.ptp(scores, axis=1),
            base[:, None] * context,
        ]
    )


def _fit_context(calibration, validation, base_cal, base_val):
    raw_cal = _context_raw(calibration["s"], calibration["q"], base_cal)
    raw_val = _context_raw(validation["s"], validation["q"], base_val)
    mean = raw_cal.mean(axis=0)
    scale = np.maximum(raw_cal.std(axis=0), 1e-6)
    x_cal = (raw_cal - mean) / scale
    x_val = (raw_val - mean) / scale
    candidates = []
    for value in AFFINE_C_GRID:
        theta = _optimize_logistic(x_cal, calibration["y"], value)
        scores = x_val @ theta[:-1] + theta[-1]
        candidates.append((value, theta, _cllr(validation["y"], scores)))
    best = min(item[2] for item in candidates)
    selected_c, theta, _ = next(item for item in candidates if item[2] <= best + 1e-8)
    model = {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": theta[:-1].tolist(),
        "bias": float(theta[-1]),
        "selected_C": selected_c,
    }
    return model


def _context_predict(model, scores, context, base):
    raw = _context_raw(scores, context, base)
    normalized = (raw - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return normalized @ np.asarray(model["weights"]) + model["bias"]


def _normalization(raw, stats=None):
    if stats is None:
        standard_deviation = raw.std(axis=0)
        stats = {
            "mean": raw.mean(axis=0).tolist(),
            "scale": np.maximum(standard_deviation, 1e-6).tolist(),
            "active": (standard_deviation > 1e-6).tolist(),
            "clip": 6.0,
        }
    values = (raw - np.asarray(stats["mean"])) / np.asarray(stats["scale"])
    values[:, ~np.asarray(stats["active"], dtype=bool)] = 0.0
    return np.clip(values, -stats["clip"], stats["clip"]), stats


def _linear_objective(theta, values, base, labels, weights, ridge):
    logits = base + values @ theta[:-1] + theta[-1]
    value = float(
        weights @ (np.logaddexp(0.0, logits) - labels * logits)
        + ridge * (theta @ theta) / 2.0
    )
    error = weights * (expit(logits) - labels)
    gradient = np.r_[values.T @ error, error.sum()] + ridge * theta
    return value, gradient


class _Correction(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dimension, NEURAL["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(NEURAL["dropout"]),
            nn.Linear(NEURAL["hidden_dim"], 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, values):
        return self.head(values).squeeze(-1)


def _neural_predict(model, values, base):
    model.eval()
    output = np.empty(len(base), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(output), NEURAL["batch_size"]):
            end = min(start + NEURAL["batch_size"], len(output))
            output[start:end] = (
                base[start:end] + model(values[start:end]).cpu().numpy()
            )
    return output


def _copy_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _fit_neural(x_cal, x_val, base_cal, base_val, y_cal, y_val, seed, rank_weight, device, progress):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = _Correction(x_cal.shape[1]).to(device)
    cal_tensor = torch.as_tensor(x_cal.astype("float32"), device=device)
    val_tensor = torch.as_tensor(x_val.astype("float32"), device=device)
    labels = torch.as_tensor(y_cal, dtype=torch.float32, device=device)
    weights = torch.as_tensor(_balanced_weights(y_cal), dtype=torch.float32, device=device)
    base_tensor = torch.as_tensor(base_cal, dtype=torch.float32, device=device)
    initial = _metrics(y_val, base_val)
    snapshots = {
        name: {"state": _copy_state(model), "epoch": 0, "metric": initial}
        for name in ("cllr", "joint")
    }
    target = np.flatnonzero(y_cal == 1)
    negative = np.flatnonzero(y_cal == 0)
    hard_target = target[base_cal[target] <= np.quantile(base_cal[target], 0.25)]
    hard_negative = negative[base_cal[negative] >= np.quantile(base_cal[negative], 0.75)]
    pools = [
        torch.as_tensor(rows, dtype=torch.int64, device=device)
        for rows in (target, negative, hard_target, hard_negative)
    ]
    order_rng = torch.Generator(device="cpu").manual_seed(seed)
    pair_rng = torch.Generator(device="cpu").manual_seed(seed + 100003)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=NEURAL["learning_rate"],
        weight_decay=NEURAL["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=3,
        threshold=1e-6,
        min_lr=1e-5,
    )
    bad = 0
    for epoch in range(1, NEURAL["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(y_cal), generator=order_rng).to(device)
        for start in range(0, len(order), NEURAL["batch_size"]):
            rows = order[start : start + NEURAL["batch_size"]]
            delta = model(cal_tensor[rows])
            logits = base_tensor[rows] + delta
            terms = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels[rows],
                reduction="none",
            )
            loss = (
                weights[rows]
                * (terms + NEURAL["movement_penalty"] * delta.square())
            ).sum() / weights[rows].sum()
            if rank_weight:
                count = NEURAL["rank_pairs_per_batch"] // 2
                sampled = [
                    pool[
                        torch.randint(len(pool), (count,), generator=pair_rng).to(device)
                    ]
                    for pool in pools
                ]
                positive_rows = torch.cat((sampled[0], sampled[2]))
                negative_rows = torch.cat((sampled[1], sampled[3]))
                difference = (
                    base_tensor[positive_rows]
                    + model(cal_tensor[positive_rows])
                    - base_tensor[negative_rows]
                    - model(cal_tensor[negative_rows])
                )
                loss = loss + rank_weight * torch.nn.functional.softplus(
                    -difference / NEURAL["rank_temperature"]
                ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch != 1 and epoch % NEURAL["validation_every"]:
            continue
        metric = _metrics(y_val, _neural_predict(model, val_tensor, base_val))
        scheduler.step(metric["Cllr"])
        if metric["Cllr"] < snapshots["cllr"]["metric"]["Cllr"] - NEURAL["minimum_improvement"]:
            snapshots["cllr"] = {"state": _copy_state(model), "epoch": epoch, "metric": metric}
            bad = 0
        else:
            bad += 1
        if (
            metric["eer_pct"] <= initial["eer_pct"] + 1e-9
            and metric["Cllr"]
            < snapshots["joint"]["metric"]["Cllr"] - NEURAL["minimum_improvement"]
        ):
            snapshots["joint"] = {"state": _copy_state(model), "epoch": epoch, "metric": metric}
        _notify(
            progress,
            stage="neural",
            seed=seed,
            rank_weight=rank_weight,
            epoch=epoch,
            validation_Cllr=metric["Cllr"],
            validation_EER_pct=metric["eer_pct"],
        )
        if bad >= NEURAL["patience_checks"]:
            break
    return snapshots


def _state_payload(state):
    return {name: value.tolist() for name, value in state.items()}


def fit_plugin(
    calibration,
    validation,
    encoder_names,
    *,
    output_path: str | Path | None,
    device: str,
    progress: Callable | None,
) -> Plugin:
    """Fit affine anchors, select one residual family on validation, and freeze it."""

    names = tuple(map(str, encoder_names))
    if not names or len(set(names)) != len(names):
        raise ValueError("encoder_names must be unique and nonempty")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    _check_trial_separation(calibration, validation)
    cal = _validate_split(calibration, names, "calibration")
    val = _validate_split(validation, names, "validation")

    _notify(progress, stage="source_affine")
    source = _fit_source(cal, val, names)
    source_cal = _source_predict(source, cal["s"])
    source_val = _source_predict(source, val["s"])
    _notify(progress, stage="context_affine")
    context_model = _fit_context(cal, val, source_cal, source_val)
    base_cal = _context_predict(context_model, cal["s"], cal["q"], source_cal)
    base_val = _context_predict(context_model, val["s"], val["q"], source_val)

    records = [
        {
            "family": "context_affine",
            "seed": None,
            "complexity": 0,
            "metric": _metrics(val["y"], base_val),
        }
    ]
    corrections = {("context_affine", None): {"kind": "context_affine"}}
    weights = _balanced_weights(cal["y"])
    for view in VIEWS:
        raw_cal = _quality_raw(base_cal, cal["q"], view)
        raw_val = _quality_raw(base_val, val["q"], view)
        quality_cal, normalization = _normalization(raw_cal)
        quality_val, _ = _normalization(raw_val, normalization)
        for ridge in QUALITY_RIDGE_GRID:
            family = f"quality_{view}_ridge{ridge:g}"
            _notify(progress, stage="quality", family=family)
            result = minimize(
                _linear_objective,
                np.zeros(quality_cal.shape[1] + 1),
                args=(quality_cal, base_cal, cal["y"].astype(float), weights, ridge),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8},
            )
            if not result.success or not np.isfinite(result.x).all():
                raise RuntimeError(f"Residual fitting failed for {family}: {result.message}")
            prediction = base_val + quality_val @ result.x[:-1] + result.x[-1]
            correction = {
                "kind": "quality",
                "view": view,
                "ridge": ridge,
                "theta": result.x.tolist(),
                "normalization": normalization,
            }
            corrections[(family, None)] = correction
            records.append(
                {
                    "family": family,
                    "seed": None,
                    "complexity": 1,
                    "metric": _metrics(val["y"], prediction),
                }
            )

        neural_raw_cal = np.column_stack(
            [raw_cal, cal["q"] if view == "full" else cal["q"][:, :12], cal["s"]]
        )
        neural_raw_val = np.column_stack(
            [raw_val, val["q"] if view == "full" else val["q"][:, :12], val["s"]]
        )
        neural_cal, neural_normalization = _normalization(neural_raw_cal)
        neural_val, _ = _normalization(neural_raw_val, neural_normalization)
        neural_val_tensor = torch.as_tensor(
            neural_val.astype("float32"),
            device=device,
        )
        for rank_weight in RANK_WEIGHTS:
            snapshots_by_seed = {}
            for seed in SEEDS:
                snapshots_by_seed[seed] = _fit_neural(
                    neural_cal,
                    neural_val,
                    base_cal,
                    base_val,
                    cal["y"],
                    val["y"],
                    seed,
                    rank_weight,
                    device,
                    progress,
                )
            for checkpoint in ("cllr", "joint"):
                family = f"warm_{view}_rank{rank_weight:g}_{checkpoint}"
                for seed in SEEDS:
                    snapshot = snapshots_by_seed[seed][checkpoint]
                    model = _Correction(neural_cal.shape[1]).to(device)
                    model.load_state_dict(snapshot["state"])
                    prediction = _neural_predict(model, neural_val_tensor, base_val)
                    correction = {
                        "kind": "neural",
                        "view": view,
                        "rank_weight": rank_weight,
                        "epoch": snapshot["epoch"],
                        "dimension": neural_cal.shape[1],
                        "normalization": neural_normalization,
                        "state": _state_payload(snapshot["state"]),
                    }
                    corrections[(family, seed)] = correction
                    records.append(
                        {
                            "family": family,
                            "seed": seed,
                            "complexity": 2,
                            "metric": _metrics(val["y"], prediction),
                        }
                    )
                del model

    families = []
    for family in sorted({record["family"] for record in records}):
        current = [record for record in records if record["family"] == family]
        families.append(
            {
                "family": family,
                "complexity": current[0]["complexity"],
                "Cllr": float(np.mean([record["metric"]["Cllr"] for record in current])),
                "eer_pct": float(np.mean([record["metric"]["eer_pct"] for record in current])),
            }
        )
    baseline = next(item for item in families if item["family"] == "context_affine")
    eligible = [
        item
        for item in families
        if item["Cllr"] <= baseline["Cllr"] + 1e-8
        and item["eer_pct"] <= baseline["eer_pct"] + 1e-8
    ]
    best_cllr = min(item["Cllr"] for item in eligible)
    selected = sorted(
        (item for item in eligible if item["Cllr"] <= best_cllr + 1e-8),
        key=lambda item: (item["complexity"], item["family"]),
    )[0]
    selected_seed = 0 if selected["complexity"] == 2 else None
    correction = copy.deepcopy(corrections[(selected["family"], selected_seed)])
    payload = {
        "format": Plugin.FORMAT,
        "correction_mode": "unbounded",
        "bound": None,
        "encoder_names": list(names),
        "context_names": list(Q_COLUMNS),
        "source_affine": source,
        "context_affine": context_model,
        "correction": correction,
        "selection": {
            "policy": "joint_validation",
            "selected_family": selected["family"],
            "deployment_seed": selected_seed,
            "candidate_pool": "full",
            "family_count": len(families),
            "validation_Cllr": selected["Cllr"],
            "validation_EER_pct": selected["eer_pct"],
            "context_validation_Cllr": baseline["Cllr"],
            "context_validation_EER_pct": baseline["eer_pct"],
            "calibration_rows": len(cal["y"]),
            "validation_rows": len(val["y"]),
            "test_labels_accepted": False,
        },
        "missing_context_default": "error",
        "missing_context_explicit_fallback": "source_affine",
    }
    fitted = Plugin(payload)
    if output_path is not None:
        fitted.save(output_path)
    _notify(progress, stage="completed", selected_family=selected["family"])
    return fitted


__all__ = ["fit_plugin"]
