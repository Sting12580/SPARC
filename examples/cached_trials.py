"""Fit and predict from NPZ caches without loading audio or pretrained models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from delivery import Plugin, Q_COLUMNS


def _strings(values: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    if values.shape != shape or values.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a string array with shape {shape}")
    strings = values.astype(str)
    if any(not value.strip() for value in strings):
        raise ValueError(f"{name} must not contain empty strings")
    return strings


def _numeric(values: np.ndarray, name: str) -> np.ndarray:
    if values.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError(f"{name} must be numeric")
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite values")
    return values


def load_trials(path: str | Path, *, require_labels: bool = False) -> dict:
    """Read the exact ordered schema; prediction never reads a labels array.

    Required arrays are scores [N,K], context [N,15], encoder_names [K],
    context_names [15], and unique trial_ids [N]. Fitting also needs binary
    labels [N], with both target (1) and nontarget (0) trials in each split.
    Names and trial IDs must use NumPy Unicode or byte-string dtypes, not object.
    """
    required = {"scores", "context", "encoder_names", "context_names", "trial_ids"}
    if require_labels:
        required.add("labels")
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing arrays: {', '.join(sorted(missing))}")
        arrays = {name: archive[name] for name in required}

    scores = _numeric(arrays["scores"], "scores")
    if scores.ndim != 2 or min(scores.shape) < 1:
        raise ValueError("scores must have nonempty shape [N,K]")
    rows, encoders = scores.shape
    context = _numeric(arrays["context"], "context")
    if context.shape != (rows, len(Q_COLUMNS)):
        raise ValueError("context must have shape [N,15]")
    encoder_names = _strings(arrays["encoder_names"], (encoders,), "encoder_names")
    if len(set(encoder_names)) != encoders:
        raise ValueError("encoder_names must be unique")
    context_names = _strings(arrays["context_names"], (len(Q_COLUMNS),), "context_names")
    if tuple(context_names) != Q_COLUMNS:
        raise ValueError("context_names must match Q_COLUMNS exactly, including order")
    trial_ids = _strings(arrays["trial_ids"], (rows,), "trial_ids")
    if len(set(trial_ids)) != rows:
        raise ValueError("trial_ids must be unique within each file")
    result = {
        "scores": scores,
        "context": context,
        "encoder_names": encoder_names,
        "context_names": context_names,
        "trial_ids": trial_ids,
    }
    if require_labels:
        labels = _numeric(arrays["labels"], "labels")
        if labels.shape != (rows,) or not np.isin(labels, [0, 1]).all():
            raise ValueError("labels must have shape [N] and contain only 0 or 1")
        if len(np.unique(labels)) != 2:
            raise ValueError("Each fitting split requires both label classes")
        result["labels"] = labels.astype(np.int8)
    return result


def fit_cached(
    calibration_path: str | Path,
    validation_path: str | Path,
    output_path: str | Path,
    *,
    device: str = "cpu",
) -> Plugin:
    """Fit the unchanged full candidate pool using separate Cal and Val caches."""
    torch.set_num_threads(1)
    calibration = load_trials(calibration_path, require_labels=True)
    validation = load_trials(validation_path, require_labels=True)
    if tuple(calibration["encoder_names"]) != tuple(validation["encoder_names"]):
        raise ValueError("Calibration and validation encoder identity or order mismatch")
    overlap = set(calibration["trial_ids"]) & set(validation["trial_ids"])
    if overlap:
        raise ValueError(f"Calibration and validation share {len(overlap)} trial_ids")
    return Plugin.fit(
        calibration,
        validation,
        calibration["encoder_names"],
        output_path=output_path,
        device=device,
    )


def predict_cached(
    bundle_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
) -> np.ndarray:
    """Write only calibrated LLRs and the corresponding trial IDs."""
    torch.set_num_threads(1)
    trials = load_trials(input_path)
    fitted = Plugin.load(bundle_path)
    llr = fitted.predict(
        trials["scores"],
        trials["context"],
        trials["encoder_names"],
        context_names=trials["context_names"],
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        np.savez_compressed(stream, llr=llr, trial_ids=trials["trial_ids"])
    return llr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Fit on calibration and select on validation")
    fit.add_argument("--calibration", required=True, type=Path)
    fit.add_argument("--validation", required=True, type=Path)
    fit.add_argument("--output", required=True, type=Path)
    fit.add_argument("--device", default="cpu")
    predict = commands.add_parser("predict", help="Predict without reading labels")
    predict.add_argument("--bundle", required=True, type=Path)
    predict.add_argument("--input", required=True, type=Path)
    predict.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "fit":
            fitted = fit_cached(args.calibration, args.validation, args.output, device=args.device)
            print(f"Saved bundle: {args.output} ({fitted.payload['selection']['selected_family']})")
        else:
            llr = predict_cached(args.bundle, args.input, args.output)
            print(f"Saved {len(llr)} predictions: {args.output}")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
