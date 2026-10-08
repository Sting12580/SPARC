"""Synthetic API verification only; this example produces no paper results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from delivery import MetadataExtractor, Plugin, Q_COLUMNS, UtteranceMetadata
from .cached_trials import fit_cached, load_trials, predict_cached


ENCODER_NAMES = ("synthetic_encoder_a", "synthetic_encoder_b")


def _synthetic_trials(rng: np.random.Generator, rows: int, split: str) -> dict:
    """Construct feasible duration, VAD and soft-LID context without audio models."""
    extractor = MetadataExtractor(device="cpu")
    context = []
    for _ in range(rows):
        pair = []
        for _ in range(2):
            duration = float(rng.uniform(1.0, 12.0))
            pair.append(
                UtteranceMetadata(
                    duration_sec=duration,
                    vad_speech_duration_sec=duration * float(rng.uniform(0.5, 0.95)),
                    lid_probabilities=rng.dirichlet(np.ones(4)),
                    lid_embedding=rng.normal(size=6),
                )
            )
        context.append(extractor.pair_context(*pair))
    context = np.asarray(context)
    labels = (np.arange(rows) % 2).astype(np.int8)
    rng.shuffle(labels)
    signal = 0.8 * (2 * labels - 1)
    noise = rng.normal(size=(rows, len(ENCODER_NAMES)))
    scores = np.column_stack([signal, 1.1 * signal]) + noise
    scores += 0.25 * context[:, 13, None]
    trials = {
        "scores": scores,
        "context": context,
        "encoder_names": np.asarray(ENCODER_NAMES),
        "context_names": np.asarray(Q_COLUMNS),
        "trial_ids": np.asarray([f"synthetic-{split}-{i:05d}" for i in range(rows)]),
    }
    if split != "prediction":
        trials["labels"] = labels
    return trials


def _expect_rejection(call: Callable[[], object], expected: str) -> str:
    try:
        call()
    except ValueError as exc:
        if expected not in str(exc):
            raise AssertionError(f"Unexpected rejection: {exc}") from exc
        return str(exc)
    raise AssertionError(f"Expected ValueError containing {expected!r}")


def run_demo(output_dir: Path, *, seed: int = 7) -> dict:
    """Exercise the default fit, serialization, prediction and schema checks."""
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = {}
    for name, rows in (("calibration", 512), ("validation", 256), ("prediction", 128)):
        paths[name] = output_dir / f"{name}.npz"
        np.savez_compressed(paths[name], **_synthetic_trials(rng, rows, name))

    bundle_path = output_dir / "bundle.json"
    fitted = fit_cached(paths["calibration"], paths["validation"], bundle_path, device="cpu")
    trials = load_trials(paths["prediction"])
    before = fitted.predict(trials["scores"], trials["context"], trials["encoder_names"])
    loaded = Plugin.load(bundle_path)
    after = loaded.predict(trials["scores"], trials["context"], trials["encoder_names"])
    np.testing.assert_array_equal(before, after)
    saved = predict_cached(bundle_path, paths["prediction"], output_dir / "predictions.npz")
    np.testing.assert_array_equal(after, saved)
    with np.load(output_dir / "predictions.npz", allow_pickle=False) as predictions:
        assert set(predictions.files) == {"llr", "trial_ids"}
        np.testing.assert_array_equal(predictions["llr"], after)
        np.testing.assert_array_equal(predictions["trial_ids"], trials["trial_ids"])

    encoder_error = _expect_rejection(
        lambda: loaded.predict(
            trials["scores"][:, ::-1], trials["context"], trials["encoder_names"][::-1]
        ),
        "Encoder identity or order mismatch",
    )
    calibration = load_trials(paths["calibration"], require_labels=True)
    validation = load_trials(paths["validation"], require_labels=True)
    overlapping = dict(validation)
    overlapping["trial_ids"] = np.concatenate([
        calibration["trial_ids"][:1], validation["trial_ids"][1:]
    ])
    overlap_error = _expect_rejection(
        lambda: Plugin.fit(calibration, overlapping, ENCODER_NAMES, device="cpu"),
        "Calibration and validation share",
    )
    all_ids = np.concatenate([
        calibration["trial_ids"], validation["trial_ids"], trials["trial_ids"]
    ])
    assert len(np.unique(all_ids)) == len(all_ids)
    report = {
        "purpose": "Synthetic API verification only; not a paper result.",
        "seed": seed,
        "calibration_rows": len(calibration["scores"]),
        "validation_rows": len(validation["scores"]),
        "prediction_rows": len(trials["scores"]),
        "selected_family": fitted.payload["selection"]["selected_family"],
        "candidate_pool": fitted.payload["selection"]["candidate_pool"],
        "family_count": fitted.payload["selection"]["family_count"],
        "save_load_predictions_equal": True,
        "all_trial_ids_disjoint": True,
        "encoder_order_rejection": encoder_error,
        "calibration_validation_overlap_rejection": overlap_error,
        "prediction_labels_used": False,
    }
    (output_dir / "verification.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    report = run_demo(args.output_dir, seed=args.seed)
    print(report["purpose"])
    print(f"Verified full candidate fit, save/load, prediction and rejection checks: {args.output_dir}")


if __name__ == "__main__":
    main()
