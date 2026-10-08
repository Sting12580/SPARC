"""Load the local example, fit one bundle, restore it, and predict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .load_example import load_example
from .plugin import Plugin


REPOSITORY = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPOSITORY / "datasets" / "librispeech_source" / "LibriSpeech"
DEFAULT_PREPARED = REPOSITORY / "datasets" / "librispeech_csl_v1"
DEFAULT_MODEL = Path(
    r"D:\Github\SafeASV\embeddings\pretrained_models\funasr\models\iic\speech_campplus_sv_zh-cn_16k-common"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--model-directory", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--target-trials", type=int, default=8)
    parser.add_argument("--nontarget-trials", type=int, default=16)
    parser.add_argument("--prediction-trials", type=int, default=24)
    args = parser.parse_args()

    data = load_example(
        args.source_root,
        args.prepared_root,
        args.model_directory,
        device=args.device,
        target_trials=args.target_trials,
        nontarget_trials=args.nontarget_trials,
        prediction_trials=args.prediction_trials,
    )
    fitted = Plugin.fit(
        data.calibration,
        data.validation,
        data.encoder_names,
        output_path=args.output,
        device=args.device,
        progress=lambda event: print(json.dumps(event), flush=True),
    )
    restored = Plugin.load(args.output)
    scores = restored.predict(
        data.prediction["scores"],
        data.prediction["context"],
        data.encoder_names,
    )
    print(
        json.dumps(
            {
                "bundle": str(args.output.resolve()),
                "selected_family": fitted.payload["selection"]["selected_family"],
                "prediction_rows": len(scores),
                "finite_predictions": bool(np.isfinite(scores).all()),
                "llr_min": float(scores.min()),
                "llr_max": float(scores.max()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
