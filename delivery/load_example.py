"""Load the smallest local source-audio corpus and one local encoder weight.

The example uses LibriSpeech source audio and the local CAMPPlus checkpoint.
The matching 15-dimensional context is read from the prepared LibriSpeech
protocol so this loader opens no second frontend model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import torch
import yaml


ENCODER_NAME = "funasr_campplus_cn_3k"
SPLIT_PARTITIONS = {
    "calibration": ("dev-clean",),
    "validation": ("dev-other",),
    "test": ("test-clean", "test-other"),
}


@dataclass(frozen=True)
class ExampleData:
    calibration: dict
    validation: dict
    prediction: dict
    encoder_names: tuple[str, ...]


class CampPlusEncoder:
    """One-path callable backed by one local CAMPPlus checkpoint."""

    def __init__(self, model_directory: str | Path, device: str = "cpu") -> None:
        from funasr.models.campplus.model import CAMPPlus

        directory = Path(model_directory)
        config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        self.device = torch.device(device)
        self.model = CAMPPlus(**config["model_conf"])
        self.model.load_state_dict(
            torch.load(
                directory / "campplus_cn_common.bin",
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        self.model.to(self.device).eval()
        self.cache: dict[Path, np.ndarray] = {}

    @torch.inference_mode()
    def __call__(self, audio_path: str | Path) -> np.ndarray:
        path = Path(audio_path)
        if path in self.cache:
            return self.cache[path]

        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        if sample_rate != 16_000:
            divisor = int(np.gcd(sample_rate, 16_000))
            waveform = resample_poly(
                waveform,
                16_000 // divisor,
                sample_rate // divisor,
            ).astype(np.float32)

        from torchaudio.compliance.kaldi import fbank

        features = fbank(
            torch.from_numpy(waveform).unsqueeze(0),
            num_mel_bins=80,
            sample_frequency=16_000,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            window_type="povey",
        )
        features = features - features.mean(dim=0, keepdim=True)
        embedding = self.model(features.unsqueeze(0).to(self.device))
        if isinstance(embedding, tuple):
            embedding = embedding[-1]
        value = embedding.squeeze(0).cpu().numpy().astype(np.float32)
        value /= max(float(np.linalg.norm(value)), 1e-12)
        self.cache[path] = value
        return value


def _audio_index(source_root: Path) -> dict[str, Path]:
    paths = {}
    for partition in SPLIT_PARTITIONS.values():
        for name in partition:
            for path in (source_root / name).rglob("*.flac"):
                if path.stem in paths:
                    raise ValueError(f"Duplicate LibriSpeech observation: {path.stem}")
                paths[path.stem] = path
    if not paths:
        raise FileNotFoundError(f"No LibriSpeech audio found below {source_root}")
    return paths


def _observation_path(observation_id: str, audio: dict[str, Path]) -> Path:
    key = str(observation_id).removeprefix("librispeech:")
    if key not in audio:
        raise FileNotFoundError(f"Missing source audio for {observation_id}")
    return audio[key]


def _selected_rows(labels: np.ndarray, target_trials: int, nontarget_trials: int) -> np.ndarray:
    target = np.flatnonzero(labels == 1)[:target_trials]
    nontarget = np.flatnonzero(labels == 0)[:nontarget_trials]
    if len(target) != target_trials or len(nontarget) != nontarget_trials:
        raise ValueError("The requested example is larger than the available protocol")
    return np.sort(np.r_[target, nontarget])


def _load_split(
    prepared_root: Path,
    split: str,
    rows: np.ndarray,
    audio: dict[str, Path],
    encoder: CampPlusEncoder,
) -> dict:
    root = prepared_root / "inputs" / "librispeech" / split
    metadata = pd.read_parquet(root / "metadata.parquet").iloc[rows].reset_index(drop=True)
    context = np.asarray(np.load(root / "q.npy", mmap_mode="r")[rows], dtype=np.float64)
    scores = np.empty((len(rows), 1), dtype=np.float64)
    for index, trial in metadata.iterrows():
        enroll = encoder(_observation_path(trial.enroll_observation_id, audio))
        test = encoder(_observation_path(trial.test_observation_id, audio))
        scores[index, 0] = float(np.dot(enroll, test))
    result = {
        "scores": scores,
        "context": context,
        "trial_ids": metadata.trial_id.astype(str).to_numpy(),
    }
    labels_path = root / "y.npy"
    if labels_path.exists():
        result["labels"] = np.asarray(
            np.load(labels_path, mmap_mode="r")[rows],
            dtype=np.int8,
        )
    return result


def load_example(
    source_root: str | Path,
    prepared_root: str | Path,
    model_directory: str | Path,
    *,
    device: str = "cpu",
    target_trials: int = 8,
    nontarget_trials: int = 16,
    prediction_trials: int = 24,
) -> ExampleData:
    """Load source audio, one encoder, and three small protocol slices."""

    source_root = Path(source_root)
    prepared_root = Path(prepared_root)
    audio = _audio_index(source_root)
    encoder = CampPlusEncoder(model_directory, device=device)

    selected = {}
    for split in ("calibration", "validation"):
        path = prepared_root / "inputs" / "librispeech" / split / "y.npy"
        labels = np.load(path, mmap_mode="r")
        selected[split] = _selected_rows(labels, target_trials, nontarget_trials)
    test_metadata = pd.read_parquet(
        prepared_root / "inputs" / "librispeech" / "test" / "metadata.parquet",
        columns=["trial_id"],
    )
    prediction_rows = np.arange(min(prediction_trials, len(test_metadata)))

    return ExampleData(
        calibration=_load_split(
            prepared_root, "calibration", selected["calibration"], audio, encoder
        ),
        validation=_load_split(
            prepared_root, "validation", selected["validation"], audio, encoder
        ),
        prediction=_load_split(
            prepared_root, "test", prediction_rows, audio, encoder
        ),
        encoder_names=(ENCODER_NAME,),
    )


__all__ = ["CampPlusEncoder", "ENCODER_NAME", "ExampleData", "load_example"]
