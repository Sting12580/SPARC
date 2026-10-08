"""Automatic utterance metadata and the plug-in's 15-dimensional pair context."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


TARGET_SAMPLE_RATE = 16_000
DEFAULT_LID_MODEL = "speechbrain/lang-id-voxlingua107-ecapa"
DEFAULT_VAD_MODEL = "snakers4/silero-vad"
DEFAULT_VAD_PACKAGE_VERSION = "6.2.1"
DEFAULT_VAD_CONFIG = {
    "sampling_rate": TARGET_SAMPLE_RATE,
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 100,
    "speech_pad_ms": 30,
}
CONTEXT_NAMES = (
    "min_duration",
    "duration_ratio",
    "short_duration_risk",
    "enroll__duration_sec",
    "test__duration_sec",
    "max__duration_sec",
    "absdiff__duration_sec",
    "enroll__vad_speech_duration_sec",
    "test__vad_speech_duration_sec",
    "min__vad_speech_duration_sec",
    "max__vad_speech_duration_sec",
    "absdiff__vad_speech_duration_sec",
    "lid_soft_overlap",
    "lid_js",
    "lid_embedding_cosine",
)


VadFunction = Callable[[np.ndarray, int], float]
LidFunction = Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]


def _unit_vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not len(array) or not np.isfinite(array).all() or norm <= 0:
        raise ValueError(f"{name} must be a finite nonzero vector.")
    return array / norm


@dataclass(frozen=True)
class UtteranceMetadata:
    duration_sec: float
    vad_speech_duration_sec: float
    lid_probabilities: np.ndarray
    lid_embedding: np.ndarray

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration_sec) or self.duration_sec <= 0:
            raise ValueError("duration_sec must be finite and positive.")
        if (
            not math.isfinite(self.vad_speech_duration_sec)
            or self.vad_speech_duration_sec < 0
            or self.vad_speech_duration_sec > self.duration_sec + 1e-6
        ):
            raise ValueError("VAD speech duration must lie within the audio duration.")

        probabilities = np.asarray(self.lid_probabilities, dtype=np.float32).reshape(-1)
        if (
            not len(probabilities)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or float(probabilities.sum()) <= 0
        ):
            raise ValueError("LID probabilities must be finite and nonnegative.")
        probabilities = probabilities / probabilities.sum()
        object.__setattr__(self, "lid_probabilities", probabilities)
        object.__setattr__(
            self,
            "lid_embedding",
            _unit_vector(self.lid_embedding, "LID embedding"),
        )


class MetadataExtractor:
    """Extract duration, Silero VAD, and SpeechBrain soft-LID metadata.

    Models are loaded lazily. Optional functions exist for tests or controlled
    replacement, while the normal user-facing plug-in only needs audio paths.
    """

    def __init__(
        self,
        *,
        sample_rate: int = TARGET_SAMPLE_RATE,
        lid_model: str = DEFAULT_LID_MODEL,
        lid_cache_dir: str | Path | None = None,
        device: str = "auto",
        vad_function: VadFunction | None = None,
        lid_function: LidFunction | None = None,
    ) -> None:
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError("The verified metadata path uses 16 kHz audio.")
        self.sample_rate = int(sample_rate)
        self.lid_model = str(lid_model)
        self.lid_cache_dir = Path(lid_cache_dir) if lid_cache_dir else (
            Path.home()
            / ".cache"
            / "contextual-calibration"
            / "speechbrain"
            / "lang-id-voxlingua107-ecapa"
        )
        self.device = self._resolve_device(device)
        self._vad_function = vad_function
        self._lid_function = lid_function
        self._vad_model = None
        self._vad_timestamps = None
        self._lid_classifier = None
        self._cache: dict[str, UtteranceMetadata] = {}

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return str(device)
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def configuration(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "lid_model": self.lid_model,
            "vad_model": DEFAULT_VAD_MODEL,
            "vad_package_version": DEFAULT_VAD_PACKAGE_VERSION,
            "vad": dict(DEFAULT_VAD_CONFIG),
            "context_names": list(CONTEXT_NAMES),
            "custom_vad": self._callable_name(self._vad_function),
            "custom_lid": self._callable_name(self._lid_function),
        }

    @staticmethod
    def _callable_name(function: Callable | None) -> str | None:
        if function is None:
            return None
        name = getattr(function, "__qualname__", None) or getattr(
            function,
            "__name__",
            type(function).__qualname__,
        )
        return f"{function.__module__}.{name}"

    def extract(self, audio_path: str | Path) -> UtteranceMetadata:
        path = Path(audio_path).expanduser().resolve()
        key = str(path)
        if key in self._cache:
            return self._cache[key]
        if not path.is_file():
            raise FileNotFoundError(path)

        waveform = self._load_audio(path)
        duration = len(waveform) / self.sample_rate
        vad_duration = self._extract_vad(waveform)
        probabilities, embedding = self._extract_lid(waveform)
        value = UtteranceMetadata(
            duration_sec=float(duration),
            vad_speech_duration_sec=float(vad_duration),
            lid_probabilities=probabilities,
            lid_embedding=embedding,
        )
        self._cache[key] = value
        return value

    def extract_many(self, audio_paths) -> list[UtteranceMetadata]:
        return [self.extract(path) for path in audio_paths]

    def clear_cache(self) -> None:
        self._cache.clear()

    def pair_context(
        self,
        enroll: UtteranceMetadata,
        test: UtteranceMetadata,
    ) -> np.ndarray:
        if enroll.lid_probabilities.shape != test.lid_probabilities.shape:
            raise ValueError("The two LID probability vectors use different label orders.")
        if enroll.lid_embedding.shape != test.lid_embedding.shape:
            raise ValueError("The two LID embeddings have different dimensions.")

        duration_a, duration_b = enroll.duration_sec, test.duration_sec
        vad_a = enroll.vad_speech_duration_sec
        vad_b = test.vad_speech_duration_sec
        minimum_duration = min(duration_a, duration_b)
        lid_overlap = float(np.dot(enroll.lid_probabilities, test.lid_probabilities))
        lid_js = self._jensen_shannon(
            enroll.lid_probabilities,
            test.lid_probabilities,
        )
        lid_cosine = float(np.dot(enroll.lid_embedding, test.lid_embedding))

        context = np.asarray(
            [
                minimum_duration,
                minimum_duration / max(duration_a, duration_b),
                1.0 / math.sqrt(max(minimum_duration, 1e-6)),
                duration_a,
                duration_b,
                max(duration_a, duration_b),
                abs(duration_a - duration_b),
                vad_a,
                vad_b,
                min(vad_a, vad_b),
                max(vad_a, vad_b),
                abs(vad_a - vad_b),
                lid_overlap,
                lid_js,
                lid_cosine,
            ],
            dtype=np.float64,
        )
        if context.shape != (15,) or not np.isfinite(context).all():
            raise ValueError("Automatic pair context must be a finite 15-vector.")
        return context

    def _load_audio(self, path: Path) -> np.ndarray:
        try:
            import soundfile as sf
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise RuntimeError("soundfile and scipy are required for metadata extraction.") from exc

        waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        if source_rate != self.sample_rate:
            divisor = math.gcd(int(source_rate), self.sample_rate)
            waveform = resample_poly(
                waveform,
                self.sample_rate // divisor,
                int(source_rate) // divisor,
            ).astype(np.float32)
        if not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError(f"Invalid waveform: {path}")
        return np.asarray(waveform, dtype=np.float32)

    def _extract_vad(self, waveform: np.ndarray) -> float:
        if self._vad_function is not None:
            return float(self._vad_function(waveform, self.sample_rate))
        if self._vad_model is None:
            try:
                from silero_vad import get_speech_timestamps, load_silero_vad
            except ImportError as exc:
                raise RuntimeError(
                    "silero-vad is required for the default metadata extractor."
                ) from exc
            self._vad_model = load_silero_vad()
            self._vad_timestamps = get_speech_timestamps

        import torch

        intervals = self._vad_timestamps(
            torch.from_numpy(waveform),
            self._vad_model,
            **DEFAULT_VAD_CONFIG,
        )
        return sum(int(item["end"]) - int(item["start"]) for item in intervals) / self.sample_rate

    def _extract_lid(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._lid_function is not None:
            probabilities, embedding = self._lid_function(waveform, self.sample_rate)
            return np.asarray(probabilities), np.asarray(embedding)
        if self._lid_classifier is None:
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
            except ImportError:
                try:
                    from speechbrain.pretrained import EncoderClassifier
                except ImportError as exc:
                    raise RuntimeError(
                        "speechbrain is required for the default soft-LID extractor."
                    ) from exc
            self._lid_classifier = EncoderClassifier.from_hparams(
                source=self.lid_model,
                savedir=str(self.lid_cache_dir),
                run_opts={"device": self.device},
            )

        import torch

        with torch.inference_mode():
            wave = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
            lengths = torch.ones(1, device=self.device)
            encoded = self._lid_classifier.encode_batch(
                wave,
                lengths,
                normalize=False,
            )
            raw = self._lid_classifier.mods.classifier(encoded).squeeze(1)
            probabilities = torch.exp(raw)
            if not torch.allclose(
                probabilities.sum(dim=-1),
                torch.ones(1, device=probabilities.device),
                atol=1e-5,
            ):
                probabilities = torch.softmax(raw, dim=-1)
            embedding = encoded.squeeze(1)
        return (
            probabilities.squeeze(0).cpu().numpy().astype(np.float32),
            embedding.squeeze(0).cpu().numpy().astype(np.float32),
        )

    @staticmethod
    def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
        midpoint = 0.5 * (left + right)
        left_term = np.where(
            left > 0,
            left * np.log(np.maximum(left, 1e-30) / np.maximum(midpoint, 1e-30)),
            0.0,
        )
        right_term = np.where(
            right > 0,
            right * np.log(np.maximum(right, 1e-30) / np.maximum(midpoint, 1e-30)),
            0.0,
        )
        return float(0.5 * (left_term.sum() + right_term.sum()))
