"""Portable fitted backend for the final unbounded residual plug-in."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from .metadata import CONTEXT_NAMES


Q_COLUMNS = CONTEXT_NAMES


def _atomic_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _quality_raw(context: np.ndarray, q: np.ndarray, view: str) -> np.ndarray:
    enroll = np.log(np.maximum(q[:, 7], 0.01))
    test = np.log(np.maximum(q[:, 8], 0.01))
    features = np.column_stack(
        [
            2 * enroll * test,
            enroll * enroll + test * test,
            enroll + test,
            q[:, 12],
            np.sqrt(np.clip(q[:, 13], 0, np.log(2))),
            1 - np.clip(q[:, 14], -1, 1),
        ]
    )
    if view == "duration":
        features = features[:, :3]
    elif view != "full":
        raise ValueError(f"Unknown residual view: {view!r}")
    return np.column_stack([context, features, context[:, None] * features])


def _normalize(raw: np.ndarray, stats: dict) -> np.ndarray:
    output = (
        raw - np.asarray(stats["mean"], dtype=np.float64)
    ) / np.asarray(stats["scale"], dtype=np.float64)
    output[:, ~np.asarray(stats["active"], dtype=bool)] = 0
    return np.clip(output, -float(stats["clip"]), float(stats["clip"]))


class Plugin:
    """One fitted context-affine anchor plus one selected residual head."""

    FORMAT = "context-affine-unbounded-residual-v1"

    def __init__(self, payload: dict):
        self.payload = copy.deepcopy(payload)
        if (
            payload.get("format") != self.FORMAT
            or payload.get("correction_mode") != "unbounded"
        ):
            raise ValueError("Expected an unbounded residual bundle")
        if payload.get("bound") is not None:
            raise ValueError("An unbounded bundle must not declare an output bound")

        self.names = tuple(payload["encoder_names"])
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Encoder identities must be unique and nonempty")
        if tuple(payload["context_names"]) != Q_COLUMNS:
            raise ValueError("Context schema does not match the 15-feature contract")

        self.source = payload["source_affine"]
        self.context = payload["context_affine"]
        if (
            len(self.source["weights"]) != len(self.names)
            or len(self.context["weights"]) != 33
        ):
            raise ValueError("Anchor dimensions do not match input schema")

        self.correction = payload["correction"]
        kind = self.correction["kind"]
        if kind not in {"context_affine", "quality", "neural"}:
            raise ValueError("Unsupported correction kind")
        self.parameters = {}
        if kind == "neural":
            self.parameters = {
                name: torch.as_tensor(np.asarray(value, dtype="float32"))
                for name, value in self.correction["state"].items()
            }

    def save(self, path: str | Path) -> None:
        _atomic_json(path, self.payload)

    @classmethod
    def load(cls, path: str | Path) -> "Plugin":
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(payload)

    @classmethod
    def fit(
        cls,
        calibration,
        validation,
        encoder_names,
        *,
        output_path: str | Path | None = None,
        device: str = "cpu",
        progress=None,
    ) -> "Plugin":
        """Fit and validation-select one Final model from user-supplied splits."""

        from .training import fit_plugin

        return fit_plugin(
            calibration,
            validation,
            encoder_names,
            output_path=output_path,
            device=device,
            progress=progress,
        )

    def _inputs(
        self,
        scores,
        context,
        encoder_names,
        context_names,
        missing_context,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if tuple(encoder_names) != self.names:
            raise ValueError("Encoder identity or order mismatch")
        scores = np.asarray(scores, dtype=np.float64)
        if (
            scores.ndim != 2
            or scores.shape[1] != len(self.names)
            or not np.isfinite(scores).all()
        ):
            raise ValueError("Scores require a finite [N,K] matrix")
        if missing_context not in {"error", "source_affine"}:
            raise ValueError("Unknown missing-context policy")
        if context is None:
            if missing_context == "source_affine":
                return scores, None
            raise ValueError("Context is required; fallback must be explicit")
        if tuple(context_names) != Q_COLUMNS:
            raise ValueError("Context identity or order mismatch")
        context = np.asarray(context, dtype=np.float64)
        if context.shape != (len(scores), 15) or not np.isfinite(context).all():
            raise ValueError("Context requires a finite [N,15] matrix")
        return scores, context

    def source_llr(self, scores: np.ndarray) -> np.ndarray:
        normalized = (
            scores - np.asarray(self.source["mean"])
        ) / np.asarray(self.source["scale"])
        return normalized @ np.asarray(self.source["weights"]) + self.source["bias"]

    def context_llr(
        self,
        scores: np.ndarray,
        context: np.ndarray,
        base: np.ndarray | None = None,
    ) -> np.ndarray:
        base = self.source_llr(scores) if base is None else base
        features = np.column_stack(
            [
                base,
                context,
                scores.std(axis=1),
                np.ptp(scores, axis=1),
                base[:, None] * context,
            ]
        )
        normalized = (
            features - np.asarray(self.context["mean"])
        ) / np.asarray(self.context["scale"])
        return normalized @ np.asarray(self.context["weights"]) + self.context["bias"]

    def predict(
        self,
        scores,
        context,
        encoder_names,
        *,
        context_names=Q_COLUMNS,
        missing_context="error",
        batch_size=131072,
    ) -> np.ndarray:
        """Return the Final calibrated LLR for each input trial."""

        scores, context = self._inputs(
            scores,
            context,
            encoder_names,
            context_names,
            missing_context,
        )
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        if context is None:
            return self.source_llr(scores)

        output = np.empty(len(scores), dtype=np.float64)
        for start in range(0, len(scores), batch_size):
            end = min(start + batch_size, len(scores))
            score_batch = scores[start:end]
            context_batch = context[start:end]
            anchor = self.context_llr(score_batch, context_batch)
            if self.correction["kind"] == "context_affine":
                output[start:end] = anchor
                continue

            residual_input = _quality_raw(
                anchor,
                context_batch,
                self.correction["view"],
            )
            if self.correction["kind"] == "neural":
                residual_input = np.column_stack(
                    [
                        residual_input,
                        context_batch
                        if self.correction["view"] == "full"
                        else context_batch[:, :12],
                        score_batch,
                    ]
                )
            normalized = _normalize(
                residual_input,
                self.correction["normalization"],
            )

            if self.correction["kind"] == "quality":
                theta = np.asarray(self.correction["theta"])
                delta = normalized @ theta[:-1] + theta[-1]
            else:
                with torch.inference_mode():
                    values = torch.as_tensor(normalized.astype("float32"))
                    parameters = self.parameters
                    hidden = torch.relu(
                        torch.nn.functional.linear(
                            values,
                            parameters["head.0.weight"],
                            parameters["head.0.bias"],
                        )
                    )
                    delta = torch.nn.functional.linear(
                        hidden,
                        parameters["head.3.weight"],
                        parameters["head.3.bias"],
                    ).squeeze(-1)
                    delta = delta.numpy().astype(np.float64)
            if not np.isfinite(delta).all():
                raise ValueError("Nonfinite correction output")
            output[start:end] = anchor + delta

        if not np.isfinite(output).all():
            raise ValueError("Nonfinite output")
        return output


__all__ = ["Plugin", "Q_COLUMNS"]
