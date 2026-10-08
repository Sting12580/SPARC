# SPARC

**Portable Anchored Residual Calibration Plug-in for Speaker Verification**

SPARC calibrates scores from frozen speaker encoders using duration, voice activity,
and soft language context. The backend fits a source-affine model, a context-affine
anchor, and a validation-selected unbounded residual correction. It returns one
calibrated log-likelihood ratio (LLR) per trial.

## Quick start

The verified environment uses Python 3.10. Run these commands from the repository
root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m examples.synthetic_demo --output-dir outputs/synthetic
```

The synthetic example runs the full default fitting procedure, writes a JSON
bundle, reloads it, and verifies predictions and input-contract checks. It creates
all inputs locally and needs no audio, pretrained weights, or external datasets.
Its outputs verify the API; they are not speaker-verification benchmark results.

## Fit and predict on your own scores

Prepare separate calibration and validation NPZ files containing scores, context,
binary labels, named schemas, and trial IDs. See the [input contract](docs/INPUTS.md).

```bash
python -m examples.cached_trials fit \
  --calibration calibration.npz --validation validation.npz \
  --output outputs/sparc.json --device cpu

python -m examples.cached_trials predict \
  --bundle outputs/sparc.json --input prediction.npz \
  --output outputs/predictions.npz
```

The fitting API accepts calibration and validation splits. Prediction consumes
scores and context, without labels. Keep the evaluation split separate from
fitting and model selection. Trial-ID checks detect overlapping calibration and
validation trials; speaker-disjoint splitting remains the caller's responsibility.

The Python interface is also available directly:

```python
from delivery import Plugin, Q_COLUMNS

names = ("encoder_a", "encoder_b")
plugin = Plugin.fit(calibration, validation, names, output_path="sparc.json")
restored = Plugin.load("sparc.json")
llr = restored.predict(scores, context, names, context_names=Q_COLUMNS)
```

`calibration` and `validation` are dictionaries with `scores`, `context`, `labels`,
and `trial_ids`. Encoder identity and order must match the saved bundle.

## Included code

- [`delivery/plugin.py`](delivery/plugin.py): bundle loading, prediction, schema
  checks, and explicit missing-context fallback.
- [`delivery/training.py`](delivery/training.py): affine fitting, residual fitting,
  checkpoint selection, and the 17-family validation selector.
- [`delivery/metadata.py`](delivery/metadata.py): duration, VAD, and soft-language
  metadata, plus the 15-dimensional pair-context builder.
- [`examples/`](examples/): cached-input commands and a self-contained synthetic
  verification example.
- [`docs/METHOD.md`](docs/METHOD.md): fitting and selection details.

The original `delivery/` implementation is preserved byte for byte. Its optional
audio example, [`delivery/run_example.py`](delivery/run_example.py), requires
LibriSpeech audio, a prepared protocol, and a local CAMPPlus checkpoint. Pass
`--source-root`, `--prepared-root`, and `--model-directory` explicitly; its historical
defaults assume a local research checkout. `delivery/requirements.txt` records the
optional audio-stack dependency list. The core requirements above cover cached
score fitting and prediction; the audio stack is a separate installation path.

## Scope

This repository provides the reusable calibration backend and its public examples.
It contains no corpus audio, pretrained encoder weights, fitted study bundles, or
frozen study predictions. The original four-corpus, 28-condition paper tables
require the corresponding study inputs and experiment artifacts and cannot be
reproduced from the synthetic example. The optional audio extraction path was not
rerun as part of this code release.

`Plugin.fit` fits new affine anchors and residual candidates on the supplied
calibration and validation data; it does not replay the paper's frozen fits.

Validation selects a family using averages over seeds 0, 1, and 2 for neural
candidates; the exported neural model uses seed 0. Family-average eligibility
does not guarantee that this individual deployed seed improves every metric.
Performance depends on the corpus and score source, and the original study's
historical Test exposure remains a limitation.

[`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json) records hashes of the preserved
implementation. [`VERIFICATION.json`](VERIFICATION.json) records the executed
public-example checks.
