# Standalone delivery

The directory is intentionally flat.

- `plugin.py`, `training.py`, and `metadata.py` are the reusable implementation.
- `load_example.py` loads the smallest locally available source-audio corpus
  (LibriSpeech, about 2.74 GB), one local CAMPPlus checkpoint, and small slices
  of the existing calibration, validation, and test protocols.
- `run_example.py` calls that loader, fits one Final bundle, loads the saved
  bundle, and produces test predictions. This executable example is also the
  delivery smoke test.

No dataset, pretrained encoder, or fitted calibration bundle is copied into
this directory.

```powershell
python -m delivery.run_example --output final_bundle.json
```
