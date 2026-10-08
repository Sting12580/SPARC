# Fitting and deployment

The published backend uses the unbounded anchored residual algorithm:

```text
source_llr  = source_affine(frozen_scores)
anchor_llr  = context_affine(source_llr, context, score_disagreement,
                            source_llr * context)
output_llr  = anchor_llr + selected_residual
```

The context-affine anchor has 33 inputs: the source LLR, 15 context features, score
standard deviation and range, and 15 LLR–context products. The anchor remains
fixed while residual candidates are fitted.

Calibration fits parameters and normalization statistics. Validation chooses
affine regularization, neural checkpoints, and one residual family. Test labels
are not accepted by the fitting API.

The predefined selection pool has 17 families:

- One unchanged context-affine anchor.
- Eight polynomial quality-residual families: full or duration-only views,
  with ridge coefficients `0.0001`, `0.001`, `0.01`, or `0.1`.
- Eight neural residual families: full or duration-only views, rank-loss weights
  `0` or `0.5`, and `cllr` or `joint` checkpoint rules. Each family has seeds
  `0`, `1`, and `2`.

The neural head has one hidden layer with eight units and zero-initialized output.
Training uses up to 160 epochs, validation every five epochs, and early stopping.
The complete configuration is in `delivery/training.py`.

An eligible family must have validation Cllr and EER no worse than context affine
within `1e-8`; neural metrics are averaged over the three fitted seeds. Among
eligible families, selection minimizes Cllr and resolves ties by simpler
complexity and family name. Neural deployment exports the selected family's
seed-0 parameters. The reported family means and the deployed model are different
quantities.

A fitted bundle stores affine parameters, normalization, correction weights,
input schemas, and selection metadata in JSON. Prediction requires exactly the
saved encoder identities and order. Missing context raises an error by default;
the Python API permits explicit `missing_context="source_affine"` fallback.

The core prediction backend always evaluates the stored neural parameters on CPU;
the `device` fitting option controls training. It does not turn the saved bundle's
prediction interface into GPU inference.
