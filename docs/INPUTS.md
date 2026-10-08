# Input contract

Each row is an enrollment–test speaker-verification trial. Scores and context must
refer to the same trial in the same row. `1` denotes the same speaker and `0`
denotes different speakers.

The cached-input CLI loads NPZ files with `allow_pickle=False`:

| Key | Shape | Contents |
| --- | --- | --- |
| `scores` | `[N,K]` | Finite frozen encoder scores |
| `context` | `[N,15]` | Finite pair-context values in the order below |
| `labels` | `[N]` | Binary 0/1 labels; required only for fitting |
| `encoder_names` | `[K]` | Unique nonempty Unicode strings in score-column order |
| `context_names` | `[15]` | Unicode strings exactly matching `Q_COLUMNS` |
| `trial_ids` | `[N]` | Unique nonempty Unicode strings |

Both fitting splits must contain target and nontarget trials. Calibration and
validation trial IDs must be disjoint. Use Unicode arrays such as
`np.asarray(names, dtype=str)`, rather than object arrays.

```python
import numpy as np
from delivery import Q_COLUMNS

np.savez_compressed(
    "calibration.npz",
    scores=scores,
    context=context,
    labels=labels,
    encoder_names=np.asarray(encoder_names, dtype=str),
    context_names=np.asarray(Q_COLUMNS, dtype=str),
    trial_ids=np.asarray(trial_ids, dtype=str),
)
```

Prediction files use the same named input fields and can omit `labels`. Prediction
output contains `trial_ids` and `llr` with one value per input row.

## Context order

| Index | Name | Meaning |
| --- | --- | --- |
| 0 | `min_duration` | Minimum endpoint duration in seconds |
| 1 | `duration_ratio` | Minimum duration divided by maximum duration |
| 2 | `short_duration_risk` | Inverse square root of minimum duration |
| 3 | `enroll__duration_sec` | Enrollment duration |
| 4 | `test__duration_sec` | Test duration |
| 5 | `max__duration_sec` | Maximum endpoint duration |
| 6 | `absdiff__duration_sec` | Absolute duration difference |
| 7 | `enroll__vad_speech_duration_sec` | Enrollment VAD speech duration |
| 8 | `test__vad_speech_duration_sec` | Test VAD speech duration |
| 9 | `min__vad_speech_duration_sec` | Minimum VAD speech duration |
| 10 | `max__vad_speech_duration_sec` | Maximum VAD speech duration |
| 11 | `absdiff__vad_speech_duration_sec` | Absolute VAD duration difference |
| 12 | `lid_soft_overlap` | Dot product of endpoint language probabilities |
| 13 | `lid_js` | Jensen–Shannon divergence using natural logarithms |
| 14 | `lid_embedding_cosine` | Cosine between endpoint language embeddings |

## Build context from audio

`MetadataExtractor` loads VAD and language models lazily. This optional path
requires the audio dependencies and model access described in the root README.

```python
from delivery import MetadataExtractor

extractor = MetadataExtractor(device="cpu")
enroll = extractor.extract("enroll.wav")
test = extractor.extract("test.wav")
context_row = extractor.pair_context(enroll, test)
```

Alternatively, construct `UtteranceMetadata` from already extracted duration, VAD
speech duration, language probabilities, and language embeddings, then call
`pair_context`. The endpoint language probability vectors must use identical
class order, and the embeddings must come from the same feature space. Maintain
the same metadata extraction configuration for fitting and deployment.
