# UAH Driver-Behaviour Temporal Classifier

PyTorch code and selected checkpoints for three-class UAH driving-behaviour
classification: `NORMAL`, `AGGRESSIVE`, and `DROWSY`.

The repository intentionally contains **no dataset files**. Point the training
commands at a local `uah_3class` directory containing `train.npz`, `val.npz`,
`test.npz`, `scaler.npz`, and `dataset_summary.json`.

## Start here

- [Project guide](luna_temporal_classifier/README.md)
- [Data and leakage audit](luna_temporal_classifier/DATA_AUDIT.md)
- [Experiment results](luna_temporal_classifier/RESULTS.md)
- [20-second window encoder](luna_temporal_classifier/model.py)
- [Causal long-context model](luna_temporal_classifier/context_tcn.py)
- [Context training](luna_temporal_classifier/train_context_tcn.py)

## Environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r luna_temporal_classifier/requirements.txt
```

The recorded environment used Python 3.9.6, NumPy 2.0.2, and PyTorch 2.8.0.

## Selected result

The selected model combines a frozen Inception1D window encoder with a causal
25-window TCN. Under the driver-disjoint protocol, the final D6 result was
75.03% accuracy and 77.38% macro-F1. See the results document for the complete
split contract, class metrics, limitations, and rejected alternatives.
