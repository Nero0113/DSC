#!/usr/bin/env python3
"""Evaluate a frozen MultiScalePrefixTCN checkpoint on D5 only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import InceptionTemporalNet
from multiscale_prefix_tcn import MultiScalePrefixTCN, count_parameters
from train import metrics
from train_context_tcn import extract_features
from train_multiscale_prefix_tcn import PrefixDataset, infer, max_training_trip_len, trip_accuracy_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder_state = torch.load(args.encoder_checkpoint, map_location=device, weights_only=False)
    encoder = InceptionTemporalNet(width=48, dropout=0.15).to(device)
    encoder.load_state_dict(encoder_state["model_state"])
    validation_features, validation_data = extract_features(args.data / "val.npz", encoder, device)
    validation_features = (
        (validation_features - state["feature_mean"]) / state["feature_std"]
    ).astype(np.float32)
    with np.load(args.data / "train.npz", allow_pickle=False) as train_data:
        train_trip_max = max_training_trip_len(train_data["trip_id"])
    validation = PrefixDataset(
        validation_features, validation_data["y"], validation_data["trip_id"],
        validation_data["window_start_s"], int(state["config"]["context"]), train_trip_max,
    )
    loader = DataLoader(validation, batch_size=256, shuffle=False, num_workers=0)
    model = MultiScalePrefixTCN(feature_dim=128).to(device)
    model.load_state_dict(state["model_state"])
    labels, predictions, probabilities, indices = infer(model, loader, device)
    result = metrics(labels, predictions)
    report = {
        "architecture": "MultiScalePrefixTCN", "seed": int(state["config"]["seed"]),
        "parameters": count_parameters(model), "best_epoch": int(state["epoch"]),
        "validation": result,
        "per_trip_accuracy": trip_accuracy_report(validation, labels, predictions, indices),
        "test_evaluated": False,
        "baseline": {"accuracy": 0.714010, "macro_f1": 0.681006},
        "delta_vs_baseline": {
            "accuracy": result["accuracy"] - 0.714010,
            "macro_f1": result["macro_f1"] - 0.681006,
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    last_path = args.out / "last_checkpoint.pt"
    if last_path.exists():
        last_state = torch.load(last_path, map_location="cpu", weights_only=False)
        (args.out / "history.json").write_text(
            json.dumps(last_state["history"], indent=2, ensure_ascii=False)
        )
    with (args.out / "validation_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trip", "window_start_s", "true", "pred", "prob_0", "prob_1", "prob_2"])
        for row_index, dataset_index in enumerate(indices):
            target = validation.records[int(dataset_index)][0]
            writer.writerow([
                validation.trips_source[target], float(validation.starts_source[target]),
                int(labels[row_index]), int(predictions[row_index]), *probabilities[row_index].tolist(),
            ])
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
