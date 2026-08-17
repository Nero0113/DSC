#!/usr/bin/env python3
"""Evaluate the selected causal session smoother (alpha fixed on D5)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from context_tcn import HierarchicalCausalTCN
from model import InceptionTemporalNet
from train import metrics
from train_context_tcn import ContextDataset, extract_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--context-checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    if abs(args.alpha - 0.05) > 1e-12:
        raise ValueError("alpha=0.05 is frozen from D5; do not tune it on D6")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    encoder_state = torch.load(args.encoder_checkpoint, map_location=device, weights_only=False)
    context_state = torch.load(args.context_checkpoint, map_location=device, weights_only=False)
    encoder = InceptionTemporalNet(width=48, dropout=0.15).to(device)
    encoder.load_state_dict(encoder_state["model_state"])
    features, data = extract_features(args.data / f"{args.split}.npz", encoder, device)
    features = (features - context_state["feature_mean"]) / context_state["feature_std"]
    dataset = ContextDataset(features, data["y"], data["trip_id"], data["window_start_s"], context=25)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    model = HierarchicalCausalTCN(feature_dim=features.shape[1]).to(device)
    model.load_state_dict(context_state["model_state"])
    model.eval()
    logits = np.empty((len(data["y"]), 3), dtype=np.float32)
    with torch.no_grad():
        for x, _, indices in loader:
            logits[indices.numpy()] = model(x.to(device)).cpu().numpy()

    predictions = np.empty(len(data["y"]), dtype=np.int64)
    for trip in np.unique(data["trip_id"]):
        indices = np.where(data["trip_id"] == trip)[0]
        indices = indices[np.argsort(data["window_start_s"][indices])]
        state = None
        for index in indices:
            state = logits[index].copy() if state is None else args.alpha * logits[index] + (1 - args.alpha) * state
            predictions[index] = int(state.argmax())
    result = {
        "split": args.split,
        "causal": True,
        "session_reset": True,
        "alpha_selected_on_D5": args.alpha,
        **metrics(data["y"], predictions),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
