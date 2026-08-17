#!/usr/bin/env python3
"""Final D1-D5 fit of the selected 25-window causal context head."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from context_tcn import HierarchicalCausalTCN, count_parameters
from model import InceptionTemporalNet
from train_context_tcn import ContextDataset, evaluate, extract_features


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path, default=None,
                        help="Restore a paused D1-D5 context-head checkpoint")
    parser.add_argument("--skip-test", action="store_true",
                        help="Do not re-read D6 when resuming or extending training")
    args = parser.parse_args()
    if args.context != 25 or (args.resume is None and args.epochs != 20):
        raise ValueError("Fresh final-fit requires context=25 and epochs=20 from D5 selection")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    encoder_state = torch.load(args.encoder_checkpoint, map_location=device, weights_only=False)
    encoder = InceptionTemporalNet(width=48, dropout=0.15).to(device)
    encoder.load_state_dict(encoder_state["model_state"])
    train_features, train_data = extract_features(args.data / "train.npz", encoder, device)
    d5_features, d5_data = extract_features(args.data / "val.npz", encoder, device)
    features = np.concatenate((train_features, d5_features))
    labels = np.concatenate((train_data["y"], d5_data["y"]))
    trips = np.concatenate((train_data["trip_id"], d5_data["trip_id"]))
    starts = np.concatenate((train_data["window_start_s"], d5_data["window_start_s"]))
    feature_mean = features.mean(0, keepdims=True)
    feature_std = features.std(0, keepdims=True).clip(1e-4)
    features = (features - feature_mean) / feature_std
    train = ContextDataset(features, labels, trips, starts, args.context)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = HierarchicalCausalTCN(feature_dim=features.shape[1]).to(device)
    counts = np.bincount(labels, minlength=3)
    weights = len(labels) / (3.0 * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32, device=device), label_smoothing=0.03,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 50)
    history = []; start_epoch = 1; payload = None
    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        history = payload["history"]
        start_epoch = int(payload["epoch"]) + 1
        if "python_rng_state" in payload:
            random.setstate(payload["python_rng_state"])
        if "numpy_rng_state" in payload:
            np.random.set_state(payload["numpy_rng_state"])
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"])
        print(f"resumed={args.resume} next_epoch={start_epoch}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); total = 0.0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total += float(loss.item()) * len(y)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": total / len(train), "lr": optimizer.param_groups[0]["lr"]})
        payload = {
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "epoch": epoch, "history": history,
            "feature_mean": feature_mean, "feature_std": feature_std,
            "encoder_checkpoint": str(args.encoder_checkpoint),
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "selection_contract": "D5 selected context=25 and epoch=20; D1-D5 final fit; D6 one-shot test",
            "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        }
        atomic_save(payload, args.out / "last_checkpoint.pt")
        print(f"epoch={epoch:03d} loss={history[-1]['train_loss']:.4f} lr={history[-1]['lr']:.2e}", flush=True)

    if payload is None:
        raise RuntimeError("No checkpoint state is available")
    if args.skip_test:
        atomic_save(payload, args.out / "paused_checkpoint.pt")
        print(f"paused_without_test={args.out / 'paused_checkpoint.pt'}", flush=True)
        return

    # Extract and evaluate D6 only after all model choices and weights are fixed.
    test_features, test_data = extract_features(args.data / "test.npz", encoder, device)
    test_features = (test_features - feature_mean) / feature_std
    test = ContextDataset(test_features, test_data["y"], test_data["trip_id"], test_data["window_start_s"], args.context)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_metrics = evaluate(model, test_loader, device)
    report = {
        "architecture": "frozen Inception1D + HierarchicalCausalTCN",
        "parameters_context": count_parameters(model),
        "fixed_hyperparameters": payload["config"],
        "selection_contract": payload["selection_contract"],
        "test": test_metrics,
    }
    atomic_save(payload, args.out / "final_context_model.pt")
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (args.out / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
