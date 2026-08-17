#!/usr/bin/env python3
"""Train a frozen-window-encoder + causal long-context TCN on D1-D4/D5."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from context_tcn import HierarchicalCausalTCN, count_parameters
from model import InceptionTemporalNet
from train import metrics


class ContextDataset(Dataset):
    def __init__(self, features, labels, trips, starts, context: int):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.indices = []
        for trip in np.unique(trips):
            ordered = np.where(trips == trip)[0]
            ordered = ordered[np.argsort(starts[ordered])]
            for position, target in enumerate(ordered):
                begin = max(0, position - context + 1)
                source = ordered[begin:position + 1]
                if len(source) < context:
                    source = np.concatenate((np.repeat(source[:1], context - len(source)), source))
                self.indices.append((int(target), source))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        target, source = self.indices[index]
        x = torch.from_numpy(self.features[source].T.copy())
        return x, int(self.labels[target]), target


@torch.no_grad()
def extract_features(npz_path: Path, encoder: InceptionTemporalNet, device: torch.device):
    data = np.load(npz_path, allow_pickle=False)
    x = np.clip(data["X"], -12.0, 12.0).astype(np.float32)
    output = []
    encoder.eval()
    for begin in range(0, len(x), 256):
        batch = torch.from_numpy(x[begin:begin + 256]).to(device)
        temporal = encoder.features(encoder.stem(batch))
        embedding = torch.cat((encoder.avg_pool(temporal), encoder.max_pool(temporal)), dim=1).squeeze(-1)
        raw_delta = batch[..., 1:] - batch[..., :-1]
        stats = torch.cat((
            batch.mean(-1), batch.std(-1, unbiased=False),
            raw_delta.abs().mean(-1), raw_delta.std(-1, unbiased=False),
        ), dim=1)
        output.append(torch.cat((embedding, stats), dim=1).cpu())
    return torch.cat(output).numpy(), data


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); labels = []; predictions = []
    for x, y, _ in loader:
        labels.append(y); predictions.append(model(x.to(device)).argmax(1).cpu())
    y = torch.cat(labels).numpy(); pred = torch.cat(predictions).numpy()
    return metrics(y, pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume model/optimizer/scheduler/history from last_checkpoint.pt")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    checkpoint = torch.load(args.encoder_checkpoint, map_location=device, weights_only=False)
    encoder = InceptionTemporalNet(width=int(checkpoint.get("width", 48))).to(device)
    encoder.load_state_dict(checkpoint["model_state"])
    train_features, train_data = extract_features(args.data / "train.npz", encoder, device)
    validation_features, validation_data = extract_features(args.data / "val.npz", encoder, device)
    feature_mean = train_features.mean(0, keepdims=True)
    feature_std = train_features.std(0, keepdims=True).clip(1e-4)
    train_features = (train_features - feature_mean) / feature_std
    validation_features = (validation_features - feature_mean) / feature_std
    train = ContextDataset(train_features, train_data["y"], train_data["trip_id"], train_data["window_start_s"], args.context)
    validation = ContextDataset(validation_features, validation_data["y"], validation_data["trip_id"], validation_data["window_start_s"], args.context)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = HierarchicalCausalTCN(feature_dim=train_features.shape[1]).to(device)
    counts = np.bincount(train_data["y"], minlength=3)
    weights = len(train_data["y"]) / (3.0 * counts)
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32, device=device), label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 50)
    best = -1.0; stale = 0; history = []; start_epoch = 1
    if args.resume is not None:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        best = float(resume["best_macro_f1"])
        stale = int(resume["stale"])
        history = resume["history"]
        start_epoch = int(resume["epoch"]) + 1
        if "python_rng_state" in resume:
            random.setstate(resume["python_rng_state"])
        if "numpy_rng_state" in resume:
            np.random.set_state(resume["numpy_rng_state"])
        if "torch_rng_state" in resume:
            torch.set_rng_state(resume["torch_rng_state"])
        print(f"resumed={args.resume} next_epoch={start_epoch}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); total = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total += float(loss.item()) * len(y)
        scheduler.step(); result = evaluate(model, validation_loader, device)
        history.append({"epoch": epoch, "train_loss": total / len(train), **result})
        # Save every genuine best.  A minimum-delta threshold belongs to a
        # stopping rule, not to checkpoint selection; otherwise a small real
        # improvement can be silently discarded.
        improved = result["macro_f1"] > best + 1e-8
        stale = 0 if improved else stale + 1
        payload = {
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "epoch": epoch,
            "best_macro_f1": max(best, result["macro_f1"]), "stale": stale,
            "history": history, "feature_mean": feature_mean, "feature_std": feature_std,
            "encoder_checkpoint": str(args.encoder_checkpoint),
            "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }
        temporary = args.out / "last_checkpoint.pt.tmp"; torch.save(payload, temporary); temporary.replace(args.out / "last_checkpoint.pt")
        if improved:
            best = result["macro_f1"]
            temporary = args.out / "best_model.pt.tmp"; torch.save(payload, temporary); temporary.replace(args.out / "best_model.pt")
        print(f"epoch={epoch:03d} loss={total/len(train):.4f} val_acc={result['accuracy']:.4f} val_f1={result['macro_f1']:.4f}", flush=True)
        if stale >= args.patience: break
    best_state = torch.load(args.out / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state"])
    result = evaluate(model, validation_loader, device)
    report = {"architecture": "frozen Inception1D + HierarchicalCausalTCN", "parameters_context": count_parameters(model), "best_epoch": best_state["epoch"], "validation": result, "test_evaluated": False}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (args.out / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
