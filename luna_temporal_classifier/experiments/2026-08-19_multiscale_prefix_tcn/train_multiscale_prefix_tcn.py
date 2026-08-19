#!/usr/bin/env python3
"""Train the fixed MultiScalePrefixTCN on D1-D4 and validate only on D5."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import InceptionTemporalNet
from multiscale_prefix_tcn import MultiScalePrefixTCN, count_parameters
from train import metrics
from train_context_tcn import extract_features


LABELS = ["NORMAL", "AGGRESSIVE", "DROWSY"]
EMA_ALPHAS = (0.25, 0.0625, 0.015625)
ARCHITECTURE_CONTRACT = {
    "name": "MultiScalePrefixTCN",
    "encoder": "frozen InceptionTemporalNet(width=48, dropout=0.15)",
    "feature_dim": 128,
    "local_context": 25,
    "local_width": 64,
    "local_kernel": 3,
    "local_dilations": [1, 2, 4, 8],
    "local_dropout": 0.20,
    "global_ema_alphas": list(EMA_ALPHAS),
    "global_summary": "3 EMA + cumulative mean + cumulative population std",
    "global_hidden": [128, 64],
    "global_dropout": 0.25,
    "current_hidden": 32,
    "age_dim": 4,
    "fusion_hidden": 96,
    "fusion_dropout": 0.25,
    "causal": True,
    "test_evaluated": False,
}


class PrefixDataset(Dataset):
    """Local contexts and strictly causal O(1) prefix statistics."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        trips: np.ndarray,
        starts: np.ndarray,
        context: int,
        max_trip_len: int,
    ):
        self.features = features.astype(np.float32)
        self.labels_source = labels.astype(np.int64)
        self.trips_source = np.asarray(trips)
        self.starts_source = starts.astype(np.float32)
        self.records: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        self.sample_labels: list[int] = []
        self.sample_trips: list[str] = []
        denominator = math.log1p(max_trip_len)

        for trip in np.unique(self.trips_source):
            ordered = np.where(self.trips_source == trip)[0]
            ordered = ordered[np.argsort(self.starts_source[ordered])]
            prefix_mean = np.zeros(features.shape[1], dtype=np.float64)
            prefix_m2 = np.zeros(features.shape[1], dtype=np.float64)
            ema = [None, None, None]
            for position, target in enumerate(ordered):
                current = self.features[target].astype(np.float64)
                if position == 0:
                    prefix_mean = current.copy()
                    ema = [current.copy() for _ in EMA_ALPHAS]
                else:
                    delta = current - prefix_mean
                    prefix_mean += delta / (position + 1)
                    prefix_m2 += delta * (current - prefix_mean)
                    ema = [
                        alpha * current + (1.0 - alpha) * previous
                        for alpha, previous in zip(EMA_ALPHAS, ema)
                    ]
                prefix_std = np.sqrt(prefix_m2 / (position + 1))
                global_prefix = np.concatenate((*ema, prefix_mean, prefix_std)).astype(np.float32)

                begin = max(0, position - context + 1)
                source = ordered[begin:position + 1]
                if len(source) < context:
                    source = np.concatenate((np.repeat(source[:1], context - len(source)), source))
                age = np.asarray([
                    min(position / 24.0, 1.0),
                    math.log1p(position) / denominator,
                    1.0 / (position + 1.0),
                    float(position >= 24),
                ], dtype=np.float32)
                self.records.append((int(target), source, global_prefix, age))
                self.sample_labels.append(int(self.labels_source[target]))
                self.sample_trips.append(str(trip))
        self.sample_labels = np.asarray(self.sample_labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        target, source, global_prefix, age = self.records[index]
        local = torch.from_numpy(self.features[source].T.copy())
        current = torch.from_numpy(self.features[target].copy())
        return (
            local,
            torch.from_numpy(global_prefix.copy()),
            current,
            torch.from_numpy(age.copy()),
            int(self.labels_source[target]),
            index,
        )


def balanced_trip_class_sampler(dataset: PrefixDataset) -> WeightedRandomSampler:
    trips_by_class: dict[int, set[str]] = {class_id: set() for class_id in range(3)}
    trip_windows = Counter(dataset.sample_trips)
    for label, trip in zip(dataset.sample_labels, dataset.sample_trips):
        trips_by_class[int(label)].add(trip)
    weights = [
        1.0 / (len(trips_by_class[int(label)]) * trip_windows[trip])
        for label, trip in zip(dataset.sample_labels, dataset.sample_trips)
    ]
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset), replacement=True,
    )


def max_training_trip_len(trips: np.ndarray) -> int:
    return max(int(np.sum(trips == trip)) for trip in np.unique(trips))


def schedule_factor(epoch_zero_based: int, epochs: int, warmup_epochs: int = 3) -> float:
    if epoch_zero_based < warmup_epochs:
        return (epoch_zero_based + 1) / warmup_epochs
    progress = (epoch_zero_based - warmup_epochs) / max(1, epochs - warmup_epochs - 1)
    eta_ratio = 0.02
    return eta_ratio + (1.0 - eta_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def atomic_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


@torch.no_grad()
def infer(model: MultiScalePrefixTCN, loader: DataLoader, device: torch.device):
    model.eval()
    labels, logits, indices = [], [], []
    for local, prefix, current, age, label, index in loader:
        output = model(
            local.to(device), prefix.to(device), current.to(device), age.to(device),
            feature_dropout=0.0,
        )
        labels.append(label)
        logits.append(output["logits"].cpu())
        indices.append(index)
    labels_array = torch.cat(labels).numpy()
    logits_tensor = torch.cat(logits)
    probabilities = logits_tensor.softmax(1).numpy()
    predictions = probabilities.argmax(1)
    indices_array = torch.cat(indices).numpy()
    return labels_array, predictions, probabilities, indices_array


def trip_accuracy_report(dataset: PrefixDataset, labels: np.ndarray, predictions: np.ndarray, indices: np.ndarray):
    totals: dict[str, list[int]] = {}
    for label, prediction, dataset_index in zip(labels, predictions, indices):
        trip = dataset.sample_trips[int(dataset_index)]
        values = totals.setdefault(trip, [0, 0])
        values[0] += int(label == prediction)
        values[1] += 1
    return [
        {"trip": trip, "correct": values[0], "windows": values[1], "accuracy": values[0] / values[1]}
        for trip, values in sorted(totals.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--feature-dropout", type=float, default=0.05)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    encoder_state = torch.load(args.encoder_checkpoint, map_location=device, weights_only=False)
    encoder = InceptionTemporalNet(width=48, dropout=0.15).to(device)
    encoder.load_state_dict(encoder_state["model_state"])
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    # Deliberately and exclusively read train/val.  D6 is forbidden this round.
    train_features, train_data = extract_features(args.data / "train.npz", encoder, device)
    validation_features, validation_data = extract_features(args.data / "val.npz", encoder, device)
    feature_mean = train_features.mean(0, keepdims=True).astype(np.float32)
    feature_std = train_features.std(0, keepdims=True).clip(1e-4).astype(np.float32)
    train_features = ((train_features - feature_mean) / feature_std).astype(np.float32)
    validation_features = ((validation_features - feature_mean) / feature_std).astype(np.float32)
    train_trip_max = max_training_trip_len(train_data["trip_id"])
    train = PrefixDataset(
        train_features, train_data["y"], train_data["trip_id"], train_data["window_start_s"],
        args.context, train_trip_max,
    )
    validation = PrefixDataset(
        validation_features, validation_data["y"], validation_data["trip_id"], validation_data["window_start_s"],
        args.context, train_trip_max,
    )
    train_loader = DataLoader(
        train, batch_size=args.batch_size, sampler=balanced_trip_class_sampler(train), num_workers=0,
    )
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MultiScalePrefixTCN(feature_dim=128).to(device)
    parameter_count = count_parameters(model)
    if parameter_count > 150_000:
        raise RuntimeError(f"parameter budget exceeded: {parameter_count}")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: schedule_factor(epoch, args.epochs, warmup_epochs=3),
    )

    best_f1 = -1.0; best_accuracy = -1.0; best_epoch = 0
    stale = 0; history: list[dict] = []; start_epoch = 1
    if args.resume is not None:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        best_f1 = float(resume["best_macro_f1"])
        best_accuracy = float(resume["best_accuracy"])
        best_epoch = int(resume["best_epoch"])
        stale = int(resume["stale"]); history = resume["history"]
        start_epoch = int(resume["epoch"]) + 1
        random.setstate(resume["python_rng_state"])
        np.random.set_state(resume["numpy_rng_state"])
        torch.set_rng_state(resume["torch_rng_state"])
        print(f"resumed={args.resume} next_epoch={start_epoch}", flush=True)

    print(json.dumps({
        "architecture": "MultiScalePrefixTCN", "device": str(device),
        "parameters": parameter_count, "seed": args.seed,
        "train_trip_max": train_trip_max, "test_evaluated": False,
    }), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); running_loss = 0.0
        for local, prefix, current, age, label, _ in train_loader:
            local, prefix = local.to(device), prefix.to(device)
            current, age, label = current.to(device), age.to(device), label.to(device)
            output = model(local, prefix, current, age, feature_dropout=args.feature_dropout)
            loss = (
                criterion(output["logits"], label)
                + 0.2 * criterion(output["local_logits"], label)
                + 0.2 * criterion(output["global_logits"], label)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step(); running_loss += float(loss.item()) * len(label)
        scheduler.step()

        labels, predictions, probabilities, indices = infer(model, validation_loader, device)
        result = metrics(labels, predictions)
        row = {
            "epoch": epoch, "train_loss": running_loss / len(train),
            "lr": optimizer.param_groups[0]["lr"], **result,
        }
        history.append(row)
        improved = result["macro_f1"] > best_f1 + 1e-12
        if improved:
            best_f1 = result["macro_f1"]; best_accuracy = result["accuracy"]
            best_epoch = epoch; stale = 0
        else:
            stale += 1
        payload = {
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "epoch": epoch,
            "best_macro_f1": best_f1, "best_accuracy": best_accuracy,
            "best_epoch": best_epoch, "stale": stale, "history": history,
            "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "feature_mean": feature_mean, "feature_std": feature_std,
            "encoder_checkpoint": str(args.encoder_checkpoint),
            "feature_names": train_data["feature_names"].tolist(),
            "architecture_contract": ARCHITECTURE_CONTRACT,
        }
        atomic_save(payload, args.out / "last_checkpoint.pt")
        if improved:
            atomic_save(payload, args.out / "best_model.pt")
        print(
            f"epoch={epoch:03d} loss={row['train_loss']:.4f} "
            f"val_acc={result['accuracy']:.4f} val_f1={result['macro_f1']:.4f} "
            f"best_epoch={best_epoch} lr={row['lr']:.2e}", flush=True,
        )
        if stale >= args.patience:
            print(f"early_stop epoch={epoch}", flush=True); break

    best_state = torch.load(args.out / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state"])
    labels, predictions, probabilities, indices = infer(model, validation_loader, device)
    result = metrics(labels, predictions)
    per_trip = trip_accuracy_report(validation, labels, predictions, indices)
    elapsed = time.monotonic() - started_at
    report = {
        "architecture": "MultiScalePrefixTCN", "seed": args.seed,
        "parameters": parameter_count, "best_epoch": int(best_state["epoch"]),
        "training_wall_seconds": elapsed, "validation": result,
        "per_trip_accuracy": per_trip, "test_evaluated": False,
        "baseline": {"accuracy": 0.714010, "macro_f1": 0.681006},
        "delta_vs_baseline": {
            "accuracy": result["accuracy"] - 0.714010,
            "macro_f1": result["macro_f1"] - 0.681006,
        },
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (args.out / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
    with (args.out / "validation_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trip", "window_start_s", "true", "pred", "prob_0", "prob_1", "prob_2"])
        for row_index, dataset_index in enumerate(indices):
            target = validation.records[int(dataset_index)][0]
            writer.writerow([
                validation.trips_source[target], float(validation.starts_source[target]),
                int(labels[row_index]), int(predictions[row_index]), *probabilities[row_index].tolist(),
            ])
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
