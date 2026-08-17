#!/usr/bin/env python3
"""Train the fixed RAD-TCN design on the driver-disjoint UAH split."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from rad_tcn import RADTCN, count_parameters


LABELS = ["NORMAL", "AGGRESSIVE", "DROWSY"]
ROAD_TO_ID = {"SECONDARY": 0, "MOTORWAY": 1}
TRAIN_DRIVER_TO_ID = {f"D{i}": i - 1 for i in range(1, 5)}


class UAHWindows(Dataset):
    """Globally standardized windows with train-only sensor augmentation."""

    # Offset jitter is larger on axes shown by the audit to shift across phones.
    OFFSET_STD = torch.tensor([0.04, 0.05, 0.12, 0.22, 0.12, 0.10, 0.22, 0.12])

    def __init__(self, path: Path, augment: bool):
        data = np.load(path, allow_pickle=False)
        self.x = np.clip(data["X"], -12.0, 12.0).astype(np.float32)
        self.y = data["y"].astype(np.int64)
        self.driver_names = data["driver_id"]
        self.trip = data["trip_id"]
        self.road_names = data["road"]
        self.road = np.asarray([ROAD_TO_ID[value] for value in self.road_names], dtype=np.int64)
        self.driver = np.asarray([TRAIN_DRIVER_TO_ID.get(value, -1) for value in self.driver_names], dtype=np.int64)
        self.start = data["window_start_s"].astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        x = torch.from_numpy(self.x[index].copy())
        if self.augment:
            # Phone calibration/domain jitter.  It is constant over time per
            # channel, unlike the small sample-wise sensor noise below.
            gain = torch.exp(0.06 * torch.randn(x.shape[0], 1))
            offset = self.OFFSET_STD[:, None] * torch.randn(x.shape[0], 1)
            x = x * gain + offset
            x += 0.015 * torch.randn_like(x)
            shift = int(torch.randint(-10, 11, ()).item())
            x = torch.roll(x, shifts=shift, dims=-1)
            if torch.rand(()) < 0.20:
                length = int(torch.randint(5, 21, ()).item())
                begin = int(torch.randint(0, x.shape[-1] - length + 1, ()).item())
                x[:, begin:begin + length] = 0.0
        return x, int(self.y[index]), int(self.road[index]), int(self.driver[index]), index


def group_balanced_sampler(dataset: UAHWindows) -> WeightedRandomSampler:
    """Balance driver x class x road cells, not overlapping windows."""
    groups = [
        (str(dataset.driver_names[i]), int(dataset.y[i]), int(dataset.road[i]))
        for i in range(len(dataset))
    ]
    counts = Counter(groups)
    weights = torch.as_tensor([1.0 / counts[group] for group in groups], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


def confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (y, pred), 1)
    return matrix


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    matrix = confusion(y, pred)
    per_class, f1_values = [], []
    for class_id, name in enumerate(LABELS):
        true_positive = int(matrix[class_id, class_id])
        false_positive = int(matrix[:, class_id].sum() - true_positive)
        false_negative = int(matrix[class_id].sum() - true_positive)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class.append({
            "class": name,
            "support": int(matrix[class_id].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "macro_f1": float(np.mean(f1_values)),
        "confusion_matrix_rows_true": matrix.tolist(),
        "per_class": per_class,
    }


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    model_state = model.state_dict()
    for name, value in ema.state_dict().items():
        source = model_state[name].detach()
        if value.dtype.is_floating_point:
            value.mul_(decay).add_(source, alpha=1.0 - decay)
        else:
            value.copy_(source)


@torch.no_grad()
def infer(model: RADTCN, loader: DataLoader, device: torch.device):
    model.eval()
    logits, labels, indices = [], [], []
    road_correct = road_count = 0
    for x, y, road, _, index in loader:
        output = model(x.to(device), grl_scale=0.0)
        logits.append(output["logits"].cpu())
        labels.append(y)
        indices.append(index)
        road_correct += int((output["road_logits"].argmax(1).cpu() == road).sum())
        road_count += len(road)
    logits_array = torch.cat(logits).numpy()
    label_array = torch.cat(labels).numpy()
    index_array = torch.cat(indices).numpy()
    return label_array, logits_array.argmax(1), logits_array, index_array, road_correct / road_count


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max(0.05, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=3.0e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1.0e-3)
    # About a two-epoch half-life at 32 optimizer updates/epoch.  A conventional
    # 0.995 EMA lagged too far behind on this small dataset.
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--road-loss", type=float, default=0.15)
    parser.add_argument("--domain-loss", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train = UAHWindows(args.data / "train.npz", augment=True)
    validation = UAHWindows(args.data / "val.npz", augment=False)
    test = UAHWindows(args.data / "test.npz", augment=False)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, sampler=group_balanced_sampler(train),
        num_workers=0, drop_last=True,
    )
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = RADTCN(width=args.width, embedding_dim=args.embedding_dim, dropout=args.dropout).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    behaviour_criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    auxiliary_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99),
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_lambda(step, warmup_steps, total_steps),
    )

    best_f1, stale, history, start_epoch, global_step = -1.0, 0, [], 1, 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        ema.load_state_dict(state["ema_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        best_f1 = float(state["best_macro_f1"])
        stale = int(state["stale"])
        history = state["history"]
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        print(f"resumed={args.resume} next_epoch={start_epoch}", flush=True)

    print(json.dumps({
        "architecture": "RAD-TCN",
        "device": str(device),
        "parameters_train": count_parameters(model),
        "fixed_hyperparameters": vars(args),
    }, default=str), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y, road, driver, _ in train_loader:
            x, y = x.to(device), y.to(device)
            road, driver = road.to(device), driver.to(device)
            progress = global_step / max(1, total_steps - 1)
            grl_scale = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
            output = model(x, grl_scale=grl_scale)
            batch_indices = torch.arange(len(y), device=device)
            selected_expert = output["expert_logits"][batch_indices, road]
            behaviour_loss = (
                0.70 * behaviour_criterion(output["logits"], y)
                + 0.30 * behaviour_criterion(selected_expert, y)
            )
            loss = (
                behaviour_loss
                + args.road_loss * auxiliary_criterion(output["road_logits"], road)
                + args.domain_loss * auxiliary_criterion(output["driver_logits"], driver)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            update_ema(ema, model, args.ema_decay)
            running_loss += float(loss.item()) * len(y)
            global_step += 1

        labels, predictions, _, _, road_accuracy = infer(ema, validation_loader, device)
        validation_metrics = classification_metrics(labels, predictions)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / (len(train_loader) * args.batch_size),
            "lr": optimizer.param_groups[0]["lr"],
            "road_accuracy": road_accuracy,
            **validation_metrics,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={row['train_loss']:.4f} "
            f"val_acc={row['accuracy']:.4f} val_f1={row['macro_f1']:.4f} "
            f"road_acc={road_accuracy:.4f} lr={row['lr']:.2e}",
            flush=True,
        )

        improved = validation_metrics["macro_f1"] > best_f1 + args.min_delta
        stale = 0 if improved else stale + 1
        common_state = {
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_macro_f1": max(best_f1, validation_metrics["macro_f1"]),
            "stale": stale,
            "history": history,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "architecture": "RAD-TCN",
            "feature_names": np.load(args.data / "train.npz", allow_pickle=False)["feature_names"].tolist(),
        }
        save_checkpoint(args.out / "last_checkpoint.pt", common_state)
        if improved:
            best_f1 = validation_metrics["macro_f1"]
            common_state["validation_metrics"] = validation_metrics
            save_checkpoint(args.out / "best_model.pt", common_state)
        if stale >= args.patience:
            print(f"early_stop epoch={epoch}", flush=True)
            break

    best = torch.load(args.out / "best_model.pt", map_location=device, weights_only=False)
    ema.load_state_dict(best["ema_state"])
    results = {
        "architecture": "RAD-TCN",
        "parameters": count_parameters(model),
        "best_epoch": best["epoch"],
        "hyperparameters": best["config"],
        "split_contract": {"train": ["D1", "D2", "D3", "D4"], "validation": ["D5"], "test": ["D6"]},
    }
    splits = [("validation", validation, validation_loader)]
    if args.evaluate_test:
        splits.append(("test", test, test_loader))
    for split_name, dataset, loader in splits:
        labels, predictions, logits, indices, road_accuracy = infer(ema, loader, device)
        results[split_name] = {**classification_metrics(labels, predictions), "road_accuracy": road_accuracy}
        with (args.out / f"{split_name}_predictions.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["driver", "trip", "road", "window_start_s", "true", "pred", "logit_0", "logit_1", "logit_2"])
            for row_index, source_index in enumerate(indices):
                writer.writerow([
                    dataset.driver_names[source_index], dataset.trip[source_index], dataset.road_names[source_index],
                    float(dataset.start[source_index]), int(labels[row_index]), int(predictions[row_index]),
                    *logits[row_index].tolist(),
                ])
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (args.out / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
