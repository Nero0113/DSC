#!/usr/bin/env python3
"""Train and evaluate LiteTemporalNet on the fixed driver-disjoint split."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import InceptionTemporalNet, LiteTemporalNet, StridedTemporalCNN, count_parameters


LABELS = ["NORMAL", "AGGRESSIVE", "DROWSY"]


class Windows(Dataset):
    def __init__(self, npz_path: Path, augment: bool = False, clip: float = 12.0):
        d = np.load(npz_path, allow_pickle=False)
        self.x = np.clip(d["X"], -clip, clip).astype(np.float32)
        self.y = d["y"].astype(np.int64)
        self.trip = d["trip_id"]
        self.driver = d["driver_id"]
        self.road = d["road"]
        self.start = d["window_start_s"].astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i].copy())
        if self.augment:
            # Per-channel sensor gain and mild noise; no cross-trip/window mixing.
            x *= 1.0 + 0.025 * torch.randn(x.shape[0], 1)
            x += 0.008 * torch.randn_like(x)
            if torch.rand(()) < 0.15:
                length = int(torch.randint(5, 21, ()).item())
                start = int(torch.randint(0, x.shape[1] - length + 1, ()).item())
                x[:, start:start + length] = 0
        return x, int(self.y[i]), i


def confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    cm = confusion(y, pred)
    per_class = []
    f1s = []
    for i, name in enumerate(LABELS):
        tp = int(cm[i, i]); fp = int(cm[:, i].sum() - tp); fn = int(cm[i].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class.append({"class": name, "support": int(cm[i].sum()),
                          "precision": precision, "recall": recall, "f1": f1})
    return {"accuracy": float(np.trace(cm) / cm.sum()), "macro_f1": float(np.mean(f1s)),
            "confusion_matrix_rows_true": cm.tolist(), "per_class": per_class}


@torch.no_grad()
def infer(model, loader, device):
    model.eval(); logits = []; ys = []; indices = []
    for x, y, idx in loader:
        logits.append(model(x.to(device)).cpu())
        ys.append(y); indices.append(idx)
    logits = torch.cat(logits).numpy(); y = torch.cat(ys).numpy(); idx = torch.cat(indices).numpy()
    return y, logits.argmax(1), logits, idx


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--model", choices=["strided", "dstcn", "inception"], default="strided")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--clip", type=float, default=12.0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--resume", type=Path, default=None,
                    help="Resume an interrupted run from last_checkpoint.pt")
    args = ap.parse_args()
    seed_all(args.seed); args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train = Windows(args.data / "train.npz", not args.no_augment, args.clip)
    val = Windows(args.data / "val.npz", False, args.clip)
    test = Windows(args.data / "test.npz", False, args.clip)
    kwargs = dict(batch_size=args.batch_size, num_workers=0)
    train_loader = DataLoader(train, shuffle=True, **kwargs)
    val_loader = DataLoader(val, shuffle=False, **kwargs)
    test_loader = DataLoader(test, shuffle=False, **kwargs)
    model_cls = {"strided": StridedTemporalCNN, "dstcn": LiteTemporalNet,
                 "inception": InceptionTemporalNet}[args.model]
    model = model_cls(width=args.width).to(device)
    counts = np.bincount(train.y, minlength=3)
    class_weights = len(train.y) / (3.0 * counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
                                    label_smoothing=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 50)
    best = -1.0; stale = 0; history = []; start_epoch = 1
    if args.resume is not None:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        best = float(resume["best_macro_f1"]); stale = int(resume["stale"])
        history = resume["history"]; start_epoch = int(resume["epoch"]) + 1
        random.setstate(resume["python_rng_state"])
        np.random.set_state(resume["numpy_rng_state"])
        torch.set_rng_state(resume["torch_rng_state"])
        print(f"resumed_from={args.resume} next_epoch={start_epoch}", flush=True)
    print(json.dumps({"device": str(device), "parameters": count_parameters(model),
                      "class_weights": class_weights.tolist()}), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); total_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(y)
        scheduler.step()
        vy, vp, _, _ = infer(model, val_loader, device)
        vm = metrics(vy, vp)
        row = {"epoch": epoch, "train_loss": total_loss / len(train), **vm}
        history.append(row)
        print(f"epoch={epoch:03d} loss={row['train_loss']:.4f} val_acc={vm['accuracy']:.4f} val_f1={vm['macro_f1']:.4f}", flush=True)
        if vm["macro_f1"] > best + 1e-5:
            best = vm["macro_f1"]; stale = 0
            torch.save({"model_state": model.state_dict(), "width": args.width, "model": args.model,
                        "clip": args.clip, "epoch": epoch, "val_metrics": vm,
                        "feature_names": np.load(args.data / "train.npz")["feature_names"].tolist()},
                       args.out / "best_model.pt")
        else:
            stale += 1
        # Atomic-ish two-file rotation: a complete prior checkpoint remains until
        # serialization of the new one has succeeded.
        checkpoint = {"model_state": model.state_dict(),
                      "optimizer_state": optimizer.state_dict(),
                      "scheduler_state": scheduler.state_dict(),
                      "epoch": epoch, "best_macro_f1": best, "stale": stale,
                      "history": history, "python_rng_state": random.getstate(),
                      "numpy_rng_state": np.random.get_state(),
                      "torch_rng_state": torch.get_rng_state(),
                      "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
        tmp = args.out / "last_checkpoint.pt.tmp"
        torch.save(checkpoint, tmp); tmp.replace(args.out / "last_checkpoint.pt")
        if stale >= args.patience:
            print(f"early_stop epoch={epoch}", flush=True); break

    checkpoint = torch.load(args.out / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    results = {"seed": args.seed, "device": str(device), "parameters": count_parameters(model),
               "best_epoch": checkpoint["epoch"], "clip": args.clip,
               "split_contract": {"train": sorted(set(train.driver.tolist())),
                                  "validation": sorted(set(val.driver.tolist())),
                                  "test": sorted(set(test.driver.tolist()))}}
    for split, ds, loader in (("validation", val, val_loader), ("test", test, test_loader)):
        y, pred, logits, idx = infer(model, loader, device)
        results[split] = metrics(y, pred)
        with open(args.out / f"{split}_predictions.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["driver", "trip", "road", "window_start_s", "true", "pred", "logit_0", "logit_1", "logit_2"])
            for j, k in enumerate(idx):
                w.writerow([ds.driver[k], ds.trip[k], ds.road[k], float(ds.start[k]), int(y[j]), int(pred[j]), *logits[j].tolist()])
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
