#!/usr/bin/env python3
"""Final-fit the selected Inception1D on D1-D5, then test once on D6.

Model selection used D1-D4 -> D5 and selected epoch 3.  This script therefore
uses exactly three fixed epochs on train+validation and never performs D6-based
early stopping or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from model import InceptionTemporalNet, count_parameters
from train import Windows, infer, metrics


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3,
                        help="Frozen from D5 selection; do not tune on D6")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--clip", type=float, default=12.0)
    args = parser.parse_args()
    if args.epochs != 3:
        raise ValueError("Final-fit epoch count is frozen at 3 from D5 selection")

    seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_d1_d4 = Windows(args.data / "train.npz", augment=True, clip=args.clip)
    train_d5 = Windows(args.data / "val.npz", augment=True, clip=args.clip)
    combined = ConcatDataset([train_d1_d4, train_d5])
    test = Windows(args.data / "test.npz", augment=False, clip=args.clip)
    train_loader = DataLoader(combined, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = InceptionTemporalNet(width=args.width, dropout=args.dropout).to(device)
    combined_labels = np.concatenate((train_d1_d4.y, train_d5.y))
    counts = np.bincount(combined_labels, minlength=3)
    class_weights = len(combined_labels) / (3.0 * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=0.02,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 50.0,
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / len(combined),
            "lr": optimizer.param_groups[0]["lr"],
        })
        checkpoint = {
            "architecture": "InceptionTemporalNet",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "history": history,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "selection_contract": "D1-D4 train, D5 selected architecture and epoch=3; D1-D5 final fit; D6 one-shot test",
            "class_weights": class_weights.tolist(),
            "feature_names": np.load(args.data / "train.npz", allow_pickle=False)["feature_names"].tolist(),
        }
        atomic_save(checkpoint, args.out / "last_checkpoint.pt")
        print(
            f"epoch={epoch} train_loss={history[-1]['train_loss']:.4f} "
            f"lr={history[-1]['lr']:.2e}", flush=True,
        )

    # The only D6 evaluation in this final-fit run happens after all weights and
    # hyperparameters have been frozen.
    labels, predictions, _, _ = infer(model, test_loader, device)
    results = {
        "architecture": "InceptionTemporalNet",
        "parameters": count_parameters(model),
        "fixed_hyperparameters": checkpoint["config"],
        "selection_contract": checkpoint["selection_contract"],
        "test": metrics(labels, predictions),
    }
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (args.out / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
    atomic_save(checkpoint, args.out / "final_model.pt")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
