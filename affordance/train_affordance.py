from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from affordance.datasets.rgbd_affordance_dataset import RGBDAffordanceDataset
from affordance.models.roi_affordance_net import ROIAffordanceNet
from affordance.models.text_encoder import MiniLMTextEncoder
from affordance.utils.heatmap import sigmoid_dice_loss
from affordance.utils.image_ops import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standalone RGB-D affordance model.")
    parser.add_argument("--annotations", type=str, required=True, help="Path to JSONL annotation file.")
    parser.add_argument("--save-dir", type=str, default="affordance_runs/default", help="Checkpoint directory.")
    parser.add_argument("--text-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collate_batch(batch: list[dict]) -> dict:
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = torch.stack([item["target"] for item in batch], dim=0)
    texts = [item["instruction"] for item in batch]
    return {"images": images, "targets": targets, "texts": texts, "entries": [item["entry"] for item in batch]}


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = sigmoid_dice_loss(logits, targets)
    loss = bce + dice
    return loss, {"bce": float(bce.item()), "dice": float(dice.item())}


def evaluate(
    model: ROIAffordanceNet,
    text_encoder: MiniLMTextEncoder,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            text_features = text_encoder(batch["texts"], device=device)
            logits = model(images, text_features)
            loss, _ = compute_loss(logits, targets)
            total_loss += float(loss.item())
            total_batches += 1
    if total_batches == 0:
        return {"loss": 0.0}
    return {"loss": total_loss / total_batches}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    save_dir = ensure_dir(args.save_dir)

    dataset = RGBDAffordanceDataset(args.annotations, image_size=args.image_size)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    text_encoder = MiniLMTextEncoder(args.text_model, freeze=True).to(device)
    model = ROIAffordanceNet(text_dim=text_encoder.output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)

            with torch.no_grad():
                text_features = text_encoder(batch["texts"], device=device)

            logits = model(images, text_features)
            loss, metrics = compute_loss(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}", bce=f"{metrics['bce']:.4f}", dice=f"{metrics['dice']:.4f}")

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, text_encoder, val_loader, device)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_metrics["loss"]}
        history.append(record)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "model_kwargs": {"text_dim": text_encoder.output_dim},
            "text_model_name": text_encoder.model_name,
            "image_size": args.image_size,
            "history": history,
        }
        torch.save(checkpoint, save_dir / "last.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(checkpoint, save_dir / "best.pt")

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"best_val={best_val:.4f}"
        )

    with (save_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


if __name__ == "__main__":
    main()

