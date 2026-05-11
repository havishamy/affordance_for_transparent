from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from affordance.utils.image_ops import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an end-to-end smoke test for the affordance pipeline.")
    parser.add_argument("--dataset-dir", type=str, default="synthetic_affordance_dataset")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--train-dir", type=str, default="affordance_runs/smoke")
    parser.add_argument("--output-dir", type=str, default="affordance_outputs/smoke")
    parser.add_argument("--fastsam-checkpoint", type=str, default="./weights/FastSAM.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_step(cmd: list[str], cwd: Path) -> None:
    print(f"[smoke] running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_first_entry(entries_file: Path) -> dict:
    with entries_file.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        raise ValueError(f"entries file is empty: {entries_file}")
    return json.loads(first_line)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = ensure_dir(repo_root / args.dataset_dir)
    train_dir = ensure_dir(repo_root / args.train_dir)
    output_dir = ensure_dir(repo_root / args.output_dir)
    entries_file = dataset_dir / "entries.jsonl"

    run_step(
        [
            sys.executable,
            "-m",
            "affordance_annotation.generate_synthetic_dataset",
            "--output-dir",
            str(dataset_dir),
            "--num-samples",
            str(args.num_samples),
            "--seed",
            str(args.seed),
        ],
        cwd=repo_root,
    )

    run_step(
        [
            sys.executable,
            "-m",
            "affordance.train_affordance",
            "--annotations",
            str(entries_file),
            "--save-dir",
            str(train_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--image-size",
            str(args.image_size),
            "--device",
            args.device,
        ],
        cwd=repo_root,
    )

    best_ckpt = train_dir / "best.pt"
    entry = load_first_entry(entries_file)

    precomputed_dir = ensure_dir(output_dir / "precomputed_roi")
    run_step(
        [
            sys.executable,
            "-m",
            "affordance.infer_affordance",
            "--checkpoint",
            str(best_ckpt),
            "--rgb",
            entry["rgb"],
            "--depth",
            entry["depth"],
            "--instruction",
            entry["instruction"],
            "--roi-mask",
            entry["roi_mask"],
            "--output-dir",
            str(precomputed_dir),
            "--device",
            args.device,
        ],
        cwd=repo_root,
    )

    fastsam_dir = ensure_dir(output_dir / "fastsam_roi")
    bbox = [str(v) for v in entry["roi_bbox"]]
    run_step(
        [
            sys.executable,
            "-m",
            "affordance.infer_affordance",
            "--checkpoint",
            str(best_ckpt),
            "--rgb",
            entry["rgb"],
            "--depth",
            entry["depth"],
            "--instruction",
            entry["instruction"],
            "--roi-bbox",
            *bbox,
            "--fastsam-checkpoint",
            args.fastsam_checkpoint,
            "--output-dir",
            str(fastsam_dir),
            "--device",
            args.device,
        ],
        cwd=repo_root,
    )

    print("[smoke] completed successfully")
    print(f"[smoke] dataset: {dataset_dir}")
    print(f"[smoke] train dir: {train_dir}")
    print(f"[smoke] outputs: {output_dir}")


if __name__ == "__main__":
    main()
