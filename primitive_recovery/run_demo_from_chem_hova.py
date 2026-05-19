from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple beaker primitive fitting demo from chem_hova_dataset.")
    parser.add_argument("--dataset-root", type=str, default="/home/dsj/FastSAM/chem_hova_dataset")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--index", type=int, default=0, help="Index within filtered beaker entries.")
    parser.add_argument("--output-dir", type=str, default="/home/dsj/FastSAM/primitive_outputs/beaker_demo")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    ann_path = dataset_root / "annotations" / args.split / f"chem_lab_{args.split}.json"
    entries = json.load(open(ann_path, "r", encoding="utf-8"))
    beaker_entries = [e for e in entries if e.get("object") == "beaker"]
    if not beaker_entries:
        raise ValueError("No beaker entries found")
    entry = beaker_entries[args.index]

    mask_path = dataset_root / "GT_gaussian" / args.split / entry["gt_path"]
    # GT heatmap is not object mask, so use the box to synthesize a coarse mask for the MVP demo.
    # Prefer existing image-size-aligned object silhouette from previews? Not guaranteed. For now create a mask file on disk.
    img_h = int(entry["height"])
    img_w = int(entry["width"])
    x1 = int(entry["bbox"][0] * img_w)
    y1 = int(entry["bbox"][1] * img_h)
    x2 = int(entry["bbox"][2] * img_w)
    y2 = int(entry["bbox"][3] * img_h)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bbox_mask_path = out_dir / "coarse_bbox_mask.png"

    import numpy as np
    import cv2

    bbox_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    bbox_mask[y1:y2, x1:x2] = 255
    cv2.imwrite(str(bbox_mask_path), bbox_mask)

    depth_path = dataset_root / "synthetic_depth_placeholder.png"
    if not depth_path.exists():
        depth = np.zeros((img_h, img_w), dtype=np.uint16)
        depth[y1:y2, x1:x2] = 1000
        cv2.imwrite(str(depth_path), depth)

    cmd = [
        args.python_bin,
        "-m",
        "primitive_recovery.fit_beaker",
        "--mask",
        str(bbox_mask_path),
        "--depth",
        str(depth_path),
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
