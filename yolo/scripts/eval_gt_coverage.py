#!/usr/bin/env python3
"""
GT Coverage 評価: 予測 bbox が GT bbox をどれだけカバーしているかを計測。

GT coverage = intersection_area / gt_area
1.0 なら GT 全体が予測 bbox に含まれる（MRZ テキストが crop に完全に入る）。

Usage:
    uv run --python 3.12 --with ultralytics yolo/scripts/eval_gt_coverage.py \
        --model-a /tmp/yolo-kernel-output-v3/best.pt \
        --model-b /tmp/yolo-kernel-output-v5/best.pt \
        --data-dir yolo/data/MRZ_Passport_dataset
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[np.ndarray]:
    """YOLO形式のラベルを xyxy 座標に変換"""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append(np.array([x1, y1, x2, y2]))
    return boxes


def compute_gt_coverage(gt_box: np.ndarray, pred_box: np.ndarray) -> float:
    """GT bbox のうち予測 bbox にカバーされている割合を計算"""
    x1 = max(gt_box[0], pred_box[0])
    y1 = max(gt_box[1], pred_box[1])
    x2 = min(gt_box[2], pred_box[2])
    y2 = min(gt_box[3], pred_box[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    return intersection / gt_area if gt_area > 0 else 0.0


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def evaluate_model(model_path: str, data_dir: Path, label: str) -> dict:
    from ultralytics import YOLO
    from PIL import Image

    model = YOLO(model_path)

    img_dir = data_dir / "valid" / "images"
    label_dir = data_dir / "valid" / "labels"

    image_paths = sorted(img_dir.glob("*.[jp][pn][g]"))
    print(f"\n{'=' * 60}")
    print(f"Model: {label} ({model_path})")
    print(f"{'=' * 60}")

    results = []
    for img_path in image_paths:
        img = Image.open(img_path)
        img_w, img_h = img.size

        label_path = label_dir / (img_path.stem + ".txt")
        gt_boxes = load_yolo_labels(label_path, img_w, img_h)
        if not gt_boxes:
            continue

        preds = model(img_path, verbose=False)[0]
        pred_boxes = preds.boxes

        for gt_box in gt_boxes:
            best_coverage = 0.0
            best_iou = 0.0
            best_conf = 0.0
            for pred_box in pred_boxes:
                xyxy = pred_box.xyxy[0].cpu().numpy()
                iou = compute_iou(gt_box, xyxy)
                coverage = compute_gt_coverage(gt_box, xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_coverage = coverage
                    best_conf = float(pred_box.conf[0])

            results.append({
                "image": img_path.name,
                "gt_coverage": best_coverage,
                "iou": best_iou,
                "conf": best_conf,
            })

            status = "OK" if best_coverage >= 0.99 else ("PARTIAL" if best_coverage > 0 else "MISS")
            print(f"  {img_path.stem[:20]:20s}  coverage={best_coverage:.3f}  iou={best_iou:.3f}  conf={best_conf:.2f}  [{status}]")

    coverages = [r["gt_coverage"] for r in results]
    ious = [r["iou"] for r in results]
    print(f"\n  GT Coverage: mean={np.mean(coverages):.4f}  min={np.min(coverages):.4f}  >=0.99: {sum(1 for c in coverages if c >= 0.99)}/{len(coverages)}")
    print(f"  IoU:         mean={np.mean(ious):.4f}  min={np.min(ious):.4f}  >=0.5: {sum(1 for i in ious if i >= 0.5)}/{len(ious)}")

    return {
        "label": label,
        "model": model_path,
        "gt_coverage_mean": float(np.mean(coverages)),
        "gt_coverage_min": float(np.min(coverages)),
        "gt_coverage_full": sum(1 for c in coverages if c >= 0.99),
        "iou_mean": float(np.mean(ious)),
        "iou_min": float(np.min(ious)),
        "total": len(coverages),
        "details": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True, help="First model (e.g. v3)")
    parser.add_argument("--model-b", required=True, help="Second model (e.g. v5)")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("yolo/outputs/gt_coverage_report.json"))
    args = parser.parse_args()

    report_a = evaluate_model(args.model_a, args.data_dir, "v3")
    report_b = evaluate_model(args.model_b, args.data_dir, "v5")

    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'':20s}  {'v3':>12s}  {'v5':>12s}")
    print(f"  {'GT Coverage mean':20s}  {report_a['gt_coverage_mean']:12.4f}  {report_b['gt_coverage_mean']:12.4f}")
    print(f"  {'GT Coverage min':20s}  {report_a['gt_coverage_min']:12.4f}  {report_b['gt_coverage_min']:12.4f}")
    print(f"  {'GT Coverage full':20s}  {report_a['gt_coverage_full']:>8d}/{report_a['total']}  {report_b['gt_coverage_full']:>8d}/{report_b['total']}")
    print(f"  {'IoU mean':20s}  {report_a['iou_mean']:12.4f}  {report_b['iou_mean']:12.4f}")
    print(f"  {'IoU min':20s}  {report_a['iou_min']:12.4f}  {report_b['iou_min']:12.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([report_a, report_b], indent=2, ensure_ascii=False))
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
