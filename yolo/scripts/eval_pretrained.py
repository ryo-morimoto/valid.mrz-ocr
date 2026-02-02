#!/usr/bin/env python3
"""
YOLOv8-nano (COCO pretrained) による MRZ 検出性能評価

MRZ_Passport_dataset (Roboflow, CC BY 4.0) の ground truth BBox と
COCO pretrained YOLOv8-nano の検出結果を比較し、
MRZ 領域検出の feasibility を定量評価する。

Usage:
    # データセットダウンロード + 評価
    uv run --python 3.12 --with ultralytics,roboflow yolo/scripts/eval_pretrained.py --api-key YOUR_KEY

    # ダウンロード済みデータで評価のみ
    uv run --python 3.12 --with ultralytics yolo/scripts/eval_pretrained.py --data-dir yolo/data/MRZ_Passport_dataset
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def download_dataset(api_key: str, output_dir: Path) -> Path:
    """Roboflow から MRZ_Passport_dataset をダウンロード"""
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("my-test-tkkjy").project("mrz_passport_dataset-7rzth")
    version = project.version(1)
    dataset = version.download("yolov8", location=str(output_dir))
    return Path(dataset.location)


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """2つのバウンディングボックスの IoU を計算 (xyxy 形式)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[np.ndarray]:
    """YOLO形式のラベルを xyxy 座標に変換して読み込む"""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        # YOLO format: class x_center y_center width height (normalized)
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append(np.array([x1, y1, x2, y2]))
    return boxes


def evaluate(data_dir: Path) -> dict:
    """COCO pretrained YOLOv8-nano で MRZ 検出性能を評価"""
    from ultralytics import YOLO
    from PIL import Image

    model = YOLO("yolov8n.pt")

    # 全 split を結合して評価
    image_paths = []
    label_dirs = {}
    for split in ["train", "valid", "test"]:
        img_dir = data_dir / split / "images"
        label_dir = data_dir / split / "labels"
        if img_dir.exists() and label_dir.exists():
            paths = sorted(img_dir.glob("*.[jp][pn][g]")) + sorted(img_dir.glob("*.jpeg"))
            for p in paths:
                label_dirs[p] = label_dir
            image_paths.extend(paths)
            print(f"  {split}: {len(paths)} images")

    if not image_paths:
        print(f"Error: No images found in {data_dir}")
        sys.exit(1)

    print(f"Evaluating on: {len(image_paths)} images (all splits)")

    results_data = {
        "split": "all",
        "total_images": len(image_paths),
        "images_with_gt": 0,
        "detections": [],
        "iou_scores": [],
        "matched_classes": {},
        "no_detection_images": [],
    }

    for img_path in image_paths:
        img = Image.open(img_path)
        img_w, img_h = img.size

        # Ground truth
        label_path = label_dirs[img_path] / (img_path.stem + ".txt")
        gt_boxes = load_yolo_labels(label_path, img_w, img_h)
        if not gt_boxes:
            continue
        results_data["images_with_gt"] += 1

        # YOLOv8-nano inference (COCO)
        preds = model(img_path, verbose=False)[0]
        pred_boxes = preds.boxes

        if len(pred_boxes) == 0:
            results_data["no_detection_images"].append(img_path.name)
            results_data["iou_scores"].append(0.0)
            continue

        # 各 GT box に対して最大 IoU の予測を見つける
        best_iou_for_image = 0.0
        for gt_box in gt_boxes:
            best_iou = 0.0
            best_class = None
            for pred_box in pred_boxes:
                xyxy = pred_box.xyxy[0].cpu().numpy()
                iou = compute_iou(gt_box, xyxy)
                if iou > best_iou:
                    best_iou = iou
                    cls_id = int(pred_box.cls[0])
                    best_class = model.names[cls_id]

            if best_class:
                results_data["matched_classes"][best_class] = (
                    results_data["matched_classes"].get(best_class, 0) + 1
                )
            best_iou_for_image = max(best_iou_for_image, best_iou)

        results_data["iou_scores"].append(best_iou_for_image)

    # 集計
    ious = results_data["iou_scores"]
    report = {
        "split": split,
        "total_images": results_data["total_images"],
        "images_with_gt": results_data["images_with_gt"],
        "images_no_detection": len(results_data["no_detection_images"]),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "median_iou": float(np.median(ious)) if ious else 0.0,
        "iou_gt_0.5": sum(1 for x in ious if x >= 0.5),
        "iou_gt_0.5_rate": sum(1 for x in ious if x >= 0.5) / len(ious) if ious else 0.0,
        "iou_gt_0.3": sum(1 for x in ious if x >= 0.3),
        "iou_gt_0.3_rate": sum(1 for x in ious if x >= 0.3) / len(ious) if ious else 0.0,
        "matched_coco_classes": dict(
            sorted(results_data["matched_classes"].items(), key=lambda x: -x[1])
        ),
        "no_detection_images_sample": results_data["no_detection_images"][:10],
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate COCO pretrained YOLOv8-nano on MRZ dataset")
    parser.add_argument("--api-key", help="Roboflow API key for downloading dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Path to downloaded dataset (skip download)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo/outputs/eval_pretrained_report.json"),
        help="Output report path",
    )
    args = parser.parse_args()

    # データセット取得
    if args.data_dir and args.data_dir.exists():
        data_dir = args.data_dir
    elif args.api_key:
        data_dir = download_dataset(args.api_key, Path("yolo/data/MRZ_Passport_dataset"))
    else:
        print("Error: --api-key or --data-dir required")
        sys.exit(1)

    # 評価
    report = evaluate(data_dir)

    # レポート出力
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # コンソール出力
    print("\n" + "=" * 60)
    print("YOLOv8-nano (COCO pretrained) MRZ Detection Evaluation")
    print("=" * 60)
    print(f"Split: {report['split']}")
    print(f"Images with GT: {report['images_with_gt']}")
    print(f"No detection: {report['images_no_detection']}")
    print(f"Mean IoU: {report['mean_iou']:.4f}")
    print(f"Median IoU: {report['median_iou']:.4f}")
    print(f"IoU >= 0.5: {report['iou_gt_0.5']}/{report['images_with_gt']} ({report['iou_gt_0.5_rate']:.1%})")
    print(f"IoU >= 0.3: {report['iou_gt_0.3']}/{report['images_with_gt']} ({report['iou_gt_0.3_rate']:.1%})")
    print(f"\nCOCO classes detected near MRZ:")
    for cls, count in report["matched_coco_classes"].items():
        print(f"  {cls}: {count}")
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
