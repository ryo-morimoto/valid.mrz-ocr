#!/usr/bin/env python3
"""
YOLOv8-nano → ONNX エクスポートスクリプト

ONNX Runtime Web (WASM) で動作させるための設定でエクスポートする。
opset=12 は ONNX Runtime Web との互換性のため。

Usage:
    python scripts/export_yolo.py
"""

from pathlib import Path


def export_yolov8_nano():
    """
    YOLOv8-nano を ONNX 形式でエクスポート

    出力: models/yolov8n.onnx (~6MB)

    注意: この事前学習モデルは COCO 80クラス用。
    MRZ領域検出にはカスタム学習が必要。
    ここでは「WASM で動作するか」の検証用。
    """
    from ultralytics import YOLO

    output_dir = Path(__file__).parent.parent / "models"
    output_dir.mkdir(exist_ok=True)

    print("Loading YOLOv8-nano pretrained model...")
    model = YOLO("yolov8n.pt")

    print("Exporting to ONNX format...")
    # ONNX Runtime Web 互換設定
    # 320x320 に縮小して高速化（MRZ検出には十分）
    model.export(
        format="onnx",
        imgsz=320,       # 入力画像サイズ（640→320で4倍高速化）
        simplify=True,   # ONNX構造の最適化
        opset=12,        # ONNX Runtime Web対応opset
        dynamic=False,   # 固定入力サイズ（WASM向け）
    )

    # エクスポートされたファイルを models/ に移動
    exported_path = Path("yolov8n.onnx")
    if exported_path.exists():
        target_path = output_dir / "yolov8n.onnx"
        exported_path.rename(target_path)
        print(f"✅ Exported: {target_path}")
        print(f"   Size: {target_path.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print("❌ Export failed: yolov8n.onnx not found")
        return False

    return True


if __name__ == "__main__":
    export_yolov8_nano()
