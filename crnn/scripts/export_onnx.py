#!/usr/bin/env python3
"""
CRNN モデルを ONNX 形式でエクスポート

Usage:
    python scripts/export_onnx.py
"""

from pathlib import Path

import torch

from model import CRNN
from dataset import NUM_CLASSES


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    input_height: int = 32,
    input_width: int = 280  # MRZ 44文字に対してシーケンス長70で十分
):
    """
    PyTorch モデルを ONNX にエクスポート

    Args:
        checkpoint_path: 学習済みチェックポイントのパス
        output_path: ONNX ファイルの出力パス
        input_height: 入力画像の高さ
        input_width: 入力画像の幅
    """
    print("=" * 50)
    print("ONNX エクスポート")
    print("=" * 50)

    # モデル読み込み
    model = CRNN(num_classes=NUM_CLASSES)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.training = False

    print(f"チェックポイント: {checkpoint_path}")
    print(f"CER: {checkpoint.get('cer', 'N/A')}%")

    # ダミー入力
    dummy_input = torch.randn(1, 1, input_height, input_width)

    # ONNX エクスポート
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 3: "width"},
            "output": {0: "time", 1: "batch"}
        }
    )

    # サイズ確認
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n出力: {output_path}")
    print(f"サイズ: {size_mb:.2f} MB")

    # 検証
    print("\n検証中...")
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX モデル検証OK")

    return output_path


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    checkpoint_path = project_dir / "checkpoints" / "best.pth"
    output_path = project_dir / "models" / "mrz_crnn.onnx"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        print(f"❌ チェックポイントが見つかりません: {checkpoint_path}")
        print("先に学習を実行してください: python scripts/train.py")
        return

    export_onnx(checkpoint_path, output_path)


if __name__ == "__main__":
    main()
