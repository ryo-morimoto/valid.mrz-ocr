#!/usr/bin/env python3
"""
CRNN 学習用データ準備スクリプト

tesseract-ft/training_data/ のデータを train/val に分割し、
CRNN 学習用に整形する。

Usage:
    python scripts/prepare_data.py
"""

import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def resize_to_height(image: np.ndarray, target_height: int = 32) -> np.ndarray:
    """
    画像を指定の高さにリサイズ（アスペクト比維持）

    Args:
        image: 入力画像
        target_height: 目標の高さ

    Returns:
        リサイズされた画像
    """
    h, w = image.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)


def prepare_dataset(
    src_dir: Path,
    dst_dir: Path,
    train_ratio: float = 0.8,
    target_height: int = 32,
    seed: int = 42
):
    """
    データセットを準備

    tesseract-ft/training_data/ から画像と GT を読み込み、
    train/val に分割して保存する。

    Args:
        src_dir: ソースディレクトリ（tesseract-ft/training_data/）
        dst_dir: 出力ディレクトリ（crnn/data/）
        train_ratio: 学習データの割合
        target_height: 画像の高さ
        seed: 乱数シード
    """
    random.seed(seed)

    # 全ファイルを取得
    tif_files = sorted(src_dir.glob("mrz_*.tif"))
    print(f"検出ファイル数: {len(tif_files)}")

    if not tif_files:
        print("❌ ファイルが見つかりません")
        return

    # シャッフル
    indices = list(range(len(tif_files)))
    random.shuffle(indices)

    # 分割
    n_train = int(len(indices) * train_ratio)
    train_indices = set(indices[:n_train])
    val_indices = set(indices[n_train:])

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    # ディレクトリ作成
    train_dir = dst_dir / "train"
    val_dir = dst_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # データ処理
    stats = {"train": 0, "val": 0, "skipped": 0}

    for i, tif_path in enumerate(tif_files):
        gt_path = tif_path.with_suffix(".gt.txt")

        if not gt_path.exists():
            stats["skipped"] += 1
            continue

        # GT 読み込み
        with open(gt_path) as f:
            gt_text = f.read().strip()

        # 画像読み込み
        image = cv2.imread(str(tif_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            stats["skipped"] += 1
            continue

        # リサイズ
        image = resize_to_height(image, target_height)

        # 出力先を決定
        if i in train_indices:
            out_dir = train_dir
            stats["train"] += 1
        else:
            out_dir = val_dir
            stats["val"] += 1

        # 保存
        base_name = tif_path.stem
        cv2.imwrite(str(out_dir / f"{base_name}.png"), image)
        shutil.copy(gt_path, out_dir / f"{base_name}.gt.txt")

    # サマリー
    print("\n" + "=" * 40)
    print("データ準備完了")
    print("=" * 40)
    print(f"Train: {stats['train']}")
    print(f"Val: {stats['val']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"出力: {dst_dir}")

    # サンプル画像のサイズを表示
    sample_images = list((dst_dir / "train").glob("*.png"))[:5]
    if sample_images:
        print("\nサンプル画像サイズ:")
        for img_path in sample_images:
            img = cv2.imread(str(img_path))
            print(f"  {img_path.name}: {img.shape}")


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent.parent
    src_dir = project_dir / "tesseract-ft" / "training_data"
    dst_dir = script_dir.parent / "data"

    print("=" * 40)
    print("CRNN データ準備")
    print("=" * 40)
    print(f"ソース: {src_dir}")
    print(f"出力: {dst_dir}")

    if not src_dir.exists():
        print(f"❌ ソースディレクトリが見つかりません: {src_dir}")
        return

    prepare_dataset(src_dir, dst_dir)


if __name__ == "__main__":
    main()
