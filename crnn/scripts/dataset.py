#!/usr/bin/env python3
"""
MRZ CRNN 用 PyTorch Dataset

MRZ 行画像と Ground Truth テキストを読み込み、
CTC Loss 用のフォーマットに変換する。
"""

from pathlib import Path

import cv2
# OpenCV のマルチスレッドを無効化（PyTorch DataLoader と競合するため）
# 参考: https://github.com/pytorch/pytorch/issues/1838
cv2.setNumThreads(0)

import numpy as np
import torch
from torch.utils.data import Dataset


# MRZ で使用する文字セット（37文字）
# CTC blank は index 37
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank


def encode_text(text: str) -> list[int]:
    """
    テキストを数値インデックスに変換

    Args:
        text: MRZ テキスト（例: "P<CZESPECIMEN<<VZOR..."）

    Returns:
        インデックスのリスト
    """
    return [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]


def decode_output(indices: list[int]) -> str:
    """
    CTC 出力をテキストにデコード

    連続する同一インデックスと blank (37) を除去する。

    Args:
        indices: モデル出力のインデックスリスト

    Returns:
        デコードされたテキスト
    """
    result = []
    prev_idx = -1

    for idx in indices:
        # blank (37) はスキップ
        if idx == len(CHARS):
            prev_idx = idx
            continue

        # 連続する同一文字はスキップ
        if idx != prev_idx:
            if idx < len(CHARS):
                result.append(IDX_TO_CHAR[idx])

        prev_idx = idx

    return "".join(result)


class MRZDataset(Dataset):
    """
    MRZ 行画像の Dataset

    画像とラベルをロードし、学習用のテンソルに変換する。
    """

    def __init__(
        self,
        data_dir: Path,
        max_width: int = 400,
        augment: bool = False
    ):
        """
        Args:
            data_dir: データディレクトリ（train/ または val/）
            max_width: 最大画像幅（パディング用）
            augment: データ拡張を行うか
        """
        self.data_dir = Path(data_dir)
        self.max_width = max_width
        self.augment = augment

        # 画像ファイル一覧を取得
        self.image_paths = sorted(self.data_dir.glob("*.png"))

        if not self.image_paths:
            raise ValueError(f"画像が見つかりません: {data_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        1サンプルを取得

        Returns:
            {
                "image": (1, H, W) テンソル,
                "label": エンコードされたラベル,
                "label_length": ラベルの長さ,
                "text": 元のテキスト
            }
        """
        img_path = self.image_paths[idx]
        gt_path = img_path.with_suffix(".gt.txt")

        # 画像読み込み
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        # GT 読み込み
        with open(gt_path) as f:
            text = f.read().strip()

        # データ拡張
        if self.augment:
            image = self._augment(image)

        # 正規化（0-1）
        image = image.astype(np.float32) / 255.0

        # パディング（幅を max_width に統一）
        h, w = image.shape
        if w < self.max_width:
            pad_w = self.max_width - w
            image = np.pad(image, ((0, 0), (0, pad_w)), constant_values=1.0)
        elif w > self.max_width:
            image = image[:, :self.max_width]

        # テンソルに変換 (1, H, W)
        image_tensor = torch.from_numpy(image).unsqueeze(0)

        # ラベルをエンコード
        label = encode_text(text)

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "label_length": len(label),
            "text": text
        }

    def _augment(self, image: np.ndarray) -> np.ndarray:
        """
        データ拡張

        - 軽度の回転
        - ノイズ追加
        - コントラスト変動
        """
        h, w = image.shape

        # 回転（±3度）
        if np.random.random() < 0.5:
            angle = np.random.uniform(-3, 3)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(
                image, M, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255
            )

        # ガウシアンノイズ
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 5, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # コントラスト変動
        if np.random.random() < 0.3:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-10, 10)
            image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

        return image


def collate_fn(batch: list[dict]) -> dict:
    """
    バッチをまとめる関数

    CTC Loss 用に、ラベルをパディングなしで連結する。

    Returns:
        {
            "images": (B, 1, H, W) テンソル,
            "labels": 連結されたラベル (1D テンソル),
            "label_lengths": 各サンプルのラベル長,
            "texts": 元のテキストリスト
        }
    """
    images = torch.stack([item["image"] for item in batch])
    labels = torch.cat([item["label"] for item in batch])
    label_lengths = torch.tensor([item["label_length"] for item in batch])
    texts = [item["text"] for item in batch]

    return {
        "images": images,
        "labels": labels,
        "label_lengths": label_lengths,
        "texts": texts
    }
