#!/usr/bin/env python3
"""
MRZ 専用 軽量 CRNN モデル

CNN で特徴抽出 → BiLSTM でシーケンスモデリング → CTC でデコード
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    CRNN (Convolutional Recurrent Neural Network) for MRZ OCR

    アーキテクチャ:
    - CNN Backbone: 特徴抽出（高さを1に圧縮）
    - BiLSTM: シーケンスモデリング
    - Linear: 文字分類（37クラス + CTC blank）

    入力: (B, 1, 32, W) - グレースケール画像
    出力: (T, B, 38) - 各タイムステップの文字確率
    """

    def __init__(self, num_classes: int = 38, hidden_size: int = 128):
        """
        Args:
            num_classes: 出力クラス数（37文字 + CTC blank = 38）
            hidden_size: LSTM の隠れ層サイズ
        """
        super().__init__()

        # CNN Backbone
        # 入力: (B, 1, 32, W)
        # 出力: (B, 256, 1, W') where W' = W/4 - 1
        self.cnn = nn.Sequential(
            # Block 1: 32 → 16
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),  # H: 32→16, W: W→W/2

            # Block 2: 16 → 8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),  # H: 16→8, W: W/2→W/4

            # Block 3: 8 → 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d((2, 1)),  # H: 8→4, W: W/4

            # Block 4: 4 → 2
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d((2, 1)),  # H: 4→2, W: W/4

            # Block 5: 2 → 1
            nn.Conv2d(256, 256, kernel_size=(2, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),  # H: 2→1, W: W/4
        )

        # BiLSTM
        # 入力: (B, 256, W/4) → (W/4, B, 256)
        # 出力: (W/4, B, hidden_size * 2)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.1
        )

        # 出力層
        # 入力: (T, B, hidden_size * 2)
        # 出力: (T, B, num_classes)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        Args:
            x: 入力画像 (B, 1, H, W)

        Returns:
            出力 (T, B, num_classes) - CTC 用のログ確率
        """
        # CNN 特徴抽出
        # (B, 1, 32, W) → (B, 256, 1, W')
        features = self.cnn(x)

        # 形状変換: (B, C, 1, W') → (B, C, W') → (W', B, C)
        b, c, h, w = features.shape
        features = features.squeeze(2)  # (B, C, W')
        features = features.permute(2, 0, 1)  # (W', B, C)

        # BiLSTM
        # (T, B, 256) → (T, B, hidden_size * 2)
        lstm_out, _ = self.lstm(features)

        # 出力層
        # (T, B, hidden_size * 2) → (T, B, num_classes)
        output = self.fc(lstm_out)

        # CTC 用にログソフトマックス
        output = torch.log_softmax(output, dim=2)

        return output


def get_model_info(model: nn.Module) -> dict:
    """
    モデルの情報を取得

    Returns:
        パラメータ数、モデルサイズなど
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # モデルサイズ（MB）
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb": size_mb
    }


if __name__ == "__main__":
    # テスト
    model = CRNN()
    info = get_model_info(model)

    print("=" * 40)
    print("CRNN モデル情報")
    print("=" * 40)
    print(f"総パラメータ数: {info['total_params']:,}")
    print(f"学習可能パラメータ: {info['trainable_params']:,}")
    print(f"モデルサイズ: {info['size_mb']:.2f} MB")

    # 推論テスト
    x = torch.randn(1, 1, 32, 280)
    output = model(x)
    print(f"\n入力形状: {x.shape}")
    print(f"出力形状: {output.shape}")
    print(f"シーケンス長: {output.shape[0]}")
