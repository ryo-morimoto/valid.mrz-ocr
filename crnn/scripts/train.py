#!/usr/bin/env python3
"""
CRNN 学習スクリプト

CTC Loss を使用して MRZ OCR モデルを学習する。

Usage:
    python scripts/train.py
"""

import time
from pathlib import Path

import torch
import torch.nn as nn

# MKLDNN を無効化（batch size >= 4 で backward がハングする問題を回避）
# 参考: https://github.com/pytorch/pytorch/issues/91547
import torch.backends.mkldnn
torch.backends.mkldnn.enabled = False
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import MRZDataset, collate_fn, decode_output, NUM_CLASSES
from model import CRNN, get_model_info


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
    verbose: bool = True
) -> float:
    """
    1エポック分の学習

    Returns:
        平均損失
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for i, batch in enumerate(dataloader):
        # 最初のバッチは詳細ログ
        detail = (i == 0 and epoch == 1)

        if detail:
            print(f"    [batch {i}] データ取得中...", flush=True)

        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"]

        if detail:
            print(f"    [batch {i}] forward...", flush=True)

        # 順伝播
        outputs = model(images)  # (T, B, C)
        T, B, C = outputs.shape

        # CTC Loss 用の入力長（全て同じ）
        input_lengths = torch.full((B,), T, dtype=torch.long)

        if detail:
            print(f"    [batch {i}] CTC loss...", flush=True)

        # CTC Loss
        loss = criterion(outputs, labels, input_lengths, label_lengths)

        if detail:
            print(f"    [batch {i}] backward...", flush=True)

        # 逆伝播
        optimizer.zero_grad()
        loss.backward()

        if detail:
            print(f"    [batch {i}] optimizer step...", flush=True)

        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()

        # 毎バッチ進捗表示
        if verbose:
            print(f"  [E{epoch}] {i+1:3d}/{num_batches} | Loss: {loss.item():.4f}", flush=True)

    return total_loss / num_batches


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> dict:
    """
    検証

    Returns:
        {
            "cer": 文字エラー率,
            "accuracy": 完全一致率,
            "samples": サンプル予測結果
        }
    """
    # 評価モードに切り替え
    model.training = False
    total_chars = 0
    total_errors = 0
    correct = 0
    total = 0
    samples = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            texts = batch["texts"]

            # 推論
            outputs = model(images)  # (T, B, C)

            # デコード
            for i, text in enumerate(texts):
                # Greedy decode
                probs = outputs[:, i, :]
                pred_indices = probs.argmax(dim=1).cpu().tolist()
                pred_text = decode_output(pred_indices)

                # CER 計算
                errors = sum(1 for a, b in zip(text, pred_text) if a != b)
                errors += abs(len(text) - len(pred_text))
                total_chars += len(text)
                total_errors += errors

                # 完全一致
                if text == pred_text:
                    correct += 1
                total += 1

                # サンプル保存
                if len(samples) < 5:
                    samples.append({
                        "gt": text,
                        "pred": pred_text,
                        "match": text == pred_text
                    })

    # 学習モードに戻す
    model.training = True

    cer = (total_errors / total_chars) * 100 if total_chars > 0 else 0
    accuracy = (correct / total) * 100 if total > 0 else 0

    return {
        "cer": cer,
        "accuracy": accuracy,
        "samples": samples
    }


def main():
    print("[init] 設定読み込み...", flush=True)
    # 設定
    data_dir = Path(__file__).parent.parent / "data"
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ハイパーパラメータ
    batch_size = 32
    epochs = 50
    lr = 1e-3
    max_width = 280  # MRZ 44文字に対してシーケンス長70で十分

    # デバイス
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] デバイス: {device}", flush=True)

    # データセット
    print("[init] train データ読み込み...", flush=True)
    train_dataset = MRZDataset(data_dir / "train", max_width=max_width, augment=True)
    print("[init] val データ読み込み...", flush=True)
    val_dataset = MRZDataset(data_dir / "val", max_width=max_width, augment=False)

    print("[init] DataLoader 作成...", flush=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"[init] Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)

    # モデル
    print("[init] モデル作成...", flush=True)
    model = CRNN(num_classes=NUM_CLASSES).to(device)
    info = get_model_info(model)
    print(f"[init] モデルサイズ: {info['size_mb']:.2f} MB", flush=True)
    print(f"[init] パラメータ数: {info['total_params']:,}", flush=True)

    # 損失関数・オプティマイザ
    print("[init] オプティマイザ作成...", flush=True)
    criterion = nn.CTCLoss(blank=NUM_CLASSES - 1, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print("[init] 準備完了", flush=True)

    # 学習ループ
    print("\n" + "=" * 60, flush=True)
    print("学習開始", flush=True)
    print("=" * 60, flush=True)

    best_cer = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # 学習
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # 検証
        val_result = validate(model, val_loader, device)

        # スケジューラ更新
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # ログ
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"CER: {val_result['cer']:.2f}% | "
              f"Acc: {val_result['accuracy']:.1f}% | "
              f"Time: {epoch_time:.1f}s")

        # ベストモデル保存
        if val_result["cer"] < best_cer:
            best_cer = val_result["cer"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cer": best_cer,
                "accuracy": val_result["accuracy"]
            }, checkpoint_dir / "best.pth")
            print(f"  → Best model saved (CER: {best_cer:.2f}%)")

        # サンプル表示（10エポックごと）
        if epoch % 10 == 0:
            print("\n  サンプル予測:")
            for s in val_result["samples"][:3]:
                match = "✅" if s["match"] else "❌"
                print(f"    GT:   {s['gt']}")
                print(f"    Pred: {s['pred']} {match}")
            print()

    total_time = time.time() - start_time

    # 最終結果
    print("\n" + "=" * 60)
    print("学習完了")
    print("=" * 60)
    print(f"総学習時間: {total_time:.1f}秒 ({total_time/60:.1f}分)")
    print(f"ベスト CER: {best_cer:.2f}%")
    print(f"チェックポイント: {checkpoint_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
