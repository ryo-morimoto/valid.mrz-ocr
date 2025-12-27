#!/usr/bin/env python3
"""
CRNN モデル精度検証スクリプト

学習済みモデルで全データを検証し、詳細な精度レポートを生成する。

Usage:
    python scripts/validate.py
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import MRZDataset, collate_fn, decode_output, NUM_CLASSES
from model import CRNN


def validate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> dict:
    """
    モデルを検証

    Returns:
        詳細な検証結果
    """
    model.training = False
    results = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            texts = batch["texts"]

            outputs = model(images)

            for i, text in enumerate(texts):
                probs = outputs[:, i, :]
                pred_indices = probs.argmax(dim=1).cpu().tolist()
                pred_text = decode_output(pred_indices)

                # エラー分析
                errors = []
                for j, (a, b) in enumerate(zip(text, pred_text)):
                    if a != b:
                        errors.append({"pos": j, "gt": a, "pred": b})

                results.append({
                    "gt": text,
                    "pred": pred_text,
                    "match": text == pred_text,
                    "errors": errors
                })

    return results


def analyze_results(results: list) -> dict:
    """
    結果を分析

    Returns:
        分析レポート
    """
    total = len(results)
    correct = sum(1 for r in results if r["match"])

    # CER
    total_chars = sum(len(r["gt"]) for r in results)
    total_errors = sum(len(r["errors"]) + abs(len(r["gt"]) - len(r["pred"])) for r in results)
    cer = (total_errors / total_chars) * 100 if total_chars > 0 else 0

    # エラーパターン分析
    error_patterns = {}
    for r in results:
        for e in r["errors"]:
            pattern = f"{e['gt']}→{e['pred']}"
            error_patterns[pattern] = error_patterns.get(pattern, 0) + 1

    # 上位エラーパターン
    top_errors = sorted(error_patterns.items(), key=lambda x: -x[1])[:10]

    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) * 100,
        "cer": cer,
        "top_errors": top_errors
    }


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    checkpoint_path = project_dir / "checkpoints" / "best.pth"

    print("=" * 60)
    print("CRNN モデル精度検証")
    print("=" * 60)

    if not checkpoint_path.exists():
        print(f"❌ チェックポイントが見つかりません: {checkpoint_path}")
        return

    # デバイス
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデル読み込み
    model = CRNN(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"チェックポイント: {checkpoint_path}")

    # データセット（train + val）
    all_results = []

    for split in ["train", "val"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        dataset = MRZDataset(split_dir, max_width=280, augment=False)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=collate_fn
        )

        print(f"\n{split} データ検証中... ({len(dataset)} サンプル)")
        results = validate_model(model, dataloader, device)
        all_results.extend(results)

    # 分析
    analysis = analyze_results(all_results)

    # レポート
    print("\n" + "=" * 60)
    print("検証結果")
    print("=" * 60)
    print(f"総サンプル数: {analysis['total']}")
    print(f"完全一致: {analysis['correct']} ({analysis['accuracy']:.1f}%)")
    print(f"CER: {analysis['cer']:.2f}%")

    print("\n" + "-" * 40)
    print("エラーパターン Top 10")
    print("-" * 40)
    for pattern, count in analysis["top_errors"]:
        print(f"  {pattern}: {count}回")

    # エラーサンプル
    print("\n" + "-" * 40)
    print("エラーサンプル")
    print("-" * 40)
    error_samples = [r for r in all_results if not r["match"]][:5]
    for r in error_samples:
        print(f"  GT:   {r['gt']}")
        print(f"  Pred: {r['pred']}")
        print()

    # 判定
    print("=" * 60)
    if analysis["cer"] <= 1.0:
        print("✅ VERDICT: 精度基準達成 (CER <= 1%)")
    else:
        print(f"❌ VERDICT: 精度基準未達 (CER {analysis['cer']:.2f}% > 1%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
