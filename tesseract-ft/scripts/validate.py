#!/usr/bin/env python3
"""
ocrb.traineddata を使用した Tesseract OCR 精度検証

MIDV-500 のトレーニングデータを使って精度を計測する。

Usage:
    python scripts/validate.py
"""

import subprocess
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass

import cv2
from rapidfuzz import fuzz


@dataclass
class ValidationResult:
    """1行の検証結果"""
    image_path: str
    gt_text: str
    ocr_text: str
    accuracy: float
    time_ms: float


def run_tesseract(
    image_path: Path,
    lang: str = "ocrb",
    tessdata_dir: Path | None = None
) -> tuple[str, float]:
    """
    Tesseract OCR を実行

    Args:
        image_path: 画像パス
        lang: 言語モデル名
        tessdata_dir: traineddata ディレクトリ

    Returns:
        (認識結果テキスト, 処理時間ms)
    """
    cmd = ["tesseract", str(image_path), "stdout", "-l", lang, "--psm", "13"]

    # MRZ 文字のみに制限
    cmd.extend(["-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"])

    if tessdata_dir:
        cmd.extend(["--tessdata-dir", str(tessdata_dir)])

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = (time.perf_counter() - start) * 1000

    return result.stdout.strip(), elapsed


def calculate_accuracy(gt: str, ocr: str) -> float:
    """
    文字列の類似度を計算（0-100%）

    RapidFuzz の ratio を使用。
    完全一致で 100%、完全不一致で 0%。
    """
    if not gt or not ocr:
        return 0.0
    return fuzz.ratio(gt, ocr)


def calculate_cer(gt: str, ocr: str) -> float:
    """
    Character Error Rate (CER) を計算

    編集距離 / GT文字数 × 100
    0% が完全一致、高いほど悪い。
    """
    if not gt:
        return 100.0 if ocr else 0.0

    # Levenshtein距離を計算
    from rapidfuzz.distance import Levenshtein
    distance = Levenshtein.distance(gt, ocr)
    return (distance / len(gt)) * 100


def main():
    """メイン処理"""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    training_data_dir = project_dir / "training_data"
    models_dir = project_dir / "models"

    print("=" * 60)
    print("Tesseract OCR-B モデル精度検証")
    print("=" * 60)

    # traineddata の確認
    ocrb_path = models_dir / "ocrb.traineddata"
    if not ocrb_path.exists():
        print(f"❌ ocrb.traineddata が見つかりません: {ocrb_path}")
        return

    print(f"モデル: {ocrb_path}")
    print(f"モデルサイズ: {ocrb_path.stat().st_size / 1024 / 1024:.1f} MB")

    # トレーニングデータの一覧を取得
    tif_files = sorted(training_data_dir.glob("mrz_*.tif"))
    gt_files = sorted(training_data_dir.glob("mrz_*.gt.txt"))

    print(f"検証画像数: {len(tif_files)}")

    if not tif_files:
        print("❌ 検証画像が見つかりません")
        return

    # 検証実行
    print("\n" + "-" * 60)
    print("検証実行中...")
    print("-" * 60)

    results: list[ValidationResult] = []
    total_time = 0.0

    for i, (tif_path, gt_path) in enumerate(zip(tif_files, gt_files)):
        # GT 読み込み
        with open(gt_path) as f:
            gt_text = f.read().strip()

        # OCR 実行
        ocr_text, elapsed = run_tesseract(tif_path, "ocrb", models_dir)
        total_time += elapsed

        # 精度計算
        accuracy = calculate_accuracy(gt_text, ocr_text)

        results.append(ValidationResult(
            image_path=str(tif_path),
            gt_text=gt_text,
            ocr_text=ocr_text,
            accuracy=accuracy,
            time_ms=elapsed
        ))

        # 進捗表示
        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(tif_files)}] 処理中... (avg: {total_time/(i+1):.1f}ms/行)")

    # サマリー
    print("\n" + "=" * 60)
    print("検証結果サマリー")
    print("=" * 60)

    avg_accuracy = sum(r.accuracy for r in results) / len(results)
    avg_time = total_time / len(results)

    # CER 計算
    total_chars = sum(len(r.gt_text) for r in results)
    total_errors = sum(
        sum(1 for a, b in zip(r.gt_text, r.ocr_text) if a != b) +
        abs(len(r.gt_text) - len(r.ocr_text))
        for r in results
    )
    cer = (total_errors / total_chars) * 100 if total_chars > 0 else 0

    # 完全一致率
    exact_match = sum(1 for r in results if r.gt_text == r.ocr_text) / len(results) * 100

    print(f"検証行数: {len(results)}")
    print(f"平均類似度: {avg_accuracy:.1f}%")
    print(f"CER (Character Error Rate): {cer:.2f}%")
    print(f"完全一致率: {exact_match:.1f}%")
    print(f"平均処理時間: {avg_time:.1f}ms/行")
    print(f"総処理時間: {total_time/1000:.1f}秒")

    # 精度分布
    print("\n" + "-" * 60)
    print("精度分布")
    print("-" * 60)
    ranges = [(99, 100.1), (95, 99), (90, 95), (80, 90), (0, 80)]
    for low, high in ranges:
        count = sum(1 for r in results if low <= r.accuracy < high)
        pct = count / len(results) * 100
        bar = "█" * int(pct / 2)
        label = f"{low:3d}-{high if high <= 100 else 100:3.0f}%"
        print(f"  {label}: {count:4d} ({pct:5.1f}%) {bar}")

    # エラー分析
    print("\n" + "-" * 60)
    print("エラーサンプル（精度 < 100%）")
    print("-" * 60)

    errors = [r for r in results if r.accuracy < 100][:10]
    for r in errors:
        print(f"\n{Path(r.image_path).name}:")
        print(f"  GT:  {r.gt_text}")
        print(f"  OCR: {r.ocr_text}")
        print(f"  Acc: {r.accuracy:.1f}%")

        # 差分をハイライト
        diff = []
        for i, (a, b) in enumerate(zip(r.gt_text, r.ocr_text)):
            if a != b:
                diff.append(f"[{i}] '{a}' → '{b}'")
        if diff:
            print(f"  Diff: {', '.join(diff[:5])}")

    # 判定
    print("\n" + "=" * 60)
    if cer <= 1.0:
        print("✅ VERDICT: 精度基準達成 (CER <= 1%)")
    else:
        print(f"❌ VERDICT: 精度基準未達 (CER {cer:.2f}% > 1%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
