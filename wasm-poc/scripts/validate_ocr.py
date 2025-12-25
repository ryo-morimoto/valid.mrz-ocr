#!/usr/bin/env python3
"""
MIDV-500 を使用した Tesseract OCR 精度検証スクリプト

Ground truth の MRZ 領域をクロップし、Tesseract で認識、
正解と比較して文字精度を計測する。

Usage:
    python scripts/validate_ocr.py
"""

import json
from pathlib import Path
from dataclasses import dataclass
import subprocess
import tempfile

import cv2
import numpy as np
from rapidfuzz import fuzz


@dataclass
class ValidationResult:
    """1画像の検証結果"""
    image_path: str
    gt_mrz_line1: str
    gt_mrz_line2: str
    ocr_mrz_line1: str
    ocr_mrz_line2: str
    accuracy_line1: float
    accuracy_line2: float
    accuracy_total: float


def load_ground_truth(gt_path: Path) -> dict:
    """Ground truth JSON を読み込む"""
    with open(gt_path) as f:
        return json.load(f)


def perspective_transform(image: np.ndarray, src_quad: np.ndarray, dst_size: tuple[int, int]) -> np.ndarray:
    """
    透視変換で画像を平坦化

    Args:
        image: 入力画像
        src_quad: 元画像の4隅座標 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        dst_size: 変換後のサイズ (width, height)

    Returns:
        透視変換された画像
    """
    dst_w, dst_h = dst_size
    # 変換先の4隅（左上、右上、右下、左下）
    dst_quad = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1]
    ], dtype=np.float32)

    src_quad = np.array(src_quad, dtype=np.float32)

    # 透視変換行列を計算
    matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)

    # 変換実行
    return cv2.warpPerspective(image, matrix, (dst_w, dst_h))


def extract_mrz_region(image: np.ndarray, gt: dict, padding: int = 5) -> np.ndarray | None:
    """
    Ground truth から MRZ 領域（field14 + field15）をクロップ

    Args:
        image: 入力画像
        gt: Ground truth データ
        padding: 余白ピクセル数

    Returns:
        クロップされた MRZ 領域画像
    """
    if "field14" not in gt or "field15" not in gt:
        return None

    # MRZ 2行分の bounding box を計算
    quad1 = np.array(gt["field14"]["quad"])
    quad2 = np.array(gt["field15"]["quad"])

    # 全体の bounding box
    all_points = np.vstack([quad1, quad2])
    x_min = max(0, all_points[:, 0].min() - padding)
    y_min = max(0, all_points[:, 1].min() - padding)
    x_max = min(image.shape[1], all_points[:, 0].max() + padding)
    y_max = min(image.shape[0], all_points[:, 1].max() + padding)

    return image[int(y_min):int(y_max), int(x_min):int(x_max)]


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    OCR 用の前処理

    - グレースケール変換
    - 二値化
    - ノイズ除去
    """
    # グレースケール変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # コントラスト強調
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 二値化（Otsu）
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def run_tesseract(image: np.ndarray, lang: str = "eng") -> str:
    """
    Tesseract OCR を実行

    MRZ 用の設定:
    - PSM 6: 単一のテキストブロック
    - Whitelist: MRZ で使用される文字のみ
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
        cv2.imwrite(temp_path, image)

    try:
        result = subprocess.run(
            [
                "tesseract", temp_path, "stdout",
                "-l", lang,
                "--psm", "6",  # 単一テキストブロック
                "-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
            ],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    finally:
        Path(temp_path).unlink(missing_ok=True)


def calculate_accuracy(gt: str, ocr: str) -> float:
    """
    文字列の類似度を計算（0-100%）

    RapidFuzz の ratio を使用
    """
    if not gt or not ocr:
        return 0.0
    return fuzz.ratio(gt, ocr)


def validate_single_image(
    image_path: Path,
    template_gt: dict,
    image_quad: list | None,
    template_size: tuple[int, int]
) -> ValidationResult | None:
    """
    1画像に対して OCR 検証を実行

    Args:
        image_path: 画像パス
        template_gt: テンプレートの MRZ ground truth
        image_quad: 画像内の文書4隅座標（Noneならテンプレート画像）
        template_size: テンプレート画像サイズ (width, height)
    """
    # 画像読み込み
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  ❌ 画像読み込み失敗: {image_path}")
        return None

    # 透視変換（撮影画像の場合）
    if image_quad is not None:
        try:
            image = perspective_transform(image, image_quad, template_size)
        except Exception as e:
            print(f"  ⚠️  透視変換失敗: {e}")
            return None

    # MRZ 領域クロップ
    mrz_region = extract_mrz_region(image, template_gt)
    if mrz_region is None:
        print(f"  ⚠️  MRZ 領域なし: {image_path}")
        return None

    # 前処理
    processed = preprocess_for_ocr(mrz_region)

    # OCR 実行
    ocr_result = run_tesseract(processed)
    lines = ocr_result.split("\n")

    # 結果を2行に分割
    ocr_line1 = lines[0] if len(lines) > 0 else ""
    ocr_line2 = lines[1] if len(lines) > 1 else ""

    # Ground truth
    gt_line1 = template_gt.get("field14", {}).get("value", "")
    gt_line2 = template_gt.get("field15", {}).get("value", "")

    # 精度計算
    acc1 = calculate_accuracy(gt_line1, ocr_line1)
    acc2 = calculate_accuracy(gt_line2, ocr_line2)
    acc_total = (acc1 + acc2) / 2

    return ValidationResult(
        image_path=str(image_path),
        gt_mrz_line1=gt_line1,
        gt_mrz_line2=gt_line2,
        ocr_mrz_line1=ocr_line1,
        ocr_mrz_line2=ocr_line2,
        accuracy_line1=acc1,
        accuracy_line2=acc2,
        accuracy_total=acc_total,
    )


@dataclass
class ImageData:
    """1画像のデータ"""
    image_path: Path
    template_gt: dict
    image_quad: list | None  # None = テンプレート画像
    template_size: tuple[int, int]


def find_all_images(data_dir: Path) -> list[ImageData]:
    """
    MIDV-500 データセットから全画像と対応する ground truth を取得
    """
    results = []

    for passport_dir in sorted(data_dir.iterdir()):
        if not passport_dir.is_dir():
            continue

        # メインの ground truth（テンプレート座標）
        main_gt_path = passport_dir / "ground_truth" / f"{passport_dir.name}.json"
        if not main_gt_path.exists():
            continue

        template_gt = load_ground_truth(main_gt_path)

        # 各画像フォルダを探索
        images_dir = passport_dir / "images"
        if not images_dir.exists():
            continue

        # テンプレート画像のサイズを取得
        template_path = images_dir / f"{passport_dir.name}.tif"
        if not template_path.exists():
            continue

        template_img = cv2.imread(str(template_path))
        if template_img is None:
            continue

        template_size = (template_img.shape[1], template_img.shape[0])  # (width, height)

        # テンプレート画像を追加
        results.append(ImageData(
            image_path=template_path,
            template_gt=template_gt,
            image_quad=None,
            template_size=template_size,
        ))

        # 各撮影条件のフォルダ
        for condition_dir in sorted(images_dir.iterdir()):
            if not condition_dir.is_dir():
                continue

            # 対応する ground truth フォルダ
            condition_gt_dir = passport_dir / "ground_truth" / condition_dir.name

            for img_path in sorted(condition_dir.glob("*.tif")):
                # 対応する ground truth（quad 情報）を探す
                gt_name = img_path.stem + ".json"
                gt_path = condition_gt_dir / gt_name

                if gt_path.exists():
                    with open(gt_path) as f:
                        quad_data = json.load(f)

                    # quad が存在する場合のみ追加
                    if "quad" in quad_data:
                        results.append(ImageData(
                            image_path=img_path,
                            template_gt=template_gt,
                            image_quad=quad_data["quad"],
                            template_size=template_size,
                        ))

    return results


def main():
    """メイン処理"""
    data_dir = Path(__file__).parent.parent.parent / "data" / "midv500"

    if not data_dir.exists():
        print(f"❌ データディレクトリが見つかりません: {data_dir}")
        return

    print("=" * 60)
    print("MIDV-500 Tesseract OCR 精度検証")
    print("=" * 60)
    print(f"データディレクトリ: {data_dir}")

    # 全画像を取得
    image_data_list = find_all_images(data_dir)
    print(f"検証対象画像数: {len(image_data_list)}")

    if not image_data_list:
        print("❌ 検証対象画像が見つかりません")
        return

    # 検証実行
    results: list[ValidationResult] = []

    print("\n" + "-" * 60)
    print("検証実行中...")
    print("-" * 60)

    for i, img_data in enumerate(image_data_list):
        print(f"[{i+1}/{len(image_data_list)}] {img_data.image_path.name}", end="")
        if img_data.image_quad is None:
            print(" (template)", end="")
        print()

        result = validate_single_image(
            img_data.image_path,
            img_data.template_gt,
            img_data.image_quad,
            img_data.template_size,
        )
        if result:
            results.append(result)
            print(f"  Line1: {result.accuracy_line1:.1f}% | Line2: {result.accuracy_line2:.1f}%")

    if not results:
        print("❌ 有効な検証結果がありません")
        return

    # サマリー
    print("\n" + "=" * 60)
    print("検証結果サマリー")
    print("=" * 60)

    avg_line1 = sum(r.accuracy_line1 for r in results) / len(results)
    avg_line2 = sum(r.accuracy_line2 for r in results) / len(results)
    avg_total = sum(r.accuracy_total for r in results) / len(results)

    print(f"検証画像数: {len(results)}")
    print(f"平均精度 (Line1): {avg_line1:.1f}%")
    print(f"平均精度 (Line2): {avg_line2:.1f}%")
    print(f"平均精度 (Total): {avg_total:.1f}%")

    # 精度分布
    print("\n" + "-" * 60)
    print("精度分布")
    print("-" * 60)
    ranges = [(90, 100), (80, 90), (70, 80), (60, 70), (50, 60), (0, 50)]
    for low, high in ranges:
        count = sum(1 for r in results if low <= r.accuracy_total < high)
        pct = count / len(results) * 100
        bar = "█" * int(pct / 2)
        print(f"  {low:3d}-{high:3d}%: {count:4d} ({pct:5.1f}%) {bar}")

    # 詳細結果（精度が高いもの）
    print("\n" + "-" * 60)
    print("精度が高い結果 (>= 80%)")
    print("-" * 60)

    high_accuracy = [r for r in results if r.accuracy_total >= 80]
    for r in high_accuracy[:10]:
        print(f"\n{Path(r.image_path).name}:")
        print(f"  GT1: {r.gt_mrz_line1}")
        print(f"  OCR: {r.ocr_mrz_line1}")
        print(f"  Acc: {r.accuracy_line1:.1f}%")

    print("\n" + "=" * 60)
    if avg_total >= 90:
        print("✅ VERDICT: OCR 精度は十分 (>= 90%)")
    elif avg_total >= 80:
        print("⚠️  VERDICT: OCR 精度は許容範囲 (>= 80%)")
    else:
        print("❌ VERDICT: OCR 精度が不足 (< 80%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
