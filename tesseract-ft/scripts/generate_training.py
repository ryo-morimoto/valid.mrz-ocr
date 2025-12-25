#!/usr/bin/env python3
"""
MIDV-500 から Tesseract トレーニングデータを生成

MRZ 行画像を切り出し、Tesseract の学習形式（.tif + .gt.txt）で保存する。

Usage:
    python scripts/generate_training.py
"""

import json
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MRZLine:
    """MRZ 1行分のデータ"""
    image: np.ndarray      # 行画像
    text: str              # Ground truth テキスト
    source_image: str      # 元画像パス
    line_number: int       # 1 or 2


def perspective_transform(
    image: np.ndarray,
    src_quad: np.ndarray,
    dst_size: tuple[int, int]
) -> np.ndarray:
    """
    透視変換で画像を平坦化

    ドキュメントの4隅座標から、正面視点の画像に変換する。
    これにより撮影角度による歪みを補正できる。

    Args:
        image: 入力画像
        src_quad: 元画像の4隅座標 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        dst_size: 変換後のサイズ (width, height)

    Returns:
        透視変換された画像
    """
    dst_w, dst_h = dst_size
    dst_quad = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1]
    ], dtype=np.float32)

    src_quad = np.array(src_quad, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)

    return cv2.warpPerspective(image, matrix, (dst_w, dst_h))


def extract_mrz_line(
    image: np.ndarray,
    quad: list,
    padding: int = 2
) -> np.ndarray:
    """
    MRZ 1行分の画像を切り出す

    quad 座標から行領域を特定し、透視変換で正規化する。
    OCR-B フォントの標準的なアスペクト比を維持。

    Args:
        image: 入力画像（ドキュメント全体）
        quad: 行の4隅座標 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        padding: 余白ピクセル数

    Returns:
        正規化された行画像
    """
    quad = np.array(quad, dtype=np.float32)

    # 行の幅と高さを計算
    width = int(np.linalg.norm(quad[1] - quad[0]))
    height = int(np.linalg.norm(quad[3] - quad[0]))

    # パディング追加
    quad_padded = quad.copy()
    # 上下左右に少し広げる
    quad_padded[0] += [-padding, -padding]  # 左上
    quad_padded[1] += [padding, -padding]   # 右上
    quad_padded[2] += [padding, padding]    # 右下
    quad_padded[3] += [-padding, padding]   # 左下

    # 透視変換で正規化
    dst_size = (width + padding * 2, height + padding * 2)
    line_image = perspective_transform(image, quad_padded, dst_size)

    return line_image


def preprocess_for_training(image: np.ndarray) -> np.ndarray:
    """
    トレーニング用の前処理

    Tesseract 学習用に画像を正規化する。
    グレースケール変換とコントラスト強調を行う。

    Args:
        image: 入力画像

    Returns:
        前処理済み画像
    """
    # グレースケール変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # コントラスト強調（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced


def load_ground_truth(gt_path: Path) -> dict:
    """Ground truth JSON を読み込む"""
    with open(gt_path) as f:
        return json.load(f)


def is_valid_mrz_line(text: str) -> bool:
    """
    MRZ 行として有効かチェック

    TD3 パスポートの MRZ は:
    - 44文字固定
    - 使用文字: A-Z, 0-9, < のみ
    """
    if len(text) != 44:
        return False

    valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
    return all(c in valid_chars for c in text)


def extract_mrz_lines_from_image(
    image_path: Path,
    template_gt: dict,
    image_quad: list | None,
    template_size: tuple[int, int]
) -> list[MRZLine]:
    """
    1画像から MRZ 2行分を抽出

    撮影画像の場合は透視変換でドキュメントを正規化してから
    MRZ 行を切り出す。

    Args:
        image_path: 画像パス
        template_gt: テンプレートの MRZ ground truth
        image_quad: 画像内のドキュメント4隅座標（None=テンプレート画像）
        template_size: テンプレート画像サイズ (width, height)

    Returns:
        MRZLine のリスト（2行分）
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    # 撮影画像の場合は透視変換でドキュメントを正規化
    if image_quad is not None:
        try:
            image = perspective_transform(image, image_quad, template_size)
        except Exception:
            return []

    results = []

    # field14 (Line 1) と field15 (Line 2) を抽出
    for field_name, line_num in [("field14", 1), ("field15", 2)]:
        if field_name not in template_gt:
            continue

        field = template_gt[field_name]
        quad = field.get("quad")
        text = field.get("value", "")

        # MRZ 形式のバリデーション
        if not is_valid_mrz_line(text):
            continue

        if not quad or not text:
            continue

        # 行画像を切り出し
        try:
            line_image = extract_mrz_line(image, quad)
            line_image = preprocess_for_training(line_image)

            results.append(MRZLine(
                image=line_image,
                text=text,
                source_image=str(image_path),
                line_number=line_num
            ))
        except Exception:
            continue

    return results


def find_all_images(data_dir: Path) -> list[tuple]:
    """
    MIDV-500 データセットから全画像情報を取得

    各画像について、パス、Ground Truth、透視変換用座標を返す。

    Returns:
        (image_path, template_gt, image_quad, template_size) のリスト
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

        # 画像フォルダを探索
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

        template_size = (template_img.shape[1], template_img.shape[0])

        # テンプレート画像を追加
        results.append((template_path, template_gt, None, template_size))

        # 各撮影条件のフォルダ
        for condition_dir in sorted(images_dir.iterdir()):
            if not condition_dir.is_dir():
                continue

            condition_gt_dir = passport_dir / "ground_truth" / condition_dir.name

            for img_path in sorted(condition_dir.glob("*.tif")):
                gt_name = img_path.stem + ".json"
                gt_path = condition_gt_dir / gt_name

                if gt_path.exists():
                    with open(gt_path) as f:
                        quad_data = json.load(f)

                    if "quad" in quad_data:
                        results.append((
                            img_path,
                            template_gt,
                            quad_data["quad"],
                            template_size
                        ))

    return results


def main():
    """
    メイン処理: MIDV-500 から Tesseract トレーニングデータを生成

    出力形式:
    - {output_dir}/mrz_{index:04d}.tif: 行画像
    - {output_dir}/mrz_{index:04d}.gt.txt: Ground truth テキスト
    - {output_dir}/mrz.training_files.txt: ファイルリスト
    """
    # パス設定
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent.parent
    data_dir = project_dir / "data" / "midv500"
    output_dir = script_dir.parent / "training_data"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Tesseract トレーニングデータ生成")
    print("=" * 60)
    print(f"データディレクトリ: {data_dir}")
    print(f"出力ディレクトリ: {output_dir}")

    if not data_dir.exists():
        print(f"❌ データディレクトリが見つかりません: {data_dir}")
        return

    # 全画像を取得
    image_list = find_all_images(data_dir)
    print(f"検出画像数: {len(image_list)}")

    if not image_list:
        print("❌ 画像が見つかりません")
        return

    # MRZ 行を抽出
    print("\n" + "-" * 60)
    print("MRZ 行画像を抽出中...")
    print("-" * 60)

    all_lines: list[MRZLine] = []

    for i, (img_path, template_gt, image_quad, template_size) in enumerate(image_list):
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(image_list)}] 処理中...")

        lines = extract_mrz_lines_from_image(
            img_path, template_gt, image_quad, template_size
        )
        all_lines.extend(lines)

    print(f"抽出行数: {len(all_lines)}")

    if not all_lines:
        print("❌ MRZ 行を抽出できませんでした")
        return

    # トレーニングデータを保存
    print("\n" + "-" * 60)
    print("トレーニングデータを保存中...")
    print("-" * 60)

    training_files = []

    for i, line in enumerate(all_lines):
        base_name = f"mrz_{i:04d}"
        tif_path = output_dir / f"{base_name}.tif"
        gt_path = output_dir / f"{base_name}.gt.txt"

        # 画像を保存（グレースケール TIFF）
        cv2.imwrite(str(tif_path), line.image)

        # Ground truth を保存
        with open(gt_path, "w") as f:
            f.write(line.text)

        training_files.append(str(tif_path.absolute()))

    # ファイルリストを保存
    list_path = output_dir / "mrz.training_files.txt"
    with open(list_path, "w") as f:
        f.write("\n".join(training_files))

    # サマリー
    print("\n" + "=" * 60)
    print("生成完了")
    print("=" * 60)
    print(f"総行数: {len(all_lines)}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"ファイルリスト: {list_path}")

    # サンプル表示
    print("\n" + "-" * 60)
    print("サンプル（最初の5行）:")
    print("-" * 60)
    for i, line in enumerate(all_lines[:5]):
        print(f"[{i}] {line.text[:40]}...")
        print(f"    Source: {Path(line.source_image).name}, Line {line.line_number}")


if __name__ == "__main__":
    main()
