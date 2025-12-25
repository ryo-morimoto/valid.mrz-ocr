#!/usr/bin/env python3
"""
MRZ OCR 検証クイックスタート

MIDV-500データセットとPaddleOCR (Baidu製) を使用した検証スクリプト。
MIDV-500は50種類の身分証明書の500ビデオクリップを含む研究用データセット。
PaddleOCRは45k+ starsの実績あるOSSで、PP-OCRv4モデルを使用。

データセット: https://arxiv.org/abs/1807.05786

Usage:
    uv run python quickstart.py              # デフォルト10サンプル
    uv run python quickstart.py --samples 1  # 1サンプルのみ（デバッグ用）
    uv run python quickstart.py --samples 0  # 全サンプル
"""

import argparse
import json
import subprocess
from pathlib import Path


def install_dependencies():
    """
    必要なパッケージをインストール

    pyproject.toml に定義された依存関係を uv sync でインストールする。
    """
    subprocess.check_call(["uv", "sync"])


def download_midv500_subset(output_dir: Path, max_docs: int = 3) -> Path:
    """
    MIDV-500データセットのパスポートサブセットをダウンロード

    全データセット（~5GB）ではなく、パスポートドキュメントを
    指定数だけダウンロードして検証時間を短縮する。

    Args:
        output_dir: 出力ディレクトリ
        max_docs: ダウンロードするパスポートドキュメント数（デフォルト3）

    Returns:
        データセットディレクトリのパス
    """
    import zipfile

    dataset_dir = output_dir / "midv500"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # パスポートドキュメント（EU中心）のリスト
    # 注: USAは passport card のみ（フルパスポートなし）
    passport_docs = [
        "16_deu_passport_new",   # ドイツ（EU）
        "25_grc_passport",       # ギリシャ（EU）
        "28_hun_passport",       # ハンガリー（EU）
        "11_cze_passport",       # チェコ（EU）
    ]

    # 指定数に制限
    docs_to_download = passport_docs[:max_docs]

    print(f"Downloading {len(docs_to_download)} passport documents from MIDV-500...")
    print(f"Documents: {', '.join(docs_to_download)}")

    # ZIPファイルはmidv-500/dataset/にある
    base_url = "ftp://smartengines.com/midv-500/dataset"

    def download_and_extract(doc_name: str) -> tuple[str, bool, str]:
        """単一ドキュメントをダウンロード・展開"""
        zip_path = dataset_dir / f"{doc_name}.zip"
        doc_dir = dataset_dir / doc_name
        images_dir = doc_dir / "images"

        # 既に展開済みならスキップ（imagesディレクトリにファイルがあるかチェック）
        if images_dir.exists() and any(images_dir.glob("*.tif")):
            return (doc_name, True, "cached")

        try:
            # ZIPダウンロード
            url = f"{base_url}/{doc_name}.zip"
            print(f"  Downloading {doc_name} (~100MB)...")

            from tqdm import tqdm
            import urllib.request

            # プログレスバー付きダウンロード
            with tqdm(unit="B", unit_scale=True, desc=f"    {doc_name}") as pbar:
                def reporthook(count, block_size, total_size):
                    if pbar.total is None and total_size > 0:
                        pbar.total = total_size
                    pbar.update(block_size)

                urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)

            # 展開
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dataset_dir)

            # ZIPファイル削除
            zip_path.unlink()

            return (doc_name, True, "downloaded")
        except Exception as e:
            return (doc_name, False, str(e))

    # 順次ダウンロード（プログレス表示のため）
    results = []
    for doc in docs_to_download:
        doc_name, success, msg = download_and_extract(doc)
        results.append((doc_name, success, msg))
        if success:
            print(f"  ✓ {doc_name} ({msg})")
        else:
            print(f"  ✗ {doc_name}: {msg}")

    success_count = sum(1 for _, s, _ in results if s)
    print(f"Download complete: {success_count}/{len(docs_to_download)} documents")

    return dataset_dir


def load_midv500_dataset(dataset_dir: Path) -> list[tuple[Path, str]]:
    """
    MIDV-500データセットから画像とGround Truth (MRZ) を読み込む

    MIDV-500のGround Truth構造:
    - テンプレートJSON (<doc_name>.json): フィールド値（MRZ含む）を持つ
    - フレームJSON (TA01_01.json等): 座標情報のみ（MRZテキストなし）

    MRZはテンプレートJSONのfield14とfield15に格納されている。

    Returns:
        (画像パス, MRZ文字列) のリスト
    """
    results = []

    # ドキュメントディレクトリを探索
    for doc_dir in sorted(dataset_dir.iterdir()):
        if not doc_dir.is_dir():
            continue

        doc_name = doc_dir.name

        # テンプレートJSONからMRZ Ground Truthを取得
        # 形式: ground_truth/<doc_name>.json
        template_json = doc_dir / "ground_truth" / f"{doc_name}.json"
        if not template_json.exists():
            continue

        try:
            with open(template_json) as f:
                gt_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # MRZ行を抽出（field14=Line1, field15=Line2 が一般的）
        # TD3パスポートの場合: 44文字 x 2行
        mrz_lines = []
        for field_name, field_data in gt_data.items():
            if not isinstance(field_data, dict):
                continue
            value = field_data.get("value", "")
            if not value:
                continue

            # MRZ行の特徴: 44文字、大文字英数字と<のみ
            cleaned = value.replace(" ", "").upper()
            if len(cleaned) == 44 and "<" in cleaned:
                import re
                if re.match(r"^[A-Z0-9<]+$", cleaned):
                    # Y座標でソートするため、quadの情報も取得
                    quad = field_data.get("quad", [])
                    y_coord = quad[0][1] if quad else 0
                    mrz_lines.append((y_coord, cleaned))

        if len(mrz_lines) != 2:
            # TD3パスポート以外はスキップ
            continue

        # Y座標でソート（上から下へ）
        mrz_lines.sort(key=lambda x: x[0])
        mrz_text = "\n".join(line[1] for line in mrz_lines)

        # フレーム画像を探す
        # 構造: images/<条件コード>/<フレーム>.tif
        # 条件コード: CA, CS, HA, HS, KA, KS, PA, PS, TA, TS
        images_dir = doc_dir / "images"
        if not images_dir.exists():
            continue

        # 各条件ディレクトリ内の画像を追加
        # 優先順: TA/TS（テンプレート整列）> その他
        # PA/PS等は歪みが大きく認識困難なため後回し
        priority_conditions = ["TA", "TS"]
        other_conditions = []

        for condition_dir in sorted(images_dir.iterdir()):
            if not condition_dir.is_dir():
                continue
            if condition_dir.name in priority_conditions:
                # 優先条件は先に追加
                for img_file in condition_dir.glob("*.tif"):
                    results.append((img_file, mrz_text))
            else:
                other_conditions.append(condition_dir)

        # 他の条件は後で追加
        for condition_dir in other_conditions:
            for img_file in condition_dir.glob("*.tif"):
                results.append((img_file, mrz_text))

    return results


def calculate_cer(prediction: str, ground_truth: str) -> float:
    """
    Character Error Rate を計算

    Levenshtein距離を使用して文字単位の誤り率を算出する。
    """
    from rapidfuzz.distance import Levenshtein

    if not ground_truth:
        return 0.0
    distance = Levenshtein.distance(prediction, ground_truth)
    return distance / len(ground_truth)


def is_mrz_line(text: str) -> bool:
    """
    テキストがMRZ行かどうかを判定

    MRZ行の特徴:
    - TD3: 44文字 (パスポート)
    - TD1: 30文字 (IDカード)
    - TD2: 36文字 (ビザ等)
    - 「<」を含む
    - 大文字英字、数字、「<」のみで構成
    """
    import re

    # 空白を除去
    text = text.replace(" ", "").upper()

    # 長さチェック（許容範囲を持たせる: 28-46文字）
    if not (28 <= len(text) <= 46):
        return False

    # 「<」を含むかチェック（MRZの特徴的な区切り文字）
    if "<" not in text:
        return False

    # MRZ文字のみで構成されているかチェック（英大文字、数字、<）
    if not re.match(r"^[A-Z0-9<]+$", text):
        return False

    # 「<」の割合が高すぎる場合は除外（ノイズ対策）
    filler_ratio = text.count("<") / len(text)
    if filler_ratio > 0.7:
        return False

    return True


def extract_mrz_from_ocr_result(ocr_result: list) -> str | None:
    """
    PaddleOCR結果からMRZパターンをフィルタリング

    PaddleOCR 3.x API:
    - ocr_result: List[OCRResult]
    - OCRResult['rec_texts']: List[str] - 認識テキスト
    - OCRResult['dt_polys']: List[array] - 座標ポリゴン

    第1段階: OCR結果からMRZらしい行（44文字、<を含む）を抽出する。
    """
    if not ocr_result:
        return None

    # PaddleOCR 3.x 形式: OCRResult オブジェクトのリスト
    first_result = ocr_result[0]

    # 辞書風アクセス（OCRResult は dict-like）
    rec_texts = first_result.get('rec_texts', []) if hasattr(first_result, 'get') else []
    dt_polys = first_result.get('dt_polys', []) if hasattr(first_result, 'get') else []

    if not rec_texts:
        return None

    mrz_lines = []

    for i, text in enumerate(rec_texts):
        if not text:
            continue

        # MRZ行かどうか判定
        if is_mrz_line(text):
            # Y座標（画像内の位置）も記録
            if i < len(dt_polys):
                poly = dt_polys[i]
                # ポリゴンのY座標平均
                y_coord = sum(p[1] for p in poly) / len(poly)
            else:
                y_coord = i  # フォールバック

            mrz_lines.append((y_coord, text.replace(" ", "").upper()))

    if not mrz_lines:
        return None

    # Y座標でソート（上から下へ）
    mrz_lines.sort(key=lambda x: x[0])

    # MRZ行を結合
    mrz_text = "\n".join(line[1] for line in mrz_lines)

    return mrz_text


def detect_mrz_region_opencv(image_path: str) -> tuple[int, int, int, int] | None:
    """
    OpenCVでMRZ領域を検出

    第2段階のフォールバック: 画像処理でMRZ領域（黒文字が密集した長方形）を検出する。
    MRZは通常、画像下部にあり、横長の矩形領域。
    """
    import cv2
    import numpy as np

    # 画像読み込み
    img = cv2.imread(image_path)
    if img is None:
        return None

    height, width = img.shape[:2]

    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 画像下部50%に焦点（MRZは通常下部にある）
    roi_start = int(height * 0.5)
    gray_roi = gray[roi_start:, :]

    # 二値化（黒文字を検出）
    _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # モルフォロジー処理で文字領域を結合
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # 輪郭検出
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # MRZらしい領域を探す（横長の矩形）
    mrz_candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # MRZ領域の条件:
        # - 幅が画像幅の50%以上
        # - アスペクト比が横長（幅/高さ > 5）
        # - 高さが画像高さの3-20%程度
        aspect_ratio = w / h if h > 0 else 0
        width_ratio = w / width
        height_ratio = h / (height - roi_start)

        if width_ratio > 0.5 and aspect_ratio > 5 and 0.03 < height_ratio < 0.3:
            # ROI座標を元画像座標に変換
            mrz_candidates.append((x, y + roi_start, w, h))

    if not mrz_candidates:
        return None

    # 最も下にある候補を選択（MRZは最下部にある）
    mrz_region = max(mrz_candidates, key=lambda r: r[1])

    # マージンを追加
    x, y, w, h = mrz_region
    margin_x = int(w * 0.02)
    margin_y = int(h * 0.1)

    x = max(0, x - margin_x)
    y = max(0, y - margin_y)
    w = min(width - x, w + 2 * margin_x)
    h = min(height - y, h + 2 * margin_y)

    return (x, y, w, h)


def extract_mrz_with_fallback(
    image_path: str, ocr_result: list, ocr_engine, debug: bool = False
) -> tuple[str, str]:
    """
    MRZテキストを抽出（フォールバック付き）

    第1段階: OCR結果からMRZパターンをフィルタ
    第2段階: 失敗時、OpenCVで領域検出→クロップ→再OCR

    Returns:
        (mrz_text, method): 抽出されたMRZテキストと使用した方法
    """
    import cv2

    # 第1段階: OCR結果からMRZパターン抽出
    mrz_text = extract_mrz_from_ocr_result(ocr_result)
    if mrz_text:
        return (mrz_text, "pattern_filter")

    if debug:
        print("[DEBUG] Pattern filter failed, trying OpenCV fallback...")

    # 第2段階: OpenCVで領域検出
    region = detect_mrz_region_opencv(image_path)
    if region is None:
        if debug:
            print("[DEBUG] OpenCV MRZ detection failed")
        return ("", "detection_failed")

    if debug:
        print(f"[DEBUG] OpenCV detected MRZ region: {region}")

    # MRZ領域をクロップ
    x, y, w, h = region
    img = cv2.imread(image_path)
    cropped = img[y : y + h, x : x + w]

    # 一時ファイルに保存して再OCR
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, cropped)
        cropped_result = ocr_engine.predict(tmp.name)

        # クロップ画像のOCR結果からテキスト抽出（PaddleOCR 3.x 形式）
        lines = []
        if cropped_result and len(cropped_result) > 0:
            first_result = cropped_result[0]
            rec_texts = first_result.get('rec_texts', []) if hasattr(first_result, 'get') else []
            if debug:
                print(f"[DEBUG] Cropped OCR rec_texts: {rec_texts}")
            for text in rec_texts:
                if not text:
                    continue
                # MRZ文字のみをフィルタ
                cleaned = text.replace(" ", "").upper()
                if is_mrz_line(cleaned):
                    lines.append(cleaned)

        # 一時ファイル削除
        Path(tmp.name).unlink(missing_ok=True)

    if lines:
        return ("\n".join(lines), "opencv_crop")

    return ("", "extraction_failed")


def validate_mrz_checksum(mrz_text: str) -> bool:
    """
    MRZチェックディジットを検証（TD3形式）

    パスポート番号、生年月日、有効期限の各チェックディジットを検証する。
    """
    lines = mrz_text.strip().split("\n")
    if len(lines) != 2 or len(lines[1]) != 44:
        return False

    line2 = lines[1]
    weights = [7, 3, 1]

    def calc_check(data: str) -> int:
        """重み付き合計からチェックディジットを計算"""
        total = 0
        for i, char in enumerate(data):
            if char == "<":
                value = 0
            elif char.isdigit():
                value = int(char)
            elif char.isalpha():
                value = ord(char.upper()) - ord("A") + 10
            else:
                value = 0
            total += value * weights[i % 3]
        return total % 10

    try:
        # パスポート番号チェック (位置0-9)
        if int(line2[9]) != calc_check(line2[0:9]):
            return False
        # 生年月日チェック (位置13-19)
        if int(line2[19]) != calc_check(line2[13:19]):
            return False
        # 有効期限チェック (位置21-27)
        if int(line2[27]) != calc_check(line2[21:27]):
            return False
        return True
    except (ValueError, IndexError):
        return False


def run_paddleocr_validation(images: list[tuple[Path, str]]) -> dict:
    """
    PaddleOCRで検証実行

    各画像に対してPaddleOCR (PP-OCRv4) を実行し、CER/LER/Checksum率を計算する。
    MRZ領域抽出ロジック:
    - 第1段階: OCR結果からMRZパターンをフィルタ
    - 第2段階: 失敗時、OpenCVで領域検出→クロップ→再OCR
    """
    from paddleocr import PaddleOCR
    from tqdm import tqdm

    # PaddleOCR初期化（英語モデル）
    # 新APIではuse_angle_cls, use_gpu, show_logは非対応
    ocr = PaddleOCR(lang="en")

    results = {
        "total": len(images),
        "cer_sum": 0.0,
        "line_matches": 0,
        "checksum_passes": 0,
        "errors": [],
        "extraction_methods": {"pattern_filter": 0, "opencv_crop": 0, "failed": 0},
    }

    import tempfile
    import cv2

    for img_path, gt in tqdm(images, desc="PaddleOCR + MRZ extraction"):
        temp_jpg = None  # finallyでのクリーンアップ用
        try:
            # TIFファイルはPaddleOCRがサポートしないため、一時JPGに変換
            img_path_str = str(img_path)

            if img_path_str.lower().endswith(".tif") or img_path_str.lower().endswith(".tiff"):
                img = cv2.imread(img_path_str)
                if img is None:
                    raise ValueError(f"Failed to load image: {img_path_str}")
                temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                temp_jpg = temp_file.name
                cv2.imwrite(temp_jpg, img)
                img_path_str = temp_jpg

            # PaddleOCR実行
            ocr_result = ocr.predict(img_path_str)

            # DEBUG: 最初の数枚で結果形式を詳細出力
            if len(results["errors"]) < 3:
                print(f"\n[DEBUG] Result type: {type(ocr_result)}")
                if isinstance(ocr_result, list) and len(ocr_result) > 0:
                    item = ocr_result[0]
                    print(f"[DEBUG] First item type: {type(item).__name__}")
                    # 辞書風アクセスを試行
                    if hasattr(item, 'keys'):
                        print(f"[DEBUG] keys: {list(item.keys())}")
                    if hasattr(item, '__getitem__'):
                        try:
                            print(f"[DEBUG] item['rec_texts']: {item['rec_texts']}")
                            print(f"[DEBUG] item['dt_polys'] len: {len(item['dt_polys'])}")
                        except Exception as e:
                            print(f"[DEBUG] getitem error: {e}")
                    # 属性を列挙
                    attrs = [a for a in dir(item) if not a.startswith('_')]
                    print(f"[DEBUG] attributes: {attrs[:15]}")

            # MRZ抽出（フォールバック付き）
            # 最初の数枚はデバッグ出力
            is_debug = len(results["errors"]) < 3
            prediction, method = extract_mrz_with_fallback(
                str(img_path), ocr_result, ocr, debug=is_debug
            )

            # 抽出方法を記録
            if method in ("pattern_filter", "opencv_crop"):
                results["extraction_methods"][method] += 1
            else:
                results["extraction_methods"]["failed"] += 1

            # メトリクス計算
            cer = calculate_cer(
                prediction.replace("\n", "").replace(" ", ""),
                gt.replace("\n", "").replace(" ", ""),
            )
            results["cer_sum"] += cer

            if prediction.replace("\n", "") == gt.replace("\n", ""):
                results["line_matches"] += 1

            if validate_mrz_checksum(prediction):
                results["checksum_passes"] += 1

            # 5%以上のエラーは記録
            if cer > 0.05:
                results["errors"].append(
                    {
                        "image": str(img_path),
                        "cer": cer,
                        "method": method,
                        "prediction": prediction[:50] + "...",
                        "ground_truth": gt[:50] + "...",
                    }
                )

        except Exception as e:
            results["errors"].append({"image": str(img_path), "error": str(e)})
            results["extraction_methods"]["failed"] += 1

        finally:
            # 一時ファイルのクリーンアップ
            if temp_jpg:
                Path(temp_jpg).unlink(missing_ok=True)

    # 平均計算
    if results["total"] > 0:
        results["cer_avg"] = results["cer_sum"] / results["total"]
        results["ler"] = 1 - (results["line_matches"] / results["total"])
        results["checksum_rate"] = results["checksum_passes"] / results["total"]

    return results


def print_report(results: dict, tool_name: str):
    """
    検証結果レポートを出力

    CER、LER、Checksum率、MRZ抽出方法統計、合格判定を表示する。
    """
    print("\n" + "=" * 60)
    print(f"MRZ OCR Validation Report - {tool_name}")
    print("=" * 60)

    print(f"\nTotal Samples: {results['total']}")

    # MRZ抽出方法の統計を表示
    methods = results.get("extraction_methods", {})
    if methods:
        print("\n--- MRZ Extraction Methods ---")
        print(f"  Pattern Filter: {methods.get('pattern_filter', 0)}")
        print(f"  OpenCV Crop:    {methods.get('opencv_crop', 0)}")
        print(f"  Failed:         {methods.get('failed', 0)}")

    cer = results.get("cer_avg", 0) * 100
    ler = results.get("ler", 0) * 100
    checksum = results.get("checksum_rate", 0) * 100

    cer_status = "✅ PASS" if cer < 1 else "❌ FAIL"
    ler_status = "✅ PASS" if ler < 5 else "❌ FAIL"
    checksum_status = "✅ PASS" if checksum > 95 else "❌ FAIL"

    print("\n--- Metrics ---")
    print(f"CER (avg):     {cer:.2f}%  {cer_status}")
    print(f"LER:           {ler:.2f}%  {ler_status}")
    print(f"Checksum Rate: {checksum:.2f}%  {checksum_status}")

    if results.get("errors"):
        print(f"\n--- Errors ({len(results['errors'])} samples) ---")
        for err in results["errors"][:5]:
            image_name = Path(err.get("image", "unknown")).name
            method = err.get("method", "unknown")
            error_msg = err.get("error")
            if error_msg is None:
                cer_value = err.get("cer", 0)
                error_msg = f"CER={cer_value:.2%} ({method})"
            print(f"  - {image_name}: {error_msg}")

    print("\n" + "=" * 60)

    # 判定
    if cer < 1 and ler < 5 and checksum > 95:
        print("🎉 OVERALL: PASS - Generic OCR meets requirements!")
        print("   → Custom training is NOT required.")
    elif cer < 3:
        print("⚠️  OVERALL: MARGINAL - Consider preprocessing improvements")
        print("   → Try: contrast normalization, skew correction")
    else:
        print("❌ OVERALL: FAIL - Custom training is required.")
        print("   → Proceed with PaddleOCR fine-tuning")

    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    """
    コマンドライン引数をパース

    --samples: 処理するサンプル数（0=全件、デフォルト10）
    --condition: 条件コードでフィルタ（TA,TS等）
    """
    parser = argparse.ArgumentParser(
        description="MRZ OCR validation using MIDV-500 and PaddleOCR"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples to process (0=all, default=10)",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Filter by condition code (e.g., TA, TS, PA). Comma-separated for multiple.",
    )
    return parser.parse_args()


def main():
    """
    メイン処理

    1. 依存関係インストール
    2. MIDV-500データセットダウンロード（並列処理）
    3. 画像とGround Truth読み込み
    4. PaddleOCR (PP-OCRv4) 検証
    """
    args = parse_args()

    print("=" * 60)
    print("MRZ OCR Validation Quick Start")
    print("Dataset: MIDV-500 (https://arxiv.org/abs/1807.05786)")
    print("=" * 60)

    # Step 1: 依存関係インストール
    print("\n[1/4] Installing dependencies...")
    try:
        install_dependencies()
    except Exception as e:
        print(f"Warning: Some dependencies may not have installed: {e}")

    # Step 2: データセットダウンロード（EU パスポート）
    # データディレクトリはプロジェクトルートの ../data を使用
    data_dir = Path(__file__).parent.parent / "data"
    print("\n[2/4] Downloading MIDV-500 passport subset (EU)...")
    dataset_dir = download_midv500_subset(data_dir, max_docs=4)

    # Step 3: 画像読み込み
    print("\n[3/4] Loading MRZ images with ground truth...")
    images = load_midv500_dataset(dataset_dir)
    print(f"Found {len(images)} MRZ images with ground truth")

    # 条件コードでフィルタ
    if args.condition:
        condition_codes = [c.strip().upper() for c in args.condition.split(",")]
        images = [
            (path, gt) for path, gt in images
            if any(f"/{code}" in str(path).upper() for code in condition_codes)
        ]
        print(f"Filtered to {len(images)} images with conditions: {condition_codes}")

    if not images:
        print("No images found. Check dataset structure.")
        print(f"Expected structure: {dataset_dir}/<doc_type>/ground_truth/*.json")
        return

    # Step 4: 検証実行
    print("\n[4/4] Running validation with PaddleOCR...")

    # サンプル数を制限（--samples オプションで指定）
    # 0 = 全サンプル実行
    max_samples = args.samples
    if max_samples > 0 and len(images) > max_samples:
        import random
        random.seed(42)  # 再現性のためシード固定
        images = random.sample(images, max_samples)
        print(f"Sampling {max_samples} images for quick validation")

    print(f"Running on {len(images)} samples...")
    print("(First run will download PP-OCRv4 model ~10MB)")
    results = run_paddleocr_validation(images)
    print_report(results, "PaddleOCR PP-OCRv4")


if __name__ == "__main__":
    main()
