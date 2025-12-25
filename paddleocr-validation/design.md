# MRZ OCR 検証設計書

## 1. 検証目的

**目的**: 既存の汎用OCRエンジンがMRZ（Machine Readable Zone）のOCR-Bフォントを実用レベルで認識できるかを判定し、カスタム学習の必要性を決定する。

**判定基準**: CER (Character Error Rate) < 1% を達成すればカスタム学習は不要。

---

## 2. 検証指標

| 指標 | 定義 | 計算式 | 合格基準 |
|------|------|--------|----------|
| **CER** (Character Error Rate) | 文字単位の誤り率 | `(挿入 + 削除 + 置換) / 正解文字数` | < 1% |
| **LER** (Line Error Rate) | 行が完全一致しなかった割合 | `不一致行数 / 総行数` | < 5% |
| **Checksum Pass Rate** | チェックディジット検証通過率 | `検証通過数 / 総サンプル数` | > 95% |

### 補足: MRZのチェックディジット

MRZには以下のフィールドにチェックディジットが含まれる（TD3形式の例）:
- パスポート番号
- 生年月日
- 有効期限
- オプションデータ
- 全体複合チェックディジット

チェックディジット検証は、OCR結果の信頼性を担保する最終ゲート。

---

## 3. テストデータ

### 3.1 MIDV-500（推奨）

| データセット | 説明 | サンプル数 | サイズ | Ground Truth |
|-------------|------|-----------|--------|--------------|
| **MIDV-500** | 50種類の身分証明書（17パスポート含む） | 15,000フレーム | ~5GB全体、パスポートのみ~500MB | ✅ あり (JSON) |

- **論文**: https://arxiv.org/abs/1807.05786
- **ダウンロード**: ftp://smartengines.com/midv-500/
- **ライセンス**: 研究利用可
- **特徴**:
  - 実写モバイル撮影データ（iPhone 5, Galaxy S3）
  - 5条件: Table, Keyboard, Hand, Partial, Clutter
  - 各ビデオから30フレーム抽出
  - MRZを含むパスポートデータのみ使用でサイズ削減

**ダウンロード方法**（quickstart.pyが自動実行）:
```python
# 並列ダウンロード（16ワーカー）でパスポートのみ取得
download_midv500_dataset(Path("data"), max_workers=16)
```

### 3.2 データ構成

| カテゴリ | ソース | 枚数 | 目的 |
|---------|--------|------|------|
| パスポート | MIDV-500 (17ドキュメント) | ~5,100枚 | 実環境精度（モバイル撮影） |

**パスポート対象国**:
- オーストリア、アゼルバイジャン、ブラジル、ドイツ（新・旧）
- ギリシャ、クロアチア、ハンガリー、リトアニア、ラトビア
- モルドバ、メキシコ、ロシア、セルビア、スロベニア
- スウェーデン、ウクライナ、アメリカ

> **注**: IDカード等も MRZ を含むものがあるが、初期検証ではパスポートのみ対象。

---

## 4. 画像条件バリエーション

モバイル前提のため、以下の条件を網羅的にテスト:

| 条件 | バリエーション | 優先度 |
|------|---------------|--------|
| **解像度** | 720p / 1080p / 4K | 高 |
| **照明** | 均一 / 影あり / 低照度 / フラッシュ | 高 |
| **角度** | 正面 / 傾き15° / 傾き30° | 高 |
| **フォーカス** | シャープ / 軽度ブラー / ピンボケ | 中 |
| **反射** | なし / 部分反射 / 強反射 | 中 |
| **汚れ/傷** | クリーン / 軽度 / 重度 | 低 |

### Augmentation パイプライン

```python
import albumentations as A

augmentation_pipeline = A.Compose([
    A.Rotate(limit=30, p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
])
```

---

## 5. 検証対象

| ツール | 説明 | 実行環境 | バックグラウンド | 備考 |
|--------|------|----------|-----------------|------|
| **PaddleOCR PP-OCRv4** | 最新の汎用OCR | Python | Baidu (45k+ stars) | 本命 |
| **Tesseract** | 定番OCR | Python | Google/コミュニティ (65k+ stars) | 比較用 |
| **EasyOCR** | 軽量OCR | Python | Jaided AI (25k+ stars) | 代替候補 |

### OSS採用基準

以下を満たすOSSのみ採用:
- バックグラウンドに企業またはFoundationが存在
- 2人以上のアクティブメンテナー
- 100人以上の活発なコミュニティ

### 検証優先順位

1. **PaddleOCR** (Python) - Baidu製、高精度、活発なメンテナンス
2. **Tesseract** (Python) - 実績豊富、MRZ用学習済みモデルあり
3. **EasyOCR** (Python) - 導入が容易、80+言語対応

---

## 6. 検証手順

### Phase 1: 環境構築 (1時間)

```bash
# Python環境
uv venv
source .venv/bin/activate
uv pip install paddleocr mrz pillow rapidfuzz tqdm
```

### Phase 2: データ準備 (5分)

```bash
# MIDV-500 データセット（パスポートのみ）を並列ダウンロード
# quickstart.py が自動実行するため手動操作不要

# データセット構造
data/
└── midv500/
    ├── 04_aut_passport/
    │   ├── ground_truth/
    │   │   ├── CA01_01.json    # Ground Truth (MRZ含む)
    │   │   └── CA01_01.tif     # テンプレート画像
    │   └── images/
    │       ├── CA01_01.jpg     # フレーム画像
    │       ├── CA01_01.json    # フレームGT
    │       └── ...             # 30フレーム × 10条件
    ├── 05_aze_passport/
    └── ... (17パスポートドキュメント)
```

### Phase 3: 検証実行 (30分)

```bash
# PaddleOCR で検証実行（MRZ抽出パイプライン付き）
uv run python quickstart.py
```

### Phase 4: 結果分析・レポート (2時間)

---

## 7. 検証スクリプト仕様

### 入力
```json
{
  "image_path": "data/midv2020/scans/001.jpg",
  "ground_truth": {
    "line1": "P<JPNYAMADA<<TARO<<<<<<<<<<<<<<<<<<<<<<<<<<<",
    "line2": "AB1234567<8JPN8501011M3012315<<<<<<<<<<<<<<02"
  }
}
```

### 出力
```json
{
  "image_path": "data/midv2020/scans/001.jpg",
  "prediction": {
    "line1": "P<JPNYAMADA<<TARO<<<<<<<<<<<<<<<<<<<<<<<<<<<",
    "line2": "AB1234567<8JPN8501011M3012315<<<<<<<<<<<<<<02"
  },
  "metrics": {
    "cer": 0.0,
    "ler": 0.0,
    "line1_match": true,
    "line2_match": true,
    "checksum_valid": true
  },
  "errors": []
}
```

### CER計算

```python
from rapidfuzz.distance import Levenshtein

def calculate_cer(prediction: str, ground_truth: str) -> float:
    """Character Error Rate を計算"""
    distance = Levenshtein.distance(prediction, ground_truth)
    return distance / len(ground_truth) if ground_truth else 0.0
```

### チェックディジット検証

```python
def validate_mrz_checksum(mrz_line2: str) -> bool:
    """MRZ TD3 Line2のチェックディジットを検証"""
    weights = [7, 3, 1]
    
    def calc_check(data: str) -> int:
        total = 0
        for i, char in enumerate(data):
            if char == '<':
                value = 0
            elif char.isdigit():
                value = int(char)
            else:
                value = ord(char) - ord('A') + 10
            total += value * weights[i % 3]
        return total % 10
    
    # パスポート番号チェック (位置0-9)
    if int(mrz_line2[9]) != calc_check(mrz_line2[0:9]):
        return False
    
    # 生年月日チェック (位置13-19)
    if int(mrz_line2[19]) != calc_check(mrz_line2[13:19]):
        return False
    
    # 有効期限チェック (位置21-27)
    if int(mrz_line2[27]) != calc_check(mrz_line2[21:27]):
        return False
    
    # 全体チェック (位置43)
    composite = mrz_line2[0:10] + mrz_line2[13:20] + mrz_line2[21:43]
    if int(mrz_line2[43]) != calc_check(composite):
        return False
    
    return True
```

---

## 8. レポートフォーマット

### サマリーレポート

```
=== MRZ OCR Validation Report ===
Date: 2024-XX-XX
Tool: @gutenye/ocr-browser v1.0.0

=== Overall Metrics ===
Total Samples: 2,000
CER (avg):     0.45%  ✅ PASS (< 1%)
LER:           3.2%   ✅ PASS (< 5%)
Checksum Rate: 96.8%  ✅ PASS (> 95%)

=== By Category ===
| Category          | Samples | CER    | LER   | Checksum |
|-------------------|---------|--------|-------|----------|
| MIDV-2020 Scans   | 400     | 0.12%  | 1.0%  | 99.5%    |
| MIDV-2020 Photos  | 400     | 0.89%  | 4.5%  | 95.2%    |
| MIDV-2019 (Hard)  | 200     | 1.45%  | 8.0%  | 89.0%    |
| Synthetic Clean   | 500     | 0.05%  | 0.2%  | 99.8%    |
| Synthetic Noisy   | 500     | 0.78%  | 5.2%  | 94.2%    |

=== Error Analysis ===
Most Common Errors:
1. 0 → O (45 cases)
2. 1 → I (23 cases)
3. < → K (12 cases)

=== Conclusion ===
✅ PASS: Generic OCR meets requirements. Custom training NOT required.
```

### 詳細エラーログ

```json
{
  "failures": [
    {
      "image": "midv2020/photos/123.jpg",
      "category": "mobile_photo",
      "ground_truth": "AB1234567<8",
      "prediction": "AB12345670B",
      "cer": 0.182,
      "error_positions": [9, 10],
      "suspected_cause": "reflection"
    }
  ]
}
```

---

## 9. 判定基準と次のアクション

| 結果 | CER | 次のアクション |
|------|-----|---------------|
| ✅ 合格 | < 1% | そのまま採用、SDK開発へ進む |
| ⚠️ 要改善 | 1-3% | 前処理改善を検討（コントラスト強調、傾き補正等） |
| ❌ 不合格 | > 3% | カスタム学習が必要 |

### 不合格時のフォールバック

1. **前処理パイプライン追加** (1週間)
   - 傾き補正
   - コントラスト正規化
   - MRZ領域クロップ最適化

2. **Tesseract + ocrb.traineddata** に切り替え (1日)

3. **カスタム学習** (2-3週間)
   - PaddleOCR fine-tuning
   - 合成データ10万枚生成

---

## 10. タイムライン

| フェーズ | 作業内容 | 所要時間 |
|---------|---------|---------|
| Phase 1 | 環境構築 | 30分 |
| Phase 2 | データ準備 | 10分 |
| Phase 3 | 検証実行 | 1時間 |
| Phase 4 | 結果分析・レポート | 30分 |
| **合計** | | **約2時間** |

---

## 11. 再現性の担保

### バージョン固定

```json
{
  "dependencies": {
    "@gutenye/ocr-browser": "^1.0.0"
  }
}
```

```toml
[project]
dependencies = [
    "mrz>=0.6.2",
    "paddleocr>=2.9.0",
    "pillow>=10.0.0",
    "rapidfuzz>=3.0.0",
    "tqdm>=4.0.0",
]
```

### シード固定

```python
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
```

### Dockerイメージ

```dockerfile
FROM python:3.11-slim
# 再現可能な環境を提供
```

---

## 12. 検証結果（2025-12-26）

### 12.1 検証環境

| 項目 | 内容 |
|------|------|
| データセット | MIDV-500 (EU パスポート 4種) |
| OCRエンジン | PaddleOCR PP-OCRv5 (lang=en) |
| サンプル数 | 20枚 (TA条件: テンプレート整列) |
| 実行環境 | Python 3.11, WSL2 (Linux) |

### 12.2 結果サマリー

#### TA条件（テンプレート整列画像）- 20サンプル

| 指標 | 結果 | 目標 | 判定 |
|------|------|------|------|
| **CER** | 1.25% | < 1% | ❌ FAIL |
| **LER** | 60% | < 5% | ❌ FAIL |
| **Checksum Rate** | 95% | > 95% | ⚠️ ボーダーライン |
| **パターン抽出成功率** | 100% | - | ✅ |

#### 5サンプルテスト（参考）

| 指標 | 結果 | 判定 |
|------|------|------|
| CER | 0.45% | ✅ PASS |
| Checksum | 100% | ✅ PASS |

### 12.3 条件別の認識傾向

| 条件コード | 説明 | MRZ検出 | 認識精度 |
|-----------|------|---------|----------|
| TA/TS | テンプレート整列 | ✅ 高 | 高（CER ~1%） |
| CA/CS | クリップ撮影 | △ 中 | 中 |
| HA/HS | 手持ち撮影 | △ 中 | 中 |
| PA/PS | 部分/傾斜撮影 | ❌ 低 | 低（検出失敗多発） |
| KA/KS | キーボード上 | △ 中 | 中 |

### 12.4 主なエラーパターン

1. **D → O 誤認識**: `P<D<<` が `P<O<<` と認識されるケースあり
2. **長い < 連続の途中切れ**: `<<<<<<<<<<C<<<<` のように余分な文字挿入
3. **斜め画像での検出失敗**: PA/PS条件ではMRZ領域自体を検出できず

### 12.5 結論

```
⚠️ OVERALL: MARGINAL - 前処理改善を検討
```

**現状の評価**:
- 整列済み画像（TA/TS条件）では CER ~1% で実用ボーダーライン
- チェックサム検証率 95% は実用可能レベル
- ただし傾斜・歪み画像では検出自体が困難

**推奨アクション**:
1. **前処理パイプライン追加**: 傾き補正、MRZ領域検出強化
2. **撮影条件の制限**: アプリ側でガイドフレーム表示、傾き制限
3. **カスタム学習の検討**: CER < 0.5% を目指す場合は fine-tuning を検討

---

## 13. 参考文献

1. MIDV-500論文: https://arxiv.org/abs/1807.05786
2. MIDV-500ダウンロード: ftp://smartengines.com/midv-500/
3. ICAO Doc 9303 (MRZ仕様): https://www.icao.int/publications/pages/publication.aspx?docnum=9303
4. PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
