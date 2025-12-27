# MRZ 専用 軽量 CRNN 実装計画

## 背景

- Tesseract + ocrb.traineddata: CER 41%、テンプレートでも67%完全一致 → 不採用
- 要件: **精度 >= 99%、速度 < 1s（目標 < 100ms）**
- 既存データ: tesseract-ft/training_data/ に 1,806 行の MRZ 画像

## CRNN アーキテクチャ

```
[行画像 H×W] → [CNN] → [BiLSTM] → [CTC] → [文字列]
   32×280        特徴抽出    シーケンス    デコード
                            モデリング
```

### 詳細設計

```
Input: (1, 32, 280) - グレースケール、高さ32固定、幅可変

CNN Backbone (特徴抽出):
  Conv2d(1, 32, 3, padding=1) → ReLU → MaxPool(2,2)    # 16×140
  Conv2d(32, 64, 3, padding=1) → ReLU → MaxPool(2,2)   # 8×70
  Conv2d(64, 128, 3, padding=1) → ReLU → MaxPool(2,1)  # 4×70
  Conv2d(128, 256, 3, padding=1) → ReLU → MaxPool(2,1) # 2×70
  Conv2d(256, 256, 2, padding=0)                        # 1×69

Reshape: (256, 69) → (69, 256) - シーケンス長69、特徴256

BiLSTM:
  BiLSTM(256, 128) → (69, 256)
  Linear(256, 37)  → (69, 37)

Output: (69, 37) - 69タイムステップ、37クラス

CTC Decode → 最大44文字
```

### クラス定義 (37クラス)

```python
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
# Index 0-25: A-Z
# Index 26-35: 0-9
# Index 36: <
# CTC blank: 37
```

## ディレクトリ構造

```
crnn/
├── PLAN.md                    # 計画ファイル
├── scripts/
│   ├── prepare_data.py        # データ準備（train/val分割）
│   ├── dataset.py             # PyTorch Dataset
│   ├── model.py               # CRNN モデル定義
│   ├── train.py               # 学習スクリプト
│   ├── export_onnx.py         # ONNX エクスポート
│   └── validate.py            # 精度検証
├── data/                      # 学習データ（シンボリックリンク）
├── checkpoints/               # 学習チェックポイント
├── models/                    # ONNX モデル
│   └── mrz_crnn.onnx
├── wasm/                      # WASM 検証用
│   ├── index.html             # ブラウザ検証ページ
│   ├── server.mjs             # COOP/COEP ヘッダー付きサーバー
│   └── package.json
└── README.md                  # 検証結果サマリー
```

## 実装ステップ

### Phase 1: データ準備

**ファイル**: `crnn/scripts/prepare_data.py`

1. tesseract-ft/training_data/ からデータをリンク
2. train/val 分割（80/20）
3. 画像の正規化（高さ32、幅を比率維持でリサイズ）

### Phase 2: モデル実装

**ファイル**: `crnn/scripts/model.py`

```python
class CRNN(nn.Module):
    def __init__(self, num_classes=38):  # 37 + CTC blank
        # CNN backbone
        # BiLSTM
        # Linear output
```

### Phase 3: 学習

**ファイル**: `crnn/scripts/train.py`

- Loss: CTCLoss
- Optimizer: AdamW
- LR: 1e-3 → 1e-5 (CosineAnnealing)
- Epochs: 50
- Batch size: 32
- Data augmentation:
  - 回転 (±3°)
  - ノイズ追加
  - コントラスト変動
  - 軽度の歪み

### Phase 4: ONNX エクスポート

**ファイル**: `crnn/scripts/export_onnx.py`

1. PyTorch → ONNX (opset 12)
2. 入力: (1, 1, 32, W) - 動的幅
3. 出力: (T, 37) - 動的タイムステップ
4. モデルサイズ目標: < 5MB

### Phase 5: WASM 速度検証

**ファイル**: `crnn/wasm/`

1. `index.html` - ブラウザで ONNX モデルをロード・推論
2. `server.mjs` - COOP/COEP ヘッダー付きローカルサーバー
3. `package.json` - 依存関係

```bash
cd crnn/wasm
npm install
node server.mjs
# → http://localhost:3000 で検証
```

計測項目:
- モデルロード時間
- 推論時間（10回平均）
- メモリ使用量

## 成功基準

| 項目 | 採用基準 | 目標 |
|------|---------|------|
| 文字精度 (CER) | <= 1% | < 0.5% |
| 行完全一致率 | >= 95% | >= 99% |
| OCR処理時間 | < 1000ms | < 100ms |
| モデルサイズ | < 10MB | < 5MB |

## 学習環境

- PyTorch 2.x
- CPU 学習可能（GPU があれば高速化）
- 推定学習時間: **CPU で 5-10分、GPU で 1-2分**
- データ量: 1,806行 × 50 epochs = 約 90,000 iterations

## ファイル一覧

| ファイル | 用途 |
|---------|------|
| `crnn/scripts/prepare_data.py` | データ準備 |
| `crnn/scripts/dataset.py` | Dataset クラス |
| `crnn/scripts/model.py` | CRNN モデル |
| `crnn/scripts/train.py` | 学習 |
| `crnn/scripts/export_onnx.py` | ONNX 出力 |
| `crnn/scripts/validate.py` | 精度検証 |
| `crnn/models/mrz_crnn.onnx` | 学習済みモデル |
| `crnn/wasm/index.html` | WASM 検証ページ |
| `crnn/wasm/server.mjs` | ローカルサーバー |
| `crnn/wasm/package.json` | npm 依存関係 |
| `crnn/README.md` | 検証結果サマリー |

## 後処理: チェックディジット補正

### MRZのチェックディジット構造

MRZ 2行目には5つのチェックディジット（CD）が存在:

| フィールド | 長さ | CD位置 | 重み |
|-----------|------|--------|------|
| パスポート番号 | 9文字 | 10文字目 | 7,3,1 繰り返し |
| 生年月日 | 6文字 | 7文字目 | 7,3,1 繰り返し |
| 有効期限 | 6文字 | 7文字目 | 7,3,1 繰り返し |
| オプション | 14文字 | 15文字目 | 7,3,1 繰り返し |
| 総合CD | 上記全フィールド | 最終文字 | 7,3,1 繰り返し |

### チェックディジット計算

```python
def calc_check_digit(data: str) -> int:
    """
    MRZ チェックディジット計算
    重み: 7, 3, 1 の繰り返し
    """
    weights = [7, 3, 1]
    total = 0

    for i, char in enumerate(data):
        if char == '<':
            value = 0
        elif char.isdigit():
            value = int(char)
        elif char.isalpha():
            value = ord(char.upper()) - ord('A') + 10
        else:
            value = 0

        total += value * weights[i % 3]

    return total % 10
```

### 補正戦略

#### 1. CD検証による誤り検出

```python
def verify_field(data: str, check_digit: str) -> bool:
    """フィールドのCD検証"""
    return calc_check_digit(data) == int(check_digit)
```

検証失敗時 → 誤りがあることは分かるが、どこかは不明

#### 2. 信頼度ベース補正

OCR出力の信頼度を活用:

```python
def correct_with_confidence(ocr_result, confidences, check_digit):
    """
    信頼度が低い文字から優先的に補正を試行
    """
    if verify_field(ocr_result, check_digit):
        return ocr_result, "verified"

    # 信頼度昇順でソート（低い順）
    low_conf_indices = sorted(
        range(len(ocr_result)),
        key=lambda i: confidences[i]
    )

    # 類似文字への置換を試行
    SIMILAR_CHARS = {
        '0': ['O', 'D', 'Q'],
        'O': ['0', 'D', 'Q'],
        '1': ['I', 'L', '7'],
        'I': ['1', 'L'],
        '8': ['B', '6'],
        'B': ['8', '6'],
        '5': ['S'],
        'S': ['5'],
    }

    for idx in low_conf_indices[:3]:
        char = ocr_result[idx]
        candidates = SIMILAR_CHARS.get(char, [])

        for candidate in candidates:
            test = list(ocr_result)
            test[idx] = candidate
            if verify_field(''.join(test), check_digit):
                return ''.join(test), "corrected"

    return ocr_result, "uncertain"
```

### 限界と注意点

1. **CDが誤読された場合**: データ側かCD側か判別不能
   - 確率的にはデータ側のエラーが約9倍多い（9文字 vs 1文字）
   - 総合CDとの整合性チェックで検出可能な場合あり

2. **複数エラーの場合**: 補正困難
   - CER 0.78% では2文字以上エラーは約4%
   - この場合は「uncertain」として扱う

3. **推奨**: モデル精度向上が本質的解決
   - データ拡張（ノイズ、回転、ブラー）
   - アンサンブル学習

### 期待される効果

| 現状 | 補正後期待値 |
|------|------------|
| CER 0.78% | - |
| 行精度 92.5% | 95-97% |

**注意**: 99%達成にはモデル精度自体の向上が必要

## 次のアクション

1. ディレクトリ作成、データ準備
2. Dataset, Model 実装
3. 学習実行
4. 精度検証（目標: CER <= 1%）
5. ONNX エクスポート、WASM 速度検証
6. Go/No-Go 判定
