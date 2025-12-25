# Tesseract MRZ ファインチューニング計画

## 背景

- Tesseract OCR の精度: 45.7% (Line1: 36%, Line2: 93%)
- 問題: `<` 文字を `L`, `K`, `C` に誤認識（MRZ 未学習）
- Tesseract 4/5 は CRNN + CTC アーキテクチャ（現代的）
- 要件: **採用基準 <1s、目標 <100ms**

## 戦略

Tesseract を MRZ/OCR-B でファインチューニングし、
Tesseract.js (WASM) で動作確認・速度計測。

## 成功基準

| 項目 | 採用基準 | 目標 |
|------|---------|------|
| OCR処理時間 | <1000ms | <100ms |
| モデルロード時間 | <3000ms | <1000ms |
| **文字精度** | **>=99%** | - |

**注**: 本番運用には 99% 以上の精度が必須（誤差1%以内）

## ディレクトリ構造

```
tesseract-ft/
├── PLAN.md                      # この計画ファイル
├── scripts/
│   ├── generate_training.py     # トレーニングデータ生成
│   ├── train.sh                 # ファインチューニング実行
│   └── validate.py              # 精度検証
├── training_data/               # 生成されたトレーニングデータ
│   ├── mrz_0001.tif
│   ├── mrz_0001.gt.txt
│   └── ...
├── output/                      # トレーニング出力
│   └── mrz.traineddata
└── models/                      # 最終モデル
    └── mrz.traineddata
```

## 実装ステップ

### Phase 1: MRZ トレーニングデータ生成

**ファイル**: `tesseract-ft/scripts/generate_training.py`

Tesseract のトレーニングには以下の形式が必要:
- `.tif` 画像 (行単位)
- `.gt.txt` Ground truth テキスト

1. MIDV-500 から MRZ 行画像を切り出し
2. 透視変換で正規化
3. Tesseract 形式で保存

**データ量**:
- 1,200 画像 × 2 行 = 2,400 行サンプル

### Phase 2: Tesseract ファインチューニング

**ファイル**: `tesseract-ft/scripts/train.sh`

```bash
# 1. ベースモデル取得
wget https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata

# 2. LSTM トレーニング
lstmtraining \
  --model_output output/mrz \
  --traineddata eng.traineddata \
  --train_listfile training_data/mrz.training_files.txt \
  --max_iterations 10000
```

**ファインチューニング設定**:
- ベースモデル: `eng.traineddata` (英語 LSTM)
- 追加学習: MRZ データのみ
- イテレーション: 10,000 (調整可能)

### Phase 3: 精度検証

**ファイル**: `tesseract-ft/scripts/validate.py`

Python で MIDV-500 検証:
- 文字精度 (Character Accuracy)
- 行精度 (Line Accuracy)
- 処理時間

### Phase 4: WASM 速度計測

ファインチューニング済みモデルを `wasm-poc/public/models/` にコピーし、
ブラウザで速度計測:
- モデルロード時間
- OCR 処理時間

## 判定フロー

```
ファインチューニング完了
    ↓
精度検証 (MIDV-500)
    ↓
精度 >= 99%? ─No→ 軽量 CRNN に切り替え
    ↓ Yes
WASM 速度計測
    ↓
処理時間 < 1s? ─No→ 軽量 CRNN に切り替え
    ↓ Yes
✅ 本番採用
```

**99% 未達の場合**: Tesseract は断念、軽量 CRNN を MRZ 専用学習

## ファイル一覧

| 新規ファイル | 用途 |
|-------------|------|
| `tesseract-ft/scripts/generate_training.py` | トレーニングデータ生成 |
| `tesseract-ft/scripts/train.sh` | ファインチューニング実行 |
| `tesseract-ft/scripts/validate.py` | 精度検証 |
| `tesseract-ft/models/mrz.traineddata` | ファインチューニング済みモデル |

## リスク

| リスク | 対策 |
|--------|------|
| トレーニング時間が長い | max_iterations を調整 |
| モデルサイズが大きい | 量子化、不要レイヤー削除 |
| WASM で遅い | 軽量 CRNN にフォールバック |

## 検証結果 (2024-12-26)

### ocrb.traineddata の精度

| 条件 | 完全一致率 | CER |
|------|-----------|-----|
| テンプレート画像 | 67% (4/6) | - |
| 撮影画像含む全体 | 15.4% | 41.06% |

### 主なエラーパターン

- `0` → `D` (テンプレートでも発生)
- `Z` → `I`, `2`
- `O` → `D`
- `<` → `A`, `C`, `Z`

### 判定

**❌ Tesseract + ocrb.traineddata は不採用**

- 目標精度 99% に対し、テンプレート画像でも 67% しか達成できず
- 撮影画像では CER 41% と大幅に悪化

## 次のアクション

**軽量 CRNN アプローチに切り替え**

1. MRZ 専用 CRNN モデル設計
2. MIDV-500 で学習
3. ONNX → WASM で速度検証
