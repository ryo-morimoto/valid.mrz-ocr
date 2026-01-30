# CRNN Training Iteration History

Training loop iterations for MRZ OCR CRNN model.

**Target:** CER < 1%

---

## Initial State (2026-01-25)

**Baseline Metrics:**
- CER: ~58% (pre-v5)
- Expected v5 improvement: significant

**Starting Configuration:**
- MAX_WIDTH: 960
- Training samples: 50K
- Batch size: 128
- Epochs: 100

---

## Iteration 1 (2026-01-25 19:12)
- CER: 98.35% → **0.00%** (Epoch 1 → Epoch 3)
- Accuracy: 0.0% → **100.0%**
- 変更: v7 (サンプル数15K, CPU拡張30%復活, 検証拡張無効化)
- 仮説: データ量増加とクリーン評価により汎化性能向上
- 結果: **目標達成** (CER < 1%)
- 学習時間: 16.1分
- モデル: FastCRNN (592,966パラメータ, 2.27 MB)
- 注意: 検証時GPU拡張なしのため、CER 0.00%は過大評価の可能性

---

## Iteration 2 (2026-01-30 16:19)
- CER: 96.37% → **0.00%** (Epoch 1 → Epoch 20)
- Accuracy: 0.0% → **99.9%**
- 変更: v7.2 (GPU拡張後パディングマスク修正)
- 仮説: GPU拡張(Kornia)がパディング領域にもノイズ・回転を適用し、テキスト/パディング境界で偽文字(L)を学習していた。content_widthでマスク復元すれば根本解決
- 結果: **目標達成** (CER 0.00%, Acc 99.9%)
- 学習時間: 17.5分
- モデル: FastCRNN (592,966パラメータ, 2.26 MB)
- 詳細:
  - v7.1ではEpoch 2からCER 0.00%(人工的に容易) → v7.2ではEpoch 20で初CER 0.00%(現実的な収束曲線)
  - Epoch 35でLRリスタートにより一時崩壊(CER 44.46%)、ベストモデルはEpoch 20で保存済み
  - パディングマスクにより検証の信頼性が向上

---

## Iteration 3: 分析・計画 (2026-01-30)

### ONNX推論テスト結果
- **正解率**: 986/1000 (99.0%)
- **CER**: 0.018%
- **エラー件数**: 14/1000

### エラーパターン分析

| Metric | Value | Trend |
|--------|-------|-------|
| Current CER | 0.018% | - |
| Best CER (val) | 0.00% | Epoch 20 |
| Accuracy | 99.0% (ONNX推論) | - |

#### 全14件のエラーが完全に同一パターン
- **位置**: 44文字目（最終文字）のみ
- **誤認識先**: 全て `L` (index=11)
- **誤認識元**: `U`(9件), `J`(4件), `V`(1件)

```
GT:   ...972J  → Pred: ...972L  (J→L)
GT:   ...JUPU  → Pred: ...JUPL  (U→L)
GT:   ...4G8U  → Pred: ...4G8L  (U→L)
GT:   ...HYJU  → Pred: ...HYJL  (U→L)
GT:   ...RPBU  → Pred: ...RPBL  (U→L)
GT:   ...X0UV  → Pred: ...X0UL  (V→L)
GT:   ...6HNJ  → Pred: ...6HNL  (J→L)
GT:   ...2KJ   → Pred: ...2KL   (J→L)
```

### 根本原因分析

#### 仮説: GPU拡張による最終文字の右端クリッピング

v7.2のパディングマスク修正は「パディング領域にノイズが残る」問題は解決したが、
**逆方向の問題**が残っている:

1. **GPU拡張（RandomRotation, RandomPerspective）がテキストコンテンツを右方向にシフト**
   - 回転(最大5度)やパースペクティブ歪み(0.15)により、右端の文字ピクセルがパディング領域にはみ出す
2. **パディングマスクがはみ出したピクセルを白(1.0)で上書き**
   - `images[i, :, :, content_widths[i]:] = 1.0` により、右にはみ出た部分が消去される
3. **最終文字の右半分が切り取られた状態で学習**
   - U, J, V などの文字は右端を切り取ると縦棒+短い水平線 = `L` に類似した形状になる
   - モデルは「44文字目の位置では部分的に見える文字 = L」というバイアスを学習

#### OCR-B フォントでの形状類似性

- `U` の右半分を削除 → 左の縦棒 = `L` に類似
- `J` の右半分を削除 → 上部の横棒+縦棒 = `L` に類似
- `V` の右半分を削除 → 左の斜め線 = `L` の縦棒に類似

#### なぜ44文字目だけか

- CNNのMaxPool2d(2,2)が2回適用され、幅方向は1/4に縮小
- テキストコンテンツは約560-620px幅（OCR-B 24px, 44文字, padding=4x2）
- content_widthの境界 = テキスト最終文字の右端
- GPU拡張で右方向にシフトした場合、最終文字のみが影響を受ける
- 1-43文字目は両側にテキストがあるため、拡張の影響が平均化される

#### 推論時に再現する理由

- 推論時（infer.py）ではGPU拡張なし = テキスト右端はクリーン
- しかしモデルは「テキスト/パディング境界の文字はL的に見える」と学習済み
- CNNの受容野（receptive field）がテキスト最終文字位置でパディング領域も参照
- BiGRUの逆方向パスがパディング領域からの情報で最終文字の予測に影響

---

## 改善計画 v7.3

### 方針: テキスト右端のクリッピングを防止

GPU拡張がテキスト最終文字を破壊する問題を、データ拡張パイプラインの調整で解決する。
アーキテクチャ変更は不要。

### 案1 (推奨): content_widthにマージンを追加してマスク位置を右にオフセット

- **意図**: GPU拡張でテキストが右にシフトしても、最終文字がクリップされないようにする
- **仮説**: マスク開始位置を `content_width + margin` に変更すれば、回転・パースペクティブで右にはみ出た最終文字ピクセルが保護される。パディング領域のノイズは多少残るが、テキスト最終文字の完全性が優先
- **リスク**: 低。マージン分だけパディング領域にGPU拡張ノイズが残るが、CTC decoderがblankとして無視する。v7.2以前でも動作していた範囲
- **期待効果**: 44文字目の誤認識14件が0件に → Accuracy 99.0% → 100%、CER 0.018% → 0.00%
- **変更箇所**: `crnn/notebooks/train_crnn.ipynb` - train_epoch関数のパディングマスク処理
- **変更内容**:
  ```python
  # Before (v7.2)
  if augment is not None:
      images = augment(images)
      B, C, H, W = images.shape
      for i in range(B):
          if content_widths[i] < W:
              images[i, :, :, content_widths[i]:] = 1.0

  # After (v7.3)
  if augment is not None:
      images = augment(images)
      B, C, H, W = images.shape
      MASK_MARGIN = 32  # GPU拡張による右方向シフトを吸収 (5度回転時の最大シフト ≈ 32px * tan(5°) ≈ 2.8px + perspective 15% ≈ 24px)
      for i in range(B):
          mask_start = min(content_widths[i] + MASK_MARGIN, W)
          if mask_start < W:
              images[i, :, :, mask_start:] = 1.0
  ```

### 案2: GPU拡張をテキスト領域のみに適用（crop→augment→paste back）

- **意図**: パディング領域を完全にGPU拡張から除外し、テキスト領域のみを変形
- **仮説**: テキストコンテンツだけを切り出して拡張し、パディング付きの画像に戻せば境界問題が根本的に解決
- **リスク**: 中。バッチ内でcontent_widthが異なるため、可変幅cropの実装が複雑。パフォーマンス低下の可能性
- **期待効果**: 案1と同等だが、より堅牢
- **変更箇所**: GPUAugmentation.forward() と train_epoch()
- **変更内容**: 各サンプルのcontent幅でcrop → augment → 元の位置にpaste + padding white
  ```python
  # 概要のみ（実装は複雑）
  for i in range(B):
      content = images[i:i+1, :, :, :content_widths[i]]
      # 個別にaugmentする必要があり、バッチ処理の利点が失われる
      augmented = augment(content)
      images[i, :, :, :content_widths[i]] = augmented[0]
      images[i, :, :, content_widths[i]:] = 1.0
  ```

### 案3: GPU拡張の回転・パースペクティブ強度を低減

- **意図**: テキスト右端のシフト量を減らし、クリッピングの影響を最小化
- **仮説**: RandomRotation degrees=5→2, RandomPerspective distortion_scale=0.15→0.05 で右端シフトが大幅に減少
- **リスク**: 低〜中。拡張の多様性が減少し、実画像への汎化性能が低下する可能性
- **期待効果**: 部分的改善（エラー14件→2-5件程度と予想）
- **変更箇所**: GPUAugmentation.__init__()
- **変更内容**:
  ```python
  # Before
  K.RandomRotation(degrees=5, p=0.5),
  K.RandomPerspective(distortion_scale=0.15, p=0.3),

  # After
  K.RandomRotation(degrees=2, p=0.5),
  K.RandomPerspective(distortion_scale=0.05, p=0.3),
  ```

### 案の比較

| 案 | リスク | 期待効果 | 実装難度 | 副作用 |
|----|--------|----------|----------|--------|
| 案1: マージン追加 | 低 | 高 | 低 | パディング境界付近に微小ノイズが残る可能性 |
| 案2: crop-augment-paste | 中 | 高 | 高 | バッチ処理不可、速度低下 |
| 案3: 拡張強度低減 | 低〜中 | 中 | 低 | 汎化性能低下リスク |

### 選定: 案1（マージン追加）

**理由**:
1. 最小の変更で最大の効果が期待できる
2. GPU拡張の多様性を維持（汎化性能を損なわない）
3. マージン32pxは保守的な値で、テキストコンテンツ(~600px)に対して5%程度の許容範囲
4. 仮にマージン内にノイズが残っても、CTC decoderは44文字超の出力をtruncateするため実害なし

### MASK_MARGINの算出根拠

GPU拡張で最終文字が右方向に最大何pxシフトするかを計算:
- **RandomRotation(degrees=5)**: 画像高さ32pxの場合、右端で最大 32 * tan(5deg) = 2.8px
- **RandomPerspective(distortion_scale=0.15)**: 最大15%の歪みで、右端が最大 960 * 0.15 = 144px シフト（理論最大値）
  - ただしcontent幅600px付近では 600 * 0.15 = 90px 程度が理論最大
  - 実際にはランダムな4点移動のため、右端が一方向に90pxシフトする確率は極めて低い
- **実用的な値**: 32px（保守的にRotation+Perspectiveの合計をカバー）
  - 不十分な場合は48pxに引き上げ可能

### 実装計画

1. `train_epoch()` のパディングマスク処理に `MASK_MARGIN = 32` を追加
2. `validate()` の同じ箇所にも適用（v7.1では検証時GPU拡張なしだが、コード一貫性のため）
3. notebookのmarkdownヘッダをv7.3に更新
4. 既存のハイパーパラメータ・アーキテクチャは変更なし

### 結果 (v7.3)

- **CER**: 97.95% → **0.00%** (Epoch 1 → Epoch 3)
- **Accuracy**: 0.0% → **100.0%**
- **ONNX推論**: 1000/1000 正解 (**100%**), CER **0.00%**
- **学習時間**: ~17分 (40 epochs完走)
- **モデル**: FastCRNN (592,966パラメータ, 2.26 MB)
- **結果**: **目標達成** (CER < 1%)
- **v7.2比改善**:
  - ONNX推論: 986/1000 (99.0%) → **1000/1000 (100%)**
  - 44文字目→L誤認識: 14件 → **0件** (完全解消)
  - Epoch 35 LRリスタート: CER 44.46% → **0.02%** (崩壊解消)
- **MASK_MARGIN=32の効果**: GPU拡張による最終文字の右端クリッピングを防止し、U/J/V→L誤認識を根本解決
