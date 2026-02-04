# CRNN MRZ OCR 改善バックログ

## 現状

- **Best CER**: 10.57% (v21)
- **目標**: < 1%
- **ギャップ**: ~10x 改善が必要
- **根本原因**: 19 ユニークラベル (train/test label overlap ゼロ) → ラベル暗記 vs 文字汎化

## CER 改善の推移と限界

```
25% ─┐
     │ v10-v12: synth remix 失敗
20% ─┤ v16: epoch 延長 (-10.46)
     │
15% ─┤ v18: 損失関数 + ICAO 合成 (-6.88)
     │
10% ─┤ v19-v21: 微調整の限界 (avg -0.94/ver) ← 現在地
     │
 5% ─┤ ??? パラダイムシフト必要
     │
 1% ─┘ 目標
```

**結論: パラメータ微調整の限界に到達。カテゴリの異なるアプローチが必要。**

---

## 改善手法一覧

優先順位: **期待効果 × 実装容易性 ÷ リスク** でソート

### Tier 1: 高効果 × 低リスク (最優先)

#### 1. MRZ 構造制約付きデコーディング
- **カテゴリ**: 推論時後処理
- **概要**: CTC デコーダ出力に ICAO 9303 構造制約を適用。チェックディジット検証、国籍コード辞書照合、日付フォーマット検証でエラーを修正
- **期待効果**: ★★★★★ (構造的に修正可能なエラーの大部分を除去)
- **根拠**: 現在の誤り (O→0, 9→5, G→S) の多くは MRZ 構造知識で判別可能。チェックディジットだけで数字位置のエラーを検出・修正できる。Research: post-OCR correction で CER 30-55% 削減の報告あり
- **リスク**: ★☆☆☆☆ (モデルに影響なし、推論時のみ)
- **実装量**: 中 (ICAO 9303 準拠のバリデーション + 候補探索ロジック)
- **ステータス**: 未着手
- **参考**: [Post-OCR Correction Study](https://arxiv.org/html/2408.02253v2), [MRZNet 99.25% f1](https://dl.acm.org/doi/abs/10.1007/s10032-021-00384-2)

#### 2. Self-Distillation CTC Loss (DCTC)
- **カテゴリ**: 損失関数
- **概要**: CTC loss に framewise regularization を追加。CTC は sequence 全体を最適化するが個々の文字を軽視する問題を self-distillation で解決
- **期待効果**: ★★★★☆ (CER 最大 2.6% 改善の報告)
- **根拠**: 現在の CTC loss は「全体的に合っていれば OK」とする傾向があり、特定文字位置の誤りが修正されにくい (v12 の 9→5 が 50 epoch 不変だった原因の可能性)。DCTC は追加パラメータ・追加推論コストなし
- **リスク**: ★★☆☆☆ (損失関数のみの変更)
- **実装量**: 小 (loss 計算の修正のみ)
- **ステータス**: 未着手
- **参考**: [Self-distillation Regularized CTC](https://arxiv.org/abs/2308.08806)

#### 3. Test-Time Augmentation (TTA)
- **カテゴリ**: 推論時
- **概要**: テスト時に複数の augmentation (微小回転、輝度変化、コントラスト) を適用し、CTC 出力を平均化
- **期待効果**: ★★★☆☆ (安定して 1-3% 改善の報告)
- **根拠**: 単一画像の撮影条件依存性を軽減。特に grc (13.7% CER) など撮影条件が悪いサンプルに有効
- **リスク**: ★☆☆☆☆ (モデル変更なし)
- **実装量**: 小
- **ステータス**: 未着手

#### 4. Manifold Mixup for CTC
- **カテゴリ**: 正則化 / データ拡張
- **概要**: CNN の hidden representation を interpolation することで、入力空間ではなく特徴空間でのデータ拡張を実現
- **期待効果**: ★★★☆☆ (CTC + text recognition に特化した論文で効果実証済み)
- **根拠**: 19 ラベルの暗記を防ぎ、特徴空間での smooth な decision boundary を学習。入力空間の augmentation より効果的
- **リスク**: ★★☆☆☆ (訓練ループの修正)
- **実装量**: 小-中
- **ステータス**: 未着手
- **参考**: [Manifold Mixup improves text recognition with CTC loss](https://arxiv.org/abs/1903.04246)

### Tier 2: 中効果 × 中リスク

#### 5. Curriculum Learning
- **カテゴリ**: 訓練戦略
- **概要**: 簡単なサンプル (uru, mda) → 難しいサンプル (grc, cze) の順に学習。Phase 3 の epoch を段階的に難易度上昇
- **期待効果**: ★★★☆☆
- **根拠**: Handwriting recognition で効果実証済み。現在は全サンプル均等だが、grc (13.7%) と mda (7.5%) の難易度差が大きい
- **リスク**: ★★☆☆☆
- **実装量**: 中 (サンプル難易度スコアリング + スケジューラ)
- **ステータス**: 未着手
- **参考**: [Curriculum learning for handwritten text line recognition](https://dblp.org/rec/conf/das/LouradourK14)

#### 6. Focal Loss / Online Hard Example Mining (OHEM)
- **カテゴリ**: 損失関数
- **概要**: 難しいサンプル/文字に損失の重みを集中。easy sample の勾配を抑制
- **期待効果**: ★★★☆☆
- **根拠**: PSFNet が focal loss でパスポート特徴認識の過学習を軽減。v21 で特定文字の改善/退行パターンが明確 → hard example mining が有効
- **リスク**: ★★☆☆☆
- **実装量**: 小
- **ステータス**: 未着手

#### 7. Ensemble (複数モデル平均)
- **カテゴリ**: 推論戦略
- **概要**: 異なる初期化・augmentation で 3-5 個のモデルを訓練し、CTC 出力を平均化
- **期待効果**: ★★★★☆ (CER 30-50% 削減の報告: Calamari)
- **根拠**: 個々のモデルの弱点が相殺される。Calamari の ensemble LSTM で 19世紀 Fraktur の CER < 1% を達成
- **リスク**: ★☆☆☆☆ (モデル変更なし)
- **実装量**: 中 (複数回訓練 + 推論パイプライン)
- **コスト**: 訓練時間 × モデル数、推論時間 × モデル数
- **ステータス**: 未着手
- **参考**: [Calamari OCR](https://github.com/Calamari-OCR/calamari)

#### 8. Pseudo-Labeling / Self-Training
- **カテゴリ**: データ拡張 (半教師あり)
- **概要**: 現在のモデルで未ラベル MRZ 画像にラベル付け → 高信頼度のみ訓練データに追加 → 再訓練
- **期待効果**: ★★★☆☆
- **根拠**: Momentum Pseudo-Labeling (MPL) が CTC ベースモデルの domain adaptation に有効。ラベル多様性 (19→拡大) の根本解決に最も近い
- **リスク**: ★★★☆☆ (ノイズラベルの蓄積リスク)
- **実装量**: 中-大 (未ラベル画像の収集 + confidence filtering + iterative training)
- **ステータス**: 未着手
- **参考**: [Intermediate CTC Loss with Pseudo-Labeling](https://www.emergentmind.com/topics/intermediate-ctc-loss)

#### 9. 合成データの質的改善: 実画像ノイズシミュレーション
- **カテゴリ**: データ拡張
- **概要**: 合成 MRZ 画像にリアルなノイズを追加 — ラミネーション反射、影、回転、ピンボケ、低解像度、背景テクスチャ
- **期待効果**: ★★★☆☆
- **根拠**: Domain adaptation の研究で、synthetic-to-real gap を縮小するにはノイズの種類・分布が重要。現在の apply_domain_adaptation() は基本的な変換のみ
- **リスク**: ★★☆☆☆
- **実装量**: 中
- **ステータス**: 未着手

### Tier 3: 高効果だが高リスク / 大規模

#### 10. Attention 機構追加 (CNN → Attention → CTC)
- **カテゴリ**: アーキテクチャ
- **概要**: CNN 特徴マップに Self-Attention や Squeeze-and-Excitation を追加。文字間の依存関係をモデル化
- **期待効果**: ★★★★☆
- **根拠**: MRZNet が pixel attention で判別力向上。現在の CNN+BiGRU は局所特徴に依存し、位置依存の誤り (pos 0 の 9→5) が発生
- **リスク**: ★★★☆☆ (アーキテクチャ変更、パラメータ増加)
- **実装量**: 中-大
- **ステータス**: 未着手

#### 11. TrOCR / Transformer ベースアーキテクチャへの移行
- **カテゴリ**: アーキテクチャ
- **概要**: CRNN を TrOCR (Vision Encoder + Text Decoder Transformer) に置換
- **期待効果**: ★★★★★ (SOTA: IAM dataset で CER 2-3%)
- **根拠**: Transformer の事前学習表現が cross-domain 汎化に優れる。CRNN vs VLM の比較で、real document accuracy が 72.5% → 93.1% に改善
- **リスク**: ★★★★★ (全面書き換え、推論コスト増大、WASM デプロイ困難)
- **実装量**: 大
- **ステータス**: 未着手
- **参考**: [TrOCR](https://arxiv.org/abs/2109.10282), [Manchu OCR study](https://arxiv.org/html/2507.06761v1)

#### 12. LLM ベース Post-OCR 修正
- **カテゴリ**: 推論後処理
- **概要**: CRNN 出力を小型 LLM (ByT5, BART) で sequence-to-sequence 修正
- **期待効果**: ★★★★☆ (CER 55% 削減の報告)
- **根拠**: MRZ の構造的知識 + 言語モデルの組み合わせで文字レベルの修正を超えた修正が可能
- **リスク**: ★★★★☆ (推論コスト増大、追加モデルの訓練が必要)
- **実装量**: 大
- **ステータス**: 未着手
- **参考**: [Scrambled text: OCR correction with synthetic data](https://arxiv.org/html/2409.19735v1)

---

## 推奨実行順序

上記を以下の順序で実施する。各ステップは **最大 2 変更** に制限し、因果分析を可能にする。

### Phase A: 推論時改善 (モデル再訓練不要)
1. **#1 MRZ 構造制約付きデコーディング** — 最大効果 × ゼロリスク
2. **#3 TTA** — 追加改善 × ゼロリスク

### Phase B: 損失関数 / 正則化改善
3. **#2 DCTC Loss** — 文字レベル学習の強化
4. **#4 Manifold Mixup** — 特徴空間の正則化

### Phase C: 訓練戦略改善
5. **#5 Curriculum Learning** または **#6 Focal Loss** — 難サンプルへのフォーカス
6. **#9 合成データノイズ改善** — domain gap 縮小

### Phase D: スケーリング
7. **#7 Ensemble** — 複数モデルによるロバスト化
8. **#8 Pseudo-Labeling** — ラベル多様性の根本解決

### Phase E: アーキテクチャ変更 (最終手段)
9. **#10 Attention 追加** — 現アーキテクチャの拡張
10. **#11 TrOCR 移行** — 全面的なパラダイムシフト

---

## 結果記録テンプレート

新しい手法を試すたびに以下を記録し、`experiment-log.md` に追記する:

```markdown
### vXX: [手法名]
- **バックログ項目**: #N [手法名]
- **仮説**: [なぜこの手法が効くと考えるか]
- **変更**: [具体的な変更内容]
- **結果**: CER X.XX% (±Y.YY)
  - 改善: [具体的な改善点]
  - 退行: [具体的な退行点]
- **教訓**: [次に活かすべき知見]
- **バックログ更新**: [ステータスを更新: 効果あり / 効果なし / 部分的]
```

---

## 更新ルール

1. 新しい改善手法を発見したら、このファイルに追加する
2. 実験結果が出たら、該当項目のステータスを更新する
3. 推奨実行順序は結果に応じて動的に見直す
4. **1 バージョンで最大 2 変更** を厳守し、因果分析を可能にする
