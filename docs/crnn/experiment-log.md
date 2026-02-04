# CRNN MRZ OCR 実験ログ

## 目標

**Test CER < 1%** (Levenshtein 編集距離ベース、MIDV-500 テストセット 4 カ国)

## 制約

- Train: 10 パスポートタイプ (19 ユニークラベル, MIDV-500 600枚 + MIDV-2019 800枚)
- Test: 4 パスポートタイプ (cze, grc, mda, ury) — **train と label overlap ゼロ**
- アーキテクチャ: FastCRNN v6 (CNN 5層 + BiGRU 1層, ~593K params)
- 3-Phase: Phase 1 (合成 CTC 整列) → Phase 2 (ICAO 構造化合成) → Phase 3 (実画像 fine-tuning)

---

## 実験結果一覧

| Version | Best CER | Delta | 主要変更 | 判定 |
|---------|----------|-------|---------|------|
| v10 | 25.28% | — | Phase 3 real-only baseline | baseline |
| v11 | 25.83% | +0.55 | Synth remix 30% (40px) | **悪化** |
| v12 | 27.43% | +1.60 | Synth remix 30% (32px) + 半決定的 DA | **悪化** |
| v14 | 36.65% | — | Progressive unfreeze (再構築) | baseline (新構成) |
| v15 | 30.73% | -5.92 | Simple baseline (unfreeze 撤廃) | **改善** |
| v16 | 20.27% | -10.46 | Extended training (100 epochs) | **大幅改善** |
| v17 | — | — | (ログ欠損) | — |
| v18 | 13.39% | -6.88 | Levenshtein CER + ICAO 合成混合 | **大幅改善** |
| v19 | 11.72% | -1.67 | Extended training (200 epochs) | **改善** |
| v20 | 10.70% | -1.02 | MIDV-2019 視覚多様性追加 | **改善** |
| v21 | 10.57% | -0.13 | Synth 50% + 2x real + ZGO hard sampling | **微改善 + 退行あり** |
| v22 | (実行中) | — | 2x real 撤廃 + G hard sampling 修正 | 待機中 |

---

## 詳細ログ

### v10: Phase 3 Real-Only Baseline
- **仮説**: 実画像のみで fine-tuning すれば domain gap が最小になる
- **変更**: Phase 3 を real data only で構成
- **結果**: CER 25.28%
- **教訓**: 19 ユニークラベルでは汎化が不足。文字レベル学習ではなくラベル暗記が発生

### v11: Synth Remix 30% (40px)
- **仮説**: 合成データ混合でラベル多様性を補完できる
- **変更**: Phase 3 に synth remix 30% を追加 (40px height)
- **結果**: CER 25.83% (+0.55)
- **教訓**: **ランダム合成は実画像と domain gap が大きく、混合しても改善しない**

### v12: Synth Remix 30% (32px) + 半決定的 DA
- **仮説**: 合成画像サイズを実画像に合わせ、DA で近づければ改善する
- **変更**: 32px height + 半決定的ドメイン適応
- **結果**: CER 27.43% (+1.60)
- **教訓**: **Synth remix は 3 バージョン連続で悪化。アプローチ自体が無効**

### v14: Progressive Unfreeze (再構築)
- **仮説**: CNN 層を段階的に解凍すれば fine-tuning が安定する
- **変更**: Phase 3 を progressive unfreeze で再構築
- **結果**: CER 36.65%
- **教訓**: Progressive unfreeze は逆効果。初期層の固定が特徴抽出を阻害

### v15: Simple Baseline
- **仮説**: シンプルな全層 fine-tuning が最適
- **変更**: Progressive unfreeze を撤廃
- **結果**: CER 30.73% (-5.92)
- **教訓**: **シンプルが勝つ。不要な複雑性は排除すべき**

### v16: Extended Training (100 epochs)
- **仮説**: 学習が不足している (50 epochs では足りない)
- **変更**: Phase 3 を 100 epochs に延長
- **結果**: CER 20.27% (-10.46)
- **教訓**: **訓練時間の延長は大きな効果あり。まだ収束していなかった**

### v18: Levenshtein CER + ICAO 合成混合
- **仮説**: (A) CTC Loss → Levenshtein CER で文字単位の最適化を改善 (B) ICAO 構造化合成データで domain gap を縮小
- **変更**: Levenshtein CER 損失関数 + ICAO 9303 準拠合成データ混合
- **結果**: CER 13.39% (-6.88)
- **教訓**: **損失関数改善 + 合成データの質向上が大きく寄与。2 つ同時に変更したため個別効果は不明**

### v19: Extended Training (200 epochs)
- **仮説**: v18 の学習曲線がまだ収束していない
- **変更**: 200 epochs に延長
- **結果**: CER 11.72% (-1.67)
- **教訓**: 改善はあるが v16 での延長効果 (-10.46) と比べ収穫逓減。**epoch 延長だけでは限界が近い**

### v20: MIDV-2019 Visual Diversity
- **仮説**: 同じラベルの異なる撮影条件画像で visual diversity を倍増すれば汎化が改善する
- **変更**: MIDV-2019 (800枚) を Phase 3 train に追加
- **結果**: CER 10.70% (-1.02)
- **教訓**: 視覚的多様性は改善に寄与するが、**ラベル多様性 (19→19 のまま) が根本ボトルネック**

### v21: Synth 50% + 2x Real + ZGO Hard Sampling
- **仮説**: (A) synth ratio 増加で文字汎化を促進 (B) 2x real repetition で実画像の影響力を維持 (C) Z/G/O の hard sampling で混同文字を強化
- **変更**: synth 30%→50%, real 2x repetition, line1 ZGO hard sampling
- **結果**: CER 10.57% (-0.13), **ただし退行あり**
  - Z→I 修正 (25→0)
  - O→0 改善 (26→17)
  - **`<` DELETED 悪化 (61→107)**: 2x real repetition が原因
  - **G→S 新規エラー (34)**: line1 ZGO hard sampling が原因
- **教訓**:
  1. **2x real repetition は削除エラーを +75% 悪化させる** (ラベル暗記を強化)
  2. **line1 の文字ハードサンプリングはコンテキストバイアスを生む** (GUZMAN, GOMEZ パターン)
  3. **3 つ同時変更は因果分析を困難にする。最大 2 変更に制限すべき**

### v22: 2x Real 撤廃 + G Hard Sampling 修正 (実行中)
- **仮説**: v21 の退行要因 2 つを除去すれば CER < 10% に改善する
- **変更**: 2x repetition 撤廃, line1 ZGO hard sampling 撤去, line2 confusable に G,C 追加
- **結果**: 実行中
- **期待**: CER ~9.5-10.0%

---

## パターンと教訓 (横断的)

### 効果があった施策
1. **訓練時間延長** (v16: -10.46, v19: -1.67) — ただし収穫逓減
2. **損失関数改善** (v18: Levenshtein CER, -6.88 の一部)
3. **合成データの質向上** (v18: ICAO 構造化, -6.88 の一部)
4. **シンプル化** (v15: -5.92, progressive unfreeze 撤廃)
5. **視覚多様性の追加** (v20: MIDV-2019, -1.02)

### 効果がなかった / 悪化した施策
1. **Synth remix** (v11, v12: 一貫して悪化) — ランダム合成は domain gap 大
2. **Progressive unfreeze** (v14: 悪化) — 不要な複雑性
3. **2x real repetition** (v21: 削除エラー +75%) — ラベル暗記を強化
4. **line1 文字ハードサンプリング** (v21: G→S 新規エラー) — コンテキストバイアス

### CER 改善の収穫逓減
```
v14→v18: -23.26 pts (4 versions, avg -5.82/version)
v18→v21: -2.82 pts (3 versions, avg -0.94/version)
```
**パラメータ微調整の限界に到達。パラダイムシフトが必要。**

---

## 更新ルール

新バージョンの結果が出るたびに以下を追記:
1. 詳細ログセクションにエントリを追加
2. 結果一覧テーブルを更新
3. パターンと教訓を該当する場合に更新
