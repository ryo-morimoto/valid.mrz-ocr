# CRNN 学習トラブルシューティング

## 問題: `loss.backward()` がハングする

### 症状
- Forward pass は正常に完了（0.05秒程度）
- CTC Loss 計算も正常（0.001秒）
- `loss.backward()` で無限にハング（30秒以上経過しても完了しない）

### 原因
**`nn.ReLU(inplace=True)` と CTC Loss の組み合わせ**

```python
# 問題のあるコード
nn.ReLU(inplace=True)  # ← これが原因
```

### なぜ問題が起きるか

1. `inplace=True` は入力テンソルを直接書き換えてメモリを節約する
2. CTC Loss の backward は複雑なアライメント計算を行う
3. inplace 操作により元のテンソルが失われ、計算グラフの勾配計算で矛盾が発生
4. PyTorch が勾配計算をやり直そうとしてデッドロック or 無限ループに陥る

### 解決策

```python
# 修正後
nn.ReLU(inplace=False)  # 新しいテンソルを作成
```

### 検証方法

```python
# 各コンポーネント単体でテスト
# 1. CNN のみ → OK
# 2. LSTM のみ → OK
# 3. CTC Loss のみ → OK
# 4. CNN + LSTM + sum().backward() → OK
# 5. CNN + LSTM + CTC Loss + backward() → ハング (inplace=True の場合)
```

### 教訓

- `inplace=True` は CTC Loss など複雑な損失関数と組み合わせると問題を起こす可能性がある
- デバッグ時は各コンポーネントを切り分けてテストする
- メモリ節約より計算グラフの正確性を優先すべき場合がある

---

## 問題: `loss.backward()` がハングする（MKLDNN 関連）

### 症状
- batch size >= 4 で backward がハング
- batch size = 1 では動作する
- CPU 環境で発生

### 原因
**MKLDNN (oneDNN) convolution と OMP スレッドの競合**

PyTorch の MKLDNN 実装が特定の条件で OMP バリアでデッドロックを起こす。

参考: https://github.com/pytorch/pytorch/issues/91547

### 解決策

```python
import torch.backends.mkldnn
torch.backends.mkldnn.enabled = False
```

### 代替策

- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` 環境変数を設定
- DataLoader の `num_workers=0` に設定（既に設定済みでも発生する場合あり）

### 影響

- MKLDNN 無効化により CPU 推論が遅くなる可能性がある
- ただし学習は正常に完了する

---

## 問題: EfficientCRNN で精度が停滞する

### 症状
- EfficientCRNN (Depthwise Separable Conv) に変更後、精度が上がらない
- Loss は下がるが CER/Accuracy が改善しない

### 原因
**`nn.ReLU(inplace=True)` と CTC Loss の組み合わせ**

EfficientCRNN の以下の箇所で `inplace=True` が使用されていた:
1. `DepthwiseSeparableConv` 内の `self.relu`
2. CNN Block 1 の `nn.ReLU(inplace=True)`

### 解決策

```python
# DepthwiseSeparableConv 内
self.relu = nn.ReLU(inplace=False)

# CNN Block 1
nn.ReLU(inplace=False),
```

### 教訓

- CTC Loss を使う場合、**全ての ReLU で `inplace=False`** を使用する
- Depthwise Separable Conv など複雑なモジュールでも同様
- メモリ節約（inplace=True）より勾配計算の安定性を優先

### v4 での対応

`inplace=False` に修正しても精度が改善しなかったため、**AttentionCRNN（通常 Conv2d）に戻した**。

Depthwise Separable Conv の問題点:
- チャネル間の相互作用が制限される
- OCR では複数チャネルの特徴を組み合わせて文字を認識するため、表現力不足
- 計算効率より表現力が重要なタスクには不向き
