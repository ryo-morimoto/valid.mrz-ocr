# valid.mrz-ocr

## CRNN Training Automation

### 目標
- CER < 1% を達成する

### 分析基準
1. CER推移: 過去3 epochの傾向（下降/停滞/上昇）
2. エラーパターン: 混同しやすい文字ペア（O/0, I/1, B/8等）
3. 過学習兆候: train_loss↓ + val_CER↑

### 改善優先順位（低リスク順）
1. データ拡張調整
2. ハイパーパラメータ調整
3. アーキテクチャ変更

### コミット形式
```
fix(crnn): <意図>

仮説: <仮説>
変更: <変更内容>
```

### Kaggle Kernel
- Kernel path: `ryozom/train-crnn`
- Output files: `training_log.csv`, `best_model.pth`, `mrz_crnn.onnx`
