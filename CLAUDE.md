# valid.mrz-ocr

## Training Automation

### 改善管理ドキュメント (モデル共通)
- **実験ログ**: `docs/{model}/experiment-log.md` — 全バージョンの変更・結果・教訓を蓄積
- **改善バックログ**: `docs/{model}/improvement-backlog.md` — リサーチ済み手法の優先順位付きリスト

### 改善プロセス (必須)
1. **バックログ参照**: 次に試す手法は `docs/{model}/improvement-backlog.md` の推奨実行順序に従う
2. **最大 2 変更/version**: 因果分析を可能にするため、1 バージョンの変更は最大 2 つに制限
3. **結果記録**: 実験完了後、`docs/{model}/experiment-log.md` に結果を追記し、バックログのステータスを更新
4. **場当たり的変更の禁止**: バックログに記載されていない手法を試す場合、先にバックログに追加してから実施
5. **バックログ枯渇時**: 未着手項目がなくなったら researcher agent が自動で改善手法を追加調査

### コミット形式
```
fix({model}): <意図>

バックログ: #N <手法名>
仮説: <仮説>
変更: <変更内容>
```

---

## CRNN (MRZ 文字認識)

- **目標**: CER < 1%
- **Kaggle Kernel**: `ryozom/train-crnn`
- **Notebook**: `crnn/notebooks/train_crnn.ipynb`
- **Output**: `crnn/outputs/vN/` (`training_log.csv`, `best_model.pth`, `mrz_crnn.onnx`)
- **主要メトリクス**: test_cer (Levenshtein CER%)
- **分析基準**:
  1. CER推移: 過去3 epochの傾向（下降/停滞/上昇）
  2. エラーパターン: 混同しやすい文字ペア（O/0, I/1, B/8等）
  3. 過学習兆候: train_loss↓ + val_CER↑

## YOLO (MRZ 領域検出)

- **目標**: mAP@0.5 > 95%
- **Kaggle Kernel**: `ryozom/train-yolo-mrz`
- **Notebook**: `yolo/notebooks/train_yolo.ipynb`
- **Output**: `yolo/outputs/` (`results.csv`, `best.pt`, `mrz_yolov8n.onnx`)
- **主要メトリクス**: metrics/mAP50(B)
- **分析基準**:
  1. mAP/precision/recall 推移
  2. GT coverage (検出 bbox が MRZ 領域を十分にカバーしているか)
  3. 過学習兆候: train_loss↓ + val_mAP↓
