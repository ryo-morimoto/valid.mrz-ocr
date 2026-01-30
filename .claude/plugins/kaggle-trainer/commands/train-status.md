---
name: train-status
description: 現在の訓練状態とイテレーション履歴を表示
allowed-tools:
  - Bash
  - Read
---

# /train-status

## 実行すること

### Step 1: Statusを確認
```bash
uvx kaggle kernels status ryozom/train-crnn
```

### Step 2: Statusに応じて分岐

#### If COMPLETE:
1. 出力をダウンロード:
```bash
mkdir -p crnn/outputs/latest
uvx kaggle kernels output ryozom/train-crnn -p crnn/outputs/latest/ --force
```

2. training_log.csv をパースしてメトリクスを表示:
   - 最終epoch の CER
   - Best CER とそのepoch
   - 総epoch数

3. iteration_history.md を読んで直近イテレーションを表示

#### If RUNNING:
1. 利用可能な出力ファイルを確認:
```bash
uvx kaggle kernels files ryozom/train-crnn
```

2. iteration_history.md を読んで現在の状態を表示

3. 「実行中。完了を待ってください。」と報告

#### If ERROR:
1. エラー状態を報告
2. Kaggle Web UI での確認を促す: https://www.kaggle.com/code/ryozom/train-crnn

## 出力形式

```markdown
## Training Status

### Kaggle Kernel
- Status: [status]
- [If complete: メトリクス表示]

### Metrics (if available)
| Metric | Value |
|--------|-------|
| Final CER | X.X% |
| Best CER | X.X% (epoch N) |
| Epochs | N |

### Recent Iterations
[iteration_history.md の内容]
```
