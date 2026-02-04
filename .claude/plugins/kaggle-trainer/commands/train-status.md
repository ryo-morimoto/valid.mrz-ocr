---
name: train-status
description: 現在の訓練状態とイテレーション履歴を表示
allowed-tools:
  - Bash
  - Read
---

# /train-status

## 引数
- `--model`: モデル名（`crnn` または `yolo`、**必須**）

## 使用方法
```
/train-status --model crnn
/train-status --model yolo
```

## モデル設定

| 設定 | crnn | yolo |
|------|------|------|
| kernel | `ryozom/train-crnn` | `ryozom/train-yolo-mrz` |
| output_dir | `crnn/outputs` | `yolo/outputs` |
| metric | `test_cer` (lower is better) | `metrics/mAP50(B)` (higher is better) |
| target | `< 1.0` | `> 95.0` |
| backlog | `docs/crnn/improvement-backlog.md` | `docs/yolo/improvement-backlog.md` |
| experiment_log | `docs/crnn/experiment-log.md` | `docs/yolo/experiment-log.md` |
| kaggle_cli | `~/.local/bin/kaggle` | `~/.local/bin/kaggle` |

## 実行すること

### Step 1: Status を確認
```bash
{kaggle_cli} kernels status {kernel}
```

### Step 2: Status に応じて分岐

#### If COMPLETE:
1. 最新出力をダウンロード:
```bash
mkdir -p {output_dir}/latest
{kaggle_cli} kernels output {kernel} -p {output_dir}/latest/ --force
```

2. メトリクスログをパース:
   - **crnn**: `training_log.csv` → Best CER, 最終 CER, 総 epoch 数
   - **yolo**: `results.csv` → Best mAP@0.5, 最終 mAP, 総 epoch 数

3. experiment-log.md を読んで直近の実験結果を表示

4. backlog の次の未着手項目を表示

#### If RUNNING:
1. 「実行中。完了を待ってください。」と報告
2. experiment-log.md の最新エントリを表示

#### If ERROR:
1. エラー状態を報告
2. Kaggle Web UI での確認を促す

## 出力形式

```markdown
## Training Status: {model}

### Kaggle Kernel
- Kernel: {kernel}
- Status: {status}

### Metrics (if available)
| Metric | Value |
|--------|-------|
| Best {metric} | X.X% (epoch N) |
| Final {metric} | X.X% |
| Epochs | N |
| Target | {target} |
| Gap | X.X% |

### Backlog Progress
- Total: N items
- Completed: N
- In Progress: N
- Remaining: N
- **Next**: #{N} {手法名}

### Recent Experiments
[experiment-log.md の直近 3 エントリ]
```
