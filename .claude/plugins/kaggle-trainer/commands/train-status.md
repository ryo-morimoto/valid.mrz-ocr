---
name: train-status
description: 現在の訓練状態とイテレーション履歴を表示
allowed_tools:
  - Bash
  - Read
---

# /train-status

現在のトレーニング状態を表示します。

## 使用方法
```
/train-status
```

## 表示内容

### 1. Kaggle Kernel Status
```bash
kaggle kernels status ryo-morimoto/train-crnn
```

### 2. Latest Metrics
最新の `crnn/outputs/latest/training_log.csv` から:
- Current CER
- Best CER (and epoch)
- Total epochs trained

### 3. Iteration History
`crnn/logs/iteration_history.md` から直近5回のイテレーションを表示

## 出力形式

```markdown
## Training Status

### Kaggle Kernel
- Status: running / complete / queued / error
- Started: YYYY-MM-DD HH:MM
- Elapsed: Xh Xm

### Current Metrics
| Metric | Value |
|--------|-------|
| Current CER | X.X% |
| Best CER | X.X% (epoch N) |
| Epochs | N |

### Recent Iterations
| # | Date | CER Change | Result |
|---|------|------------|--------|
| 5 | 01-25 | 5.2% → 3.1% | Improved |
| 4 | 01-24 | 5.5% → 5.2% | Improved |
| ... |

### Next Action
[推奨アクション]
```
