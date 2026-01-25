---
name: train-loop
description: CER目標達成まで訓練→分析→改善→再実行を自動で回す
allowed-tools:
  - Task
  - Bash
  - Read
  - Edit
---

# /train-loop

CER目標達成まで自動でトレーニングループを回します。

## 引数
- `--target-cer`: 目標CER%（デフォルト: 1.0）
- `--max-iterations`: 最大イテレーション数（デフォルト: 10）

## 使用方法
```
/train-loop
/train-loop --target-cer 0.5
/train-loop --max-iterations 5
```

## 実行フロー

```
iteration = 1
target_cer = args.target_cer or 1.0
max_iterations = args.max_iterations or 10

while iteration <= max_iterations:

    ## Step 1: 結果取得
    最新のKaggle kernel出力を取得:
    ```bash
    kaggle kernels output ryo-morimoto/train-crnn -p crnn/outputs/latest/
    ```

    ## Step 2: CER確認
    training_log.csv または出力ログから最新CERを確認。
    if CER <= target_cer:
        SUCCESS: 「CER {value}% 達成！{iteration}回のイテレーションで完了」
        break

    ## Step 3: Planner起動
    Task(planner)を呼び出し:
    - 入力: crnn/outputs/latest/, crnn/notebooks/train_crnn.ipynb
    - 出力: 改善計画

    ## Step 4: Reviewer起動
    Task(reviewer)を呼び出し:
    - 入力: 改善計画
    - 出力: レビュー結果

    if レビュー結果 == NEEDS_REVISION:
        plannerに差し戻し（最大3回）

    ## Step 5: Worker起動
    Task(worker)を呼び出し:
    - 入力: 承認された改善計画, iteration番号
    - 出力: 実行結果

    ## Step 6: 待機
    Worker内でKaggle完了を待機（最大3時間）

    iteration++

if iteration > max_iterations:
    「{max_iterations}回実行しましたがCER {value}%。手動確認を推奨」
```

## イテレーション履歴
各イテレーションの結果を `crnn/logs/iteration_history.md` に追記:

```markdown
## Iteration N (YYYY-MM-DD HH:MM)
- CER: X.X% → Y.Y%
- 変更: [概要]
- 仮説: [仮説]
- 結果: [改善/悪化/変化なし]
```
