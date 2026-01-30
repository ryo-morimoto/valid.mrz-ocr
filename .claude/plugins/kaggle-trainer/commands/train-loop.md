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

    ## Step 1: Kernel状態確認
    ```bash
    uvx kaggle kernels status ryozom/train-crnn
    ```

    ## Step 2: 状態に応じて分岐

    ### If RUNNING:
    バックグラウンドでポーリング開始:
    ```bash
    # run_in_background: true で実行
    while true; do
      result=$(uvx kaggle kernels status ryozom/train-crnn 2>&1)
      echo "$(date '+%H:%M:%S'): $result"
      case "$result" in
        *COMPLETE*) echo "TRAINING_COMPLETE"; break ;;
        *ERROR*|*CANCEL*) echo "TRAINING_FAILED"; break ;;
      esac
      sleep 300  # 5分間隔
    done
    ```
    → 完了通知を待つ

    ### If COMPLETE:
    続行

    ### If ERROR:
    エラー報告して終了

    ## Step 3: 結果取得
    ```bash
    mkdir -p crnn/outputs/latest
    uvx kaggle kernels output ryozom/train-crnn -p crnn/outputs/latest/ --force
    ```

    ## Step 4: CER確認
    training_log.csv から最新CERを確認。
    if CER <= target_cer:
        SUCCESS: 「CER {value}% 達成！{iteration}回のイテレーションで完了」
        break

    ## Step 5: Planner起動
    Task(planner)を呼び出し:
    - 入力: crnn/outputs/latest/, crnn/notebooks/train_crnn.ipynb
    - 出力: 改善計画

    ## Step 6: Reviewer起動
    Task(reviewer)を呼び出し:
    - 入力: 改善計画
    - 出力: レビュー結果

    if レビュー結果 == NEEDS_REVISION:
        plannerに差し戻し（最大3回）

    ## Step 7: Worker起動
    Task(worker)を呼び出し:
    - 入力: 承認された改善計画, iteration番号
    - 出力: ノートブック更新 + Kaggle push

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
