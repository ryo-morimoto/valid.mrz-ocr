---
name: train-loop
description: CER/mAP目標達成まで訓練→分析→改善→再実行を自動で回す
allowed-tools:
  - Task
  - Bash
  - Read
  - Edit
  - Write
  - WebSearch
  - WebFetch
---

# /train-loop

バックログ駆動で目標達成まで自動トレーニングループを回します。

## 引数
- `--model`: モデル名（`crnn` または `yolo`、**必須**）
- `--max-iterations`: 最大イテレーション数（デフォルト: 10）

## 使用方法
```
/train-loop --model crnn
/train-loop --model yolo
/train-loop --model crnn --max-iterations 5
```

## モデル設定

引数の `--model` に基づき、CLAUDE.md からモデル固有の設定を読み取る:

| 設定 | crnn | yolo |
|------|------|------|
| kernel | `ryozom/train-crnn` | `ryozom/train-yolo-mrz` |
| notebook | `crnn/notebooks/train_crnn.ipynb` | `yolo/notebooks/train_yolo.ipynb` |
| notebook_dir | `crnn/notebooks` | `yolo/notebooks` |
| output_dir | `crnn/outputs` | `yolo/outputs` |
| metric | `test_cer` (lower is better) | `metrics/mAP50(B)` (higher is better) |
| target | `< 1.0` | `> 95.0` |
| backlog | `docs/crnn/improvement-backlog.md` | `docs/yolo/improvement-backlog.md` |
| experiment_log | `docs/crnn/experiment-log.md` | `docs/yolo/experiment-log.md` |
| kaggle_cli | `~/.local/bin/kaggle` または `uvx kaggle` | 同左 |

## 実行フロー

```
iteration = 1
max_iterations = args.max_iterations or 10
model = args.model  # 必須

while iteration <= max_iterations:

    ## Step 1: Kernel 状態確認
    kaggle kernels status {kernel}

    ## Step 2: 状態に応じて分岐

    ### If RUNNING:
    バックグラウンドでポーリング（run_in_background: true）:
    ```bash
    while true; do
      result=$({kaggle_cli} kernels status {kernel} 2>&1)
      echo "$(date '+%H:%M:%S'): $result"
      if echo "$result" | grep -q "COMPLETE\|ERROR\|CANCEL"; then
        echo "DONE"; break
      fi
      sleep 300
    done
    ```
    → 完了通知を待つ

    ### If COMPLETE:
    続行

    ### If ERROR:
    エラー報告して終了

    ## Step 3: 結果取得
    # バージョン番号を決定（既存の最大 +1）
    version=$(ls {output_dir}/ | grep -oP 'v\d+' | sort -V | tail -1)
    next_version=v$(( ${version#v} + 1 ))  # 新結果用
    mkdir -p {output_dir}/{version}
    kaggle kernels output {kernel} -p {output_dir}/{version}/ --force

    ## Step 4: 目標達成チェック
    メトリクスログから最新値を確認:
    - crnn: training_log.csv の test_cer 列の最小値
    - yolo: results.csv の metrics/mAP50(B) 列の最大値

    if 目標達成:
        SUCCESS メッセージを表示
        experiment-log.md に最終結果を記録
        break

    ## Step 5: バックログ確認
    {backlog} を読み、ステータスが「未着手」の項目を推奨実行順序で取得。

    ### If 未着手項目あり:
    → Step 6 へ

    ### If 未着手項目なし (バックログ枯渇):
    → Step 5b: Researcher 起動
    Task(researcher) を呼び出し:
    - 入力: model名, experiment_log, backlog, 現在のメトリクス
    - 出力: backlog に新規手法を追加
    → Step 5 に戻って次の未着手項目を取得

    ## Step 6: Planner 起動
    Task(planner) を呼び出し:
    - 入力: model名, output_dir/{version}/, notebook, backlog, experiment_log
    - 出力: バックログ項目に基づく改善計画

    ## Step 7: Reviewer 起動
    Task(reviewer) を呼び出し:
    - 入力: 改善計画, backlog
    - 出力: レビュー結果

    if NEEDS_REVISION:
        planner に差し戻し（最大 3 回）

    ## Step 8: Worker 起動
    Task(worker) を呼び出し:
    - 入力: 承認された改善計画, model名, next_version
    - 出力: notebook 更新 + git commit + Kaggle push

    ## Step 9: 結果記録
    experiment-log.md に新エントリを追記（結果は次イテレーションで記入）
    backlog の該当項目ステータスを「実験中」に更新

    iteration++

if iteration > max_iterations:
    「{max_iterations}回実行しました。手動確認を推奨」
```

## 制約ルール (全 Step 共通)

1. **バックログ駆動**: planner は必ず backlog の未着手項目から選択する。場当たり的な変更は禁止
2. **最大 2 変更/version**: 因果分析のため、1 イテレーションの変更は最大 2 つ
3. **結果の蓄積**: 全イテレーションの結果を experiment-log.md に記録する
4. **バックログ枯渇時は調査**: 未着手項目がなくなったら researcher agent で手法を追加調査してから続行
