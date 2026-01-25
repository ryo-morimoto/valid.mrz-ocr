# Kaggle Trainer Plugin Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a Claude Code plugin that automates the Kaggle CRNN training loop: fetch results → analyze → plan improvements → Codex review → edit notebook → re-run until CER < 1%.

**Architecture:** Plugin with 3 agents (planner, reviewer, worker), 2 commands (/train-loop, /train-status), and 2 hooks (pre-commit validation, post-push logging). The main loop orchestrates agents sequentially, with Codex CLI providing external review.

**Tech Stack:** Claude Code Plugin API, Kaggle CLI, OpenAI Codex CLI, Bash

---

## Task 1: Create Plugin Structure

**Files:**
- Create: `.claude/plugins/kaggle-trainer/plugin.json`

**Step 1: Create plugin directory structure**

```bash
mkdir -p .claude/plugins/kaggle-trainer/{agents,commands,hooks}
```

**Step 2: Write plugin.json manifest**

```json
{
  "name": "kaggle-trainer",
  "version": "1.0.0",
  "description": "Automated Kaggle CRNN training loop with Codex review",
  "commands": "commands",
  "agents": "agents",
  "hooks": "hooks"
}
```

**Step 3: Verify structure**

Run: `ls -la .claude/plugins/kaggle-trainer/`
Expected: `plugin.json`, `agents/`, `commands/`, `hooks/` directories

**Step 4: Commit**

```bash
git add .claude/plugins/kaggle-trainer/plugin.json
git commit -m "feat(plugin): init kaggle-trainer plugin structure"
```

---

## Task 2: Add Training Rules to CLAUDE.md

**Files:**
- Create: `CLAUDE.md`

**Step 1: Create CLAUDE.md with training rules**

```markdown
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
- Kernel path: `ryo-morimoto/train-crnn`
- Output files: `training_log.csv`, `best_model.pth`, `mrz_crnn.onnx`
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add CRNN training automation rules"
```

---

## Task 3: Create Planner Agent

**Files:**
- Create: `.claude/plugins/kaggle-trainer/agents/planner.md`

**Step 1: Write planner agent definition**

```markdown
---
name: planner
description: Kaggle訓練結果を直接読み、分析と改善計画を一括で作成する
tools:
  - Bash
  - Read
  - Grep
---

# Planner Agent

あなたはCRNN訓練結果を分析し、改善計画を作成するエージェントです。

## 入力
- Kaggle kernel出力ディレクトリのパス
- 現在のnotebook (`crnn/notebooks/train_crnn.ipynb`)

## 分析タスク
1. ログファイルからepochごとのCER、loss、accuracyを抽出
2. CER推移を判定:
   - 下降傾向（良好）: 直近3 epochでCERが減少
   - 停滞: 3 epoch以上で変化 < 0.1%
   - 上昇（過学習）: val_CER上昇 + train_loss下降
3. エラーサンプルがあれば混同文字ペアを特定

## 計画作成タスク
CLAUDE.mdの改善優先順位に従い、変更案を作成:
1. データ拡張調整（最優先）
2. ハイパーパラメータ調整
3. アーキテクチャ変更（最終手段）

各変更には必ず以下を明記:
- **意図**: なぜこの変更をするか
- **仮説**: この変更で何が改善されると予想するか
- **変更箇所**: notebookの具体的なコード位置

## 出力形式

```markdown
## 分析結果

| Metric | Value | Trend |
|--------|-------|-------|
| Current CER | X.X% | ↓/→/↑ |
| Best CER | X.X% | epoch N |
| Train Loss | X.XXX | ↓/→/↑ |

### 診断
[1-2文で現状を説明]

### 混同パターン
- O ↔ 0: N件
- I ↔ 1: N件

---

## 改善計画 v{N}

### 変更1: [タイトル]
- **意図**: [なぜ]
- **仮説**: [期待効果]
- **変更箇所**: `crnn/notebooks/train_crnn.ipynb` L{N}-{M}
- **変更内容**:
  ```python
  # Before
  old_code

  # After
  new_code
  ```

### 変更2: ...
```
```

**Step 2: Commit**

```bash
git add .claude/plugins/kaggle-trainer/agents/planner.md
git commit -m "feat(plugin): add planner agent for result analysis"
```

---

## Task 4: Create Reviewer Agent

**Files:**
- Create: `.claude/plugins/kaggle-trainer/agents/reviewer.md`

**Step 1: Write reviewer agent definition**

```markdown
---
name: reviewer
description: OpenAI Codex CLIで改善計画をレビューする
tools:
  - Bash
  - Read
---

# Reviewer Agent

あなたはOpenAI Codex CLIを使って改善計画をレビューするエージェントです。

## 入力
- plannerが作成した改善計画（Markdown形式）

## タスク
1. 改善計画の内容を読み取る
2. Codex CLIに計画をレビュー依頼
3. レビュー結果を整形して返す

## 実行手順

### Step 1: 計画をファイルに保存
```bash
cat > /tmp/improvement_plan.md << 'EOF'
${PLAN_CONTENT}
EOF
```

### Step 2: Codex CLIでレビュー
```bash
codex -q "以下のCRNN訓練改善計画をレビューしてください。

技術的な問題点、見落としている観点、より良いアプローチがあれば指摘してください。
特に以下の観点で確認:
1. 仮説は妥当か
2. 変更によるリスクは考慮されているか
3. 優先順位は適切か

$(cat /tmp/improvement_plan.md)"
```

## 出力形式

```markdown
## Codex Review Result

### Status: APPROVED / NEEDS_REVISION

### Feedback
[Codexからのフィードバック]

### Concerns (if any)
- [懸念点1]
- [懸念点2]

### Suggestions (if any)
- [改善提案1]
- [改善提案2]
```

## 判定基準
- APPROVED: 重大な問題なし → workerに進む
- NEEDS_REVISION: 問題あり → plannerに差し戻し
```

**Step 2: Commit**

```bash
git add .claude/plugins/kaggle-trainer/agents/reviewer.md
git commit -m "feat(plugin): add reviewer agent for Codex review"
```

---

## Task 5: Create Worker Agent

**Files:**
- Create: `.claude/plugins/kaggle-trainer/agents/worker.md`

**Step 1: Write worker agent definition**

```markdown
---
name: worker
description: 改善計画に従いnotebookを編集し、Kaggleで実行する
tools:
  - Bash
  - Read
  - Edit
---

# Worker Agent

あなたは改善計画を実行するエージェントです。

## 入力
- reviewerが承認した改善計画
- 現在のイテレーション番号

## タスク
1. 改善計画の変更箇所をnotebookに適用
2. 変更をgit commit（意図・仮説をメッセージに含める）
3. Kaggle kernelをpush
4. 実行完了を待機
5. 結果ファイルを取得

## 実行手順

### Step 1: Notebookを編集
改善計画の各変更について、Editツールで適用:
- ファイル: `crnn/notebooks/train_crnn.ipynb`
- 変更箇所: 計画で指定されたライン

### Step 2: Git Commit
```bash
git add crnn/notebooks/train_crnn.ipynb
git commit -m "$(cat << 'EOF'
fix(crnn): [意図を1行で]

仮説: [期待される効果]
変更:
- [変更1の概要]
- [変更2の概要]

Iteration: N
EOF
)"
```

### Step 3: Kaggle Push
```bash
cd crnn/notebooks
kaggle kernels push
```

### Step 4: 完了待機
```bash
# ステータス確認ループ
while true; do
  status=$(kaggle kernels status ryo-morimoto/train-crnn | grep -o 'complete\|running\|error')
  if [ "$status" = "complete" ]; then
    echo "Training complete"
    break
  elif [ "$status" = "error" ]; then
    echo "Training failed"
    exit 1
  fi
  echo "Status: $status, waiting 60s..."
  sleep 60
done
```

### Step 5: 結果取得
```bash
kaggle kernels output ryo-morimoto/train-crnn -p crnn/outputs/iteration_N/
```

## 出力形式

```markdown
## Worker Execution Result

### Status: SUCCESS / FAILED

### Commit
- Hash: abc1234
- Message: fix(crnn): ...

### Kaggle Kernel
- Run ID: xxx
- Duration: Xh Xm
- Output: crnn/outputs/iteration_N/

### Next Step
[planner に戻るか、目標達成で終了か]
```
```

**Step 2: Commit**

```bash
git add .claude/plugins/kaggle-trainer/agents/worker.md
git commit -m "feat(plugin): add worker agent for notebook editing and Kaggle execution"
```

---

## Task 6: Create train-loop Command

**Files:**
- Create: `.claude/plugins/kaggle-trainer/commands/train-loop.md`

**Step 1: Write train-loop command**

```markdown
---
name: train-loop
description: CER目標達成まで訓練→分析→改善→再実行を自動で回す
allowed_tools:
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
```

**Step 2: Commit**

```bash
git add .claude/plugins/kaggle-trainer/commands/train-loop.md
git commit -m "feat(plugin): add /train-loop command for automated training"
```

---

## Task 7: Create train-status Command

**Files:**
- Create: `.claude/plugins/kaggle-trainer/commands/train-status.md`

**Step 1: Write train-status command**

```markdown
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
## 🏃 Training Status

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
| 5 | 01-25 | 5.2% → 3.1% | ✅ Improved |
| 4 | 01-24 | 5.5% → 5.2% | ✅ Improved |
| ... |

### Next Action
[推奨アクション]
```
```

**Step 2: Commit**

```bash
git add .claude/plugins/kaggle-trainer/commands/train-status.md
git commit -m "feat(plugin): add /train-status command for status display"
```

---

## Task 8: Create Pre-Commit Review Hook

**Files:**
- Create: `.claude/plugins/kaggle-trainer/hooks/pre-commit-review.md`

**Step 1: Write pre-commit hook**

```markdown
---
name: pre-commit-review
event: PreToolUse
match_tools:
  - Bash
match_arg_regex: "git commit.*crnn"
---

# Pre-Commit Review Hook

CRNN関連のコミット前に仮説が含まれているか確認します。

## トリガー条件
- `git commit` コマンドで
- パス or メッセージに `crnn` を含む

## 検証ルール
コミットメッセージに以下が含まれているか確認:
1. `仮説:` または `Hypothesis:` キーワード
2. `fix(crnn):` または `feat(crnn):` プレフィックス

## 処理

### 含まれている場合
→ ALLOW: コミット続行

### 含まれていない場合
→ BLOCK with message:

```
⚠️ CRNN訓練のコミットには仮説を含めてください。

期待される形式:
fix(crnn): <意図を1行で>

仮説: <この変更で期待される効果>
変更:
- <変更点1>
- <変更点2>

現在のメッセージ:
{actual_message}
```
```

**Step 2: Commit**

```bash
git add .claude/plugins/kaggle-trainer/hooks/pre-commit-review.md
git commit -m "feat(plugin): add pre-commit hook for hypothesis validation"
```

---

## Task 9: Create Post-Kaggle-Push Hook

**Files:**
- Create: `.claude/plugins/kaggle-trainer/hooks/post-kaggle-push.md`

**Step 1: Write post-push hook**

```markdown
---
name: post-kaggle-push
event: PostToolUse
match_tools:
  - Bash
match_arg_regex: "kaggle kernels push"
---

# Post-Kaggle-Push Hook

Kaggle push後にイテレーション履歴を記録します。

## トリガー条件
- `kaggle kernels push` コマンド実行後

## 処理

### Step 1: 履歴ファイル確認
```bash
mkdir -p crnn/logs
touch crnn/logs/iteration_history.md
```

### Step 2: 最新コミット情報取得
```bash
commit_hash=$(git rev-parse --short HEAD)
commit_msg=$(git log -1 --pretty=%B)
timestamp=$(date '+%Y-%m-%d %H:%M')
```

### Step 3: 履歴に追記
```markdown
---

## Iteration (${timestamp})

**Commit:** ${commit_hash}

**Changes:**
${commit_msg}

**Status:** ⏳ Running on Kaggle

**Result:** (pending)
```

## 出力
```
📝 Iteration logged to crnn/logs/iteration_history.md
⏳ Kaggle kernel started. Use /train-status to monitor.
```
```

**Step 2: Commit**

```bash
git add .claude/plugins/kaggle-trainer/hooks/post-kaggle-push.md
git commit -m "feat(plugin): add post-push hook for iteration logging"
```

---

## Task 10: Create Logs Directory and Initial History

**Files:**
- Create: `crnn/logs/.gitkeep`
- Create: `crnn/logs/iteration_history.md`

**Step 1: Create logs directory**

```bash
mkdir -p crnn/logs
```

**Step 2: Create initial iteration history file**

```markdown
# CRNN Training Iteration History

Training loop iterations for MRZ OCR CRNN model.

**Target:** CER < 1%

---

## Initial State (2026-01-25)

**Baseline Metrics:**
- CER: ~58% (pre-v5)
- Expected v5 improvement: significant

**Starting Configuration:**
- MAX_WIDTH: 960
- Training samples: 50K
- Batch size: 128
- Epochs: 100
```

**Step 3: Create .gitkeep**

```bash
touch crnn/logs/.gitkeep
```

**Step 4: Commit**

```bash
git add crnn/logs/
git commit -m "feat(crnn): add iteration history tracking"
```

---

## Task 11: Verify Plugin Structure

**Step 1: Verify all files exist**

Run: `find .claude/plugins/kaggle-trainer -type f | sort`

Expected:
```
.claude/plugins/kaggle-trainer/agents/planner.md
.claude/plugins/kaggle-trainer/agents/reviewer.md
.claude/plugins/kaggle-trainer/agents/worker.md
.claude/plugins/kaggle-trainer/commands/train-loop.md
.claude/plugins/kaggle-trainer/commands/train-status.md
.claude/plugins/kaggle-trainer/hooks/post-kaggle-push.md
.claude/plugins/kaggle-trainer/hooks/pre-commit-review.md
.claude/plugins/kaggle-trainer/plugin.json
```

**Step 2: Verify plugin.json is valid JSON**

Run: `cat .claude/plugins/kaggle-trainer/plugin.json | python3 -m json.tool`

Expected: Valid JSON output without errors

**Step 3: Verify CLAUDE.md exists**

Run: `head -20 CLAUDE.md`

Expected: Training automation rules visible

---

## Task 12: Test Plugin Loading (Manual)

**Step 1: Restart Claude Code session**

```bash
# Exit current session and restart
claude
```

**Step 2: Verify plugin loaded**

Check that `/train-loop` and `/train-status` commands are available.

**Step 3: Test /train-status**

Run: `/train-status`

Expected: Shows current training status (may show "no history" initially)

---

## Summary

| Task | Component | Description |
|------|-----------|-------------|
| 1 | Plugin | Create directory structure and manifest |
| 2 | Rules | Add training rules to CLAUDE.md |
| 3 | Agent | Create planner agent |
| 4 | Agent | Create reviewer agent |
| 5 | Agent | Create worker agent |
| 6 | Command | Create /train-loop |
| 7 | Command | Create /train-status |
| 8 | Hook | Create pre-commit validation |
| 9 | Hook | Create post-push logging |
| 10 | Logs | Create iteration history tracking |
| 11 | Verify | Verify plugin structure |
| 12 | Test | Test plugin loading |
