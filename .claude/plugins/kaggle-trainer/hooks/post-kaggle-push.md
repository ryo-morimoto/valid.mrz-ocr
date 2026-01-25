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

**Status:** Running on Kaggle

**Result:** (pending)
```

## 出力
```
Iteration logged to crnn/logs/iteration_history.md
Kaggle kernel started. Use /train-status to monitor.
```
