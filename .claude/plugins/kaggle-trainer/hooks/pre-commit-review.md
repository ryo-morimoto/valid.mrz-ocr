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
CRNN訓練のコミットには仮説を含めてください。

期待される形式:
fix(crnn): <意図を1行で>

仮説: <この変更で期待される効果>
変更:
- <変更点1>
- <変更点2>

現在のメッセージ:
{actual_message}
```
