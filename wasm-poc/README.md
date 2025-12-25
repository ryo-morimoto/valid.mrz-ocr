# MRZ WASM PoC

ブラウザ上で YOLOv8-nano + Tesseract.js を WASM で動作させる検証。

## 検証結果サマリー（2024-12-26）

### ✅ VERDICT: GO

| 項目 | 目標 | 結果 | 判定 |
|------|------|------|------|
| YOLOv8-nano 推論 | <200ms | **32ms** | ✅ |
| Tesseract.js 認識 | <500ms | **166ms** | ✅ |
| 合計パイプライン | <700ms | **198ms** | ✅ |
| メモリ使用量 | <500MB | **14.2MB** | ✅ |

### 検証環境

| 項目 | 値 |
|------|-----|
| ブラウザ | Chrome (Windows) |
| CPU | Intel Core i7-7700K 4.20GHz |
| CPUコア数 | 8 |
| OS | Windows + WSL2 (Ubuntu) |
| Node.js | v20.19.6 |
| pnpm | 9.15.0 |
| Python | 3.12.12 |

### 重要な前提条件

1. **COOP/COEP ヘッダー必須**（マルチスレッド WASM 有効化）
   ```
   Cross-Origin-Opener-Policy: same-origin
   Cross-Origin-Embedder-Policy: require-corp
   ```

2. **入力サイズ 320×320**（640×640 では遅すぎた）
   - 640×640: 374-502ms
   - 320×320: 32ms（約10倍高速化）

3. **セッションキャッシュ**（モデルは初回のみロード）

## アーキテクチャ

```
イベント駆動パイプライン:

[Camera 30fps] → [YOLOv8-nano] → [安定検出判定] → [Tesseract.js] → [MRZ Result]
                   Detection       Stability          Recognition
                   (32ms)          (3frame連続)        (166ms)
```

- **Detection**: 毎フレーム実行可能（32ms）、MRZ 領域を検出
- **Recognition**: 安定検出時のみ実行（ワンショット）

## ディレクトリ構造

```
wasm-poc/
├── models/              # ONNX モデル
│   └── yolov8n.onnx     # ~12MB (320×320入力)
├── scripts/             # モデルエクスポート
│   └── export_yolo.py
├── public/              # ブラウザ検証ページ
│   ├── index.html
│   └── models/          # 公開用モデル（.gitignore）
├── server.mjs           # COOP/COEP対応サーバー
├── package.json
└── README.md
```

## クイックスタート

### 1. 依存関係インストール

```bash
# Nix シェルに入る (ultralytics 含む)
nix develop

# pnpm依存
cd wasm-poc
pnpm install
```

### 2. ONNX モデル生成

```bash
# YOLOv8-nano → ONNX (320×320)
python scripts/export_yolo.py

# public/models にコピー
mkdir -p public/models
cp models/yolov8n.onnx public/models/

# 確認
ls -lh models/
# yolov8n.onnx    ~12MB
```

### 3. ブラウザで検証

```bash
# COOP/COEP ヘッダー付きサーバー起動（重要！）
node server.mjs
# → http://localhost:3000 を開く
```

**注意**: `pnpm run serve` では COOP/COEP ヘッダーが付かず、マルチスレッドが無効になる。

## 判定基準

| 項目 | 成功基準 |
|------|----------|
| YOLOv8-nano | 推論 < 200ms |
| Tesseract.js | 認識 < 500ms |
| 合計パイプライン時間 | < 700ms |
| メモリ使用量 | < 500MB |

## パフォーマンス最適化の知見

### ✅ 効果あり

| 最適化 | 効果 |
|--------|------|
| 入力サイズ縮小 (640→320) | 10倍以上高速化 |
| COOP/COEP ヘッダー | ~50% 高速化 |
| セッションキャッシュ | ロード時間 0ms |
| SIMD 有効化 | 有効 |

### ❌ 効果なし/非対応

| 項目 | 理由 |
|------|------|
| WebGL バックエンド | YOLO の resize operator 非対応 |
| WebGPU バックエンド | onnxruntime-web 1.17.0 では未サポート |

## 次のステップ（Go判定後）

1. MIDV-500 から MRZ 領域アノテーション生成
2. YOLOv8-nano 転移学習（MRZ検出用）
3. イベント駆動パイプライン実装
4. 実画像での精度・速度測定

## トラブルシューティング

### モデルが見つからない

```
Could not fetch models/yolov8n.onnx
```

→ `python scripts/export_yolo.py` を実行し、`public/models/` にコピー

### 推論が遅い（>300ms）

1. COOP/COEP ヘッダーを確認（`node server.mjs` を使用）
2. 入力サイズが 320×320 か確認
3. Chrome を使用（Firefox より高速）

### 未サポートオペレーター

```
Error: operator xxx is not supported
```

→ opset バージョンを変更するか、別バックエンドを検討

### メモリ不足

→ INT8 量子化を検討:

```python
# ONNX 量子化
from onnxruntime.quantization import quantize_dynamic
quantize_dynamic("yolov8n.onnx", "yolov8n_int8.onnx")
```
