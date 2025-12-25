# WASM + ONNX Runtime 最速検証 PoC

## 検証項目

| # | 検証内容 | 成功基準 | 失敗時の代替 |
|---|---------|---------|-------------|
| 1 | YOLOv8-nano ONNX → WASM動作 | 推論実行可能 | TFLite.js |
| 2 | PaddleOCR mobile ONNX → WASM動作 | 推論実行可能 | Tesseract.js |
| 3 | ブラウザ推論速度 | < 500ms/image | WebGPU検討 |
| 4 | メモリ使用量 | < 500MB | モデル量子化 |
| 5 | 初回ロード時間 | < 5秒 | lazy load |

---

## Step 1: 環境準備 (30分)

```bash
mkdir mrz-wasm-poc && cd mrz-wasm-poc

# ディレクトリ構造
mkdir -p models src public
```

## Step 2: ONNXモデル取得 (30分)

```bash
# YOLOv8-nano (pretrained)
pip install ultralytics
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=640, simplify=True, opset=12)
"
mv yolov8n.onnx models/

# PaddleOCR mobile recognition (pretrained)
# Option A: paddle2onnx
pip install paddlepaddle paddleocr paddle2onnx

python -c "
from paddleocr import PaddleOCR
import os

# モデルダウンロード
ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False)

# 変換 (認識モデルのみ)
import paddle2onnx
paddle2onnx.export(
    model_dir='~/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer',
    model_filename='inference.pdmodel',
    params_filename='inference.pdiparams',
    save_file='models/ppocr_rec.onnx',
    opset_version=12,
    enable_onnx_checker=True
)
"

# モデルサイズ確認
ls -lh models/
# yolov8n.onnx    ~6MB
# ppocr_rec.onnx  ~10MB
```

## Step 3: ONNX Runtime Web で検証 (最速ルート)

**Rustではなく、まずonnxruntime-webで動作確認**

```bash
npm init -y
npm install onnxruntime-web
```

### src/test-yolo.js

```javascript
// Step 3a: YOLOv8-nano WASM動作検証
import * as ort from 'onnxruntime-web';

// WASM backend設定
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

async function testYoloInference() {
  console.time('model-load');
  
  try {
    // モデル読み込み
    const session = await ort.InferenceSession.create(
      './models/yolov8n.onnx',
      { 
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      }
    );
    console.timeEnd('model-load');
    
    // 入力情報確認
    console.log('Input names:', session.inputNames);
    console.log('Output names:', session.outputNames);
    
    // ダミー入力で推論テスト
    const inputShape = [1, 3, 640, 640];
    const inputData = new Float32Array(1 * 3 * 640 * 640).fill(0.5);
    const inputTensor = new ort.Tensor('float32', inputData, inputShape);
    
    console.time('inference');
    const results = await session.run({ images: inputTensor });
    console.timeEnd('inference');
    
    // 出力確認
    const output = results[session.outputNames[0]];
    console.log('Output shape:', output.dims);
    console.log('✅ YOLOv8-nano WASM: SUCCESS');
    
    return { success: true, loadTime: performance.now() };
    
  } catch (error) {
    console.error('❌ YOLOv8-nano WASM: FAILED');
    console.error(error.message);
    return { success: false, error: error.message };
  }
}

// 実行
testYoloInference();
```

### src/test-ppocr.js

```javascript
// Step 3b: PaddleOCR WASM動作検証
import * as ort from 'onnxruntime-web';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

async function testPaddleOCRInference() {
  console.time('model-load');
  
  try {
    const session = await ort.InferenceSession.create(
      './models/ppocr_rec.onnx',
      { 
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      }
    );
    console.timeEnd('model-load');
    
    console.log('Input names:', session.inputNames);
    console.log('Output names:', session.outputNames);
    
    // PaddleOCR recognition: [1, 3, 48, 320]
    const inputShape = [1, 3, 48, 320];
    const inputData = new Float32Array(1 * 3 * 48 * 320).fill(0.5);
    const inputTensor = new ort.Tensor('float32', inputData, inputShape);
    
    console.time('inference');
    const results = await session.run({ x: inputTensor });
    console.timeEnd('inference');
    
    const output = results[session.outputNames[0]];
    console.log('Output shape:', output.dims);
    console.log('✅ PaddleOCR WASM: SUCCESS');
    
    return { success: true };
    
  } catch (error) {
    console.error('❌ PaddleOCR WASM: FAILED');
    console.error(error.message);
    
    // エラー詳細（未サポートオペレーター等）
    if (error.message.includes('operator')) {
      console.error('→ 未サポートオペレーターの可能性');
    }
    
    return { success: false, error: error.message };
  }
}

testPaddleOCRInference();
```

### public/index.html

```html
<!DOCTYPE html>
<html>
<head>
  <title>MRZ WASM PoC</title>
</head>
<body>
  <h1>MRZ OCR WASM 検証</h1>
  
  <div id="results">
    <h2>検証結果</h2>
    <pre id="log"></pre>
  </div>
  
  <h2>実画像テスト</h2>
  <input type="file" id="imageInput" accept="image/*">
  <canvas id="canvas" style="max-width: 100%;"></canvas>
  <pre id="output"></pre>

  <script type="module">
    import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/ort.min.js';
    
    const log = document.getElementById('log');
    const addLog = (msg) => { log.textContent += msg + '\n'; };
    
    // メモリ使用量監視
    function getMemoryUsage() {
      if (performance.memory) {
        return (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2) + ' MB';
      }
      return 'N/A';
    }
    
    async function runTests() {
      addLog('=== WASM検証開始 ===');
      addLog(`初期メモリ: ${getMemoryUsage()}`);
      
      // Test 1: YOLOv8-nano
      addLog('\n[1] YOLOv8-nano テスト...');
      try {
        const startLoad = performance.now();
        const yoloSession = await ort.InferenceSession.create(
          './models/yolov8n.onnx',
          { executionProviders: ['wasm'] }
        );
        const loadTime = (performance.now() - startLoad).toFixed(0);
        addLog(`  ロード時間: ${loadTime}ms`);
        addLog(`  メモリ: ${getMemoryUsage()}`);
        
        // 推論テスト
        const input = new ort.Tensor('float32', 
          new Float32Array(1*3*640*640), [1,3,640,640]);
        
        const startInfer = performance.now();
        await yoloSession.run({ images: input });
        const inferTime = (performance.now() - startInfer).toFixed(0);
        addLog(`  推論時間: ${inferTime}ms`);
        addLog('  ✅ SUCCESS');
        
      } catch (e) {
        addLog(`  ❌ FAILED: ${e.message}`);
      }
      
      // Test 2: PaddleOCR
      addLog('\n[2] PaddleOCR テスト...');
      try {
        const startLoad = performance.now();
        const ocrSession = await ort.InferenceSession.create(
          './models/ppocr_rec.onnx',
          { executionProviders: ['wasm'] }
        );
        const loadTime = (performance.now() - startLoad).toFixed(0);
        addLog(`  ロード時間: ${loadTime}ms`);
        addLog(`  メモリ: ${getMemoryUsage()}`);
        
        const input = new ort.Tensor('float32',
          new Float32Array(1*3*48*320), [1,3,48,320]);
        
        const startInfer = performance.now();
        await ocrSession.run({ x: input });
        const inferTime = (performance.now() - startInfer).toFixed(0);
        addLog(`  推論時間: ${inferTime}ms`);
        addLog('  ✅ SUCCESS');
        
      } catch (e) {
        addLog(`  ❌ FAILED: ${e.message}`);
      }
      
      addLog('\n=== 検証完了 ===');
      addLog(`最終メモリ: ${getMemoryUsage()}`);
    }
    
    // 起動時に実行
    runTests();
  </script>
</body>
</html>
```

## Step 4: ローカルサーバーで検証

```bash
# 簡易サーバー起動
npx serve .

# ブラウザで http://localhost:3000 を開く
# DevToolsのConsoleで結果確認
```

---

## 判定基準

### ✅ Go (本格実装に進む)

| 項目 | 基準 |
|------|------|
| YOLOv8 動作 | エラーなく推論完了 |
| PaddleOCR 動作 | エラーなく推論完了 |
| 推論速度 | 合計 < 500ms |
| メモリ | < 500MB |
| ロード時間 | < 10秒 (初回) |

### ❌ No-Go (代替案に切り替え)

以下の場合、アーキテクチャ再検討：

1. **未サポートオペレーターエラー**
   → モデル構造の変更 or 別フレームワーク

2. **推論速度 > 2秒**
   → WebGPU検討 or サーバーサイド推論

3. **メモリ > 1GB**
   → モデル量子化 or 分割ロード

---

## 代替案（No-Go時）

| 問題 | 代替案 |
|------|--------|
| ONNX Runtime WASMで動かない | **onnxruntime-web (WebGL/WebGPU)** を試す |
| PaddleOCRモデルが非互換 | **Tesseract.js** + OCR-B学習データ |
| YOLOv8が重すぎる | **TFLite.js** + MobileNetSSD |
| WASM全体が遅い | **サーバーサイド推論** (API化) |

### 代替案B: Tesseract.js (最も安全)

```javascript
// WASMが駄目な場合のフォールバック
import Tesseract from 'tesseract.js';

// OCR-B用学習データを使用
const result = await Tesseract.recognize(
  imageElement,
  'ocrb', // カスタム言語データ
  { 
    tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
  }
);
```

---

## 検証タイムライン

| 時間 | 作業 |
|------|------|
| 0-1h | 環境構築、ONNXモデル取得 |
| 1-2h | onnxruntime-webで動作確認 |
| 2-3h | 実画像での精度・速度測定 |
| 3-4h | 結果分析、Go/No-Go判定 |

**所要時間: 約半日**

---

## 検証結果テンプレート

```markdown
# WASM PoC 検証結果

## 環境
- ブラウザ: Chrome xxx
- OS: xxx
- デバイス: xxx

## 結果

### YOLOv8-nano
- 動作: ✅ / ❌
- ロード時間: xxxms
- 推論時間: xxxms
- メモリ: xxxMB
- エラー: (あれば)

### PaddleOCR mobile
- 動作: ✅ / ❌
- ロード時間: xxxms  
- 推論時間: xxxms
- メモリ: xxxMB
- エラー: (あれば)

## 判定
- [ ] Go: 本格実装に進む
- [ ] No-Go: 代替案に切り替え

## 次のアクション
-
```
