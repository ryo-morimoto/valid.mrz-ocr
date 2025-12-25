/**
 * マルチスレッド WASM 対応サーバー
 *
 * SharedArrayBuffer を使用するために必要なヘッダーを設定:
 * - Cross-Origin-Opener-Policy: same-origin
 * - Cross-Origin-Embedder-Policy: require-corp
 */
import { createServer } from 'http';
import { readFile, stat } from 'fs/promises';
import { join, extname } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = join(__dirname, 'public');
const PORT = 3000;

// MIME タイプマッピング
const MIME_TYPES = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.onnx': 'application/octet-stream',
  '.wasm': 'application/wasm',
};

const server = createServer(async (req, res) => {
  // COOP/COEP ヘッダー設定（SharedArrayBuffer 有効化）
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');

  let filePath = join(PUBLIC_DIR, req.url === '/' ? 'index.html' : req.url);

  try {
    const stats = await stat(filePath);
    if (stats.isDirectory()) {
      filePath = join(filePath, 'index.html');
    }

    const content = await readFile(filePath);
    const ext = extname(filePath);
    const mimeType = MIME_TYPES[ext] || 'application/octet-stream';

    res.writeHead(200, { 'Content-Type': mimeType });
    res.end(content);

    console.log(`200 ${req.url}`);
  } catch (err) {
    res.writeHead(404);
    res.end('Not Found');
    console.log(`404 ${req.url}`);
  }
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`\n🚀 Server running at http://0.0.0.0:${PORT}`);
  console.log('   COOP/COEP headers enabled (SharedArrayBuffer available)\n');
});
