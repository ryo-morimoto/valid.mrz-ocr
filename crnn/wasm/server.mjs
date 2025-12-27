/**
 * COOP/COEP ヘッダー付きローカルサーバー
 *
 * SharedArrayBuffer を有効化するために必要。
 * ONNX Runtime Web のマルチスレッド WASM で高速化される。
 */

import { createServer } from 'http';
import { readFile, stat } from 'fs/promises';
import { extname, join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const PORT = 3001;

const MIME_TYPES = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
};

const server = createServer(async (req, res) => {
  // COOP/COEP ヘッダー（SharedArrayBuffer 有効化に必須）
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');

  let filePath = req.url === '/' ? '/index.html' : req.url;

  // 静的ファイル配信（models/ も同じディレクトリから）
  filePath = join(__dirname, filePath);

  try {
    const stats = await stat(filePath);
    if (!stats.isFile()) {
      res.writeHead(404);
      res.end('Not Found');
      return;
    }

    const ext = extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';

    const content = await readFile(filePath);
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(content);

    console.log(`200 ${req.url}`);
  } catch (err) {
    if (err.code === 'ENOENT') {
      res.writeHead(404);
      res.end('Not Found');
      console.log(`404 ${req.url}`);
    } else {
      res.writeHead(500);
      res.end('Internal Server Error');
      console.error(err);
    }
  }
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 CRNN WASM Test Server running at http://localhost:${PORT}`);
  console.log('   COOP/COEP headers enabled (SharedArrayBuffer available)');
  console.log('');
  console.log('   Open in browser to run speed tests');
});
