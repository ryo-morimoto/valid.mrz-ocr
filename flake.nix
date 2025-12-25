{
  description = "MRZ OCR Validation Environment";

  inputs = {
    # 25.11 を使用 (ultralytics, paddle2onnx が必要)
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Python環境の構築
        # Python 3.12 + ultralytics (YOLOv8)
        # OCR は Tesseract.js をブラウザで使用
        pythonEnv = pkgs.python312.withPackages (ps: with ps; [
          pip
          numpy
          opencv4  # cv2
          pillow
          tqdm
          requests
          rapidfuzz

          # WASM PoC 用 (ONNX エクスポート)
          ultralytics    # YOLOv8
          onnx           # ONNX形式エクスポート

          # OCR 検証用
          easyocr        # CRNN ベース OCR

          # CRNN 学習用
          torch          # PyTorch
          torchvision    # 画像処理
        ]);
      in
      {
        # デフォルト: Python + Node.js 両方使える環境
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.tesseract
            pkgs.uv

            # WASM PoC 用
            pkgs.nodejs_20
            pkgs.pnpm
          ];

          # 共有ライブラリパス設定
          # opencv-python, PyTorch (ultralytics) 等に必要
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.libGL
            pkgs.glib
          ];

          shellHook = ''
            echo "=== MRZ OCR Validation Environment ==="
            echo "Python: $(python --version)"
            echo "Node.js: $(node --version)"
            echo ""
            echo "Projects:"
            echo "  paddleocr-validation/  # PaddleOCR検証"
            echo "  wasm-poc/              # WASM OCR検証"
            echo ""
            echo "Commands:"
            echo "  cd paddleocr-validation && uv run python quickstart.py"
            echo "  cd wasm-poc && pnpm run serve"
            echo "======================================="
          '';
        };

        # WASM PoC 専用シェル (軽量)
        devShells.wasm = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.nodejs_20
            pkgs.pnpm
            pkgs.uv
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.libGL
            pkgs.glib
          ];

          shellHook = ''
            echo "=== WASM PoC Environment ==="
            echo "Python: $(python --version)"
            echo "Node.js: $(node --version)"
            echo ""
            echo "Setup:"
            echo "  cd wasm-poc"
            echo "  python scripts/export_yolo.py"
            echo "  python scripts/export_ppocr.py"
            echo "  pnpm install && pnpm run serve"
            echo "============================="
          '';
        };
      }
    );
}
