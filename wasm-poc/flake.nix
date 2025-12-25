{
  description = "WASM PoC - YOLOv8 + Tesseract.js Browser OCR";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Python環境 (YOLOv8 → ONNX エクスポート用)
        pythonEnv = pkgs.python312.withPackages (ps: with ps; [
          pip
          numpy
          pillow
          ultralytics  # YOLOv8
          onnx         # ONNXエクスポート
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Node.js 環境
            pkgs.nodejs_20
            pkgs.pnpm

            # Python (モデルエクスポート用)
            pythonEnv
          ];

          # ultralytics (PyTorch) に必要
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.libGL
            pkgs.glib
          ];

          shellHook = ''
            echo "=== WASM PoC Environment ==="
            echo "Node.js: $(node --version)"
            echo "pnpm: $(pnpm --version)"
            echo "Python: $(python --version)"
            echo ""
            echo "Setup:"
            echo "  pnpm install"
            echo "  python scripts/export_yolo.py"
            echo "  pnpm run serve"
            echo "============================="
          '';
        };
      }
    );
}
