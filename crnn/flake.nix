{
  description = "CRNN MRZ OCR Training Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          # CUDA 関連パッケージのみ unfree 許可
          config.allowUnfreePredicate = pkg:
            builtins.any (prefix: pkgs.lib.hasPrefix prefix (pkgs.lib.getName pkg)) [
              "cuda"
              "cudnn"
              "libcu"
              "libnv"
              "libnpp"  # NVIDIA Performance Primitives
              "torch"
            ];
        };

        # CUDA 対応 Python 環境
        pythonEnv = pkgs.python312.withPackages (ps: with ps; [
          # 学習用
          torchWithCuda
          torchvision

          # データ処理
          numpy
          opencv4
          pillow
          tqdm

          # ONNX エクスポート
          onnx
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
          ];

          # CUDA ライブラリパス
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.libGL
            pkgs.glib
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn
          ];

          shellHook = ''
            echo "=== CRNN Training Environment ==="
            echo "Python: $(python --version)"
            echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
            echo ""
            echo "Usage:"
            echo "  python scripts/train.py      # 学習"
            echo "  python scripts/export_onnx.py # ONNX エクスポート"
            echo "  python scripts/validate.py   # 精度検証"
            echo "================================="
          '';
        };

        # CPU のみ環境（CUDA ビルド不要で高速）
        devShells.cpu = pkgs.mkShell {
          buildInputs = [
            (pkgs.python312.withPackages (ps: with ps; [
              torch
              torchvision
              numpy
              opencv4
              pillow
              tqdm
              onnx
            ]))
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.libGL
            pkgs.glib
          ];

          shellHook = ''
            echo "=== CRNN Training Environment (CPU) ==="
            echo "Python: $(python --version)"
            echo ""
            echo "Usage:"
            echo "  python scripts/train.py"
            echo "========================================"
          '';
        };
      }
    );
}
