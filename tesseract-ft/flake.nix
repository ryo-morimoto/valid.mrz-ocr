{
  description = "Tesseract Fine-tuning for OCR-B/MRZ";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Python環境 (画像処理 + 学習データ生成)
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          numpy
          opencv4      # cv2
          pillow
          tqdm
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv

            # Tesseract (トレーニングツール含む)
            pkgs.tesseract
          ];

          # OpenCV に必要
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.libGL
            pkgs.glib
          ];

          shellHook = ''
            echo "=== Tesseract Fine-tuning ==="
            echo "Python: $(python --version)"
            echo "Tesseract: $(tesseract --version 2>&1 | head -1)"
            echo ""
            echo "Usage:"
            echo "  python scripts/generate_training.py"
            echo "  python scripts/validate.py"
            echo "=============================="
          '';
        };
      }
    );
}
