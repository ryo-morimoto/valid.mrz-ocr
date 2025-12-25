{
  description = "PaddleOCR MRZ Validation Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Python環境 (uv で依存関係を管理するため最小限)
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          pip
          virtualenv
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.uv

            # OpenCV/PaddleOCR が必要とするシステムライブラリ
            pkgs.libGL
            pkgs.glib
          ];

          # 共有ライブラリパス (opencv-python, paddlepaddle に必要)
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.libGL
            pkgs.glib
          ];

          shellHook = ''
            echo "=== PaddleOCR MRZ Validation ==="
            echo "Python: $(python --version)"
            echo "uv: $(uv --version)"
            echo ""
            echo "Setup:"
            echo "  uv sync"
            echo "  uv run python quickstart.py"
            echo "================================"
          '';
        };
      }
    );
}
