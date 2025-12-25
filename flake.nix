{
  description = "MRZ OCR Validation - Root (direnv用)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  # ルートは最小限の環境のみ提供
  # 各サブディレクトリに cd して direnv allow で個別環境を有効化
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.git
          ];

          shellHook = ''
            echo "=== MRZ OCR Validation ==="
            echo ""
            echo "サブディレクトリに移動して環境を有効化:"
            echo "  cd paddleocr-validation && direnv allow"
            echo "  cd wasm-poc && direnv allow"
            echo "  cd tesseract-ft && direnv allow"
            echo "  cd crnn && direnv allow"
            echo "=========================="
          '';
        };
      }
    );
}
