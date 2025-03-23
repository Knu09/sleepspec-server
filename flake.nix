{
  description = "Flask ML API environment with sklearn, pydub, etc.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          flask
          pydub
          numpy
          scipy
          scikit-learn
        ]);
      in {
        devShells.default = pkgs.mkShell {
          name = "flask-ml-env";

          packages = [
            pythonEnv
            pkgs.ffmpeg     # for pydub audio processing
            pkgs.git        # if you're pulling anything
          ];

          shellHook = ''
            echo "Flask ML environment loaded."
            echo "Run your Flask app with: python your_app.py"
          '';
        };
      });
}
