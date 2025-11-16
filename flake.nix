{
  description = "Flask ML API environment with sklearn, pydub, etc.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = {flake-parts, ...} @ inputs:
    flake-parts.lib.mkFlake {inherit inputs;}
    {
      systems = ["x86_64-linux" "aarch64-linux"];
      perSystem = {
        config,
        pkgs,
        ...
      }: let
        python = pkgs.python312Packages;

        # Define flask_cors package
        flask_cors = python.buildPythonPackage {
          pname = "flask_cors";
          version = "5.0.0"; # Specify the desired version
          format = "wheel";

          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/56/07/1afa0514c876282bebc1c9aee83c6bb98fe6415cf57b88d9b06e7e29bf9c/Flask_Cors-5.0.0-py2.py3-none-any.whl";
            sha256 = "b9e307d082a9261c100d8fb0ba909eec6a228ed1b60a8315fd85f783d61910bc";
          };
          meta = with pkgs.lib; {
            description = "A Flask extension for handling Cross Origin Resource Sharing (CORS), making cross-origin AJAX possible.";
            homepage = "https://github.com/corydolphin/flask-cors";
            license = licenses.mit;
          };
        };

        pythonEnv = pkgs.python312.withPackages (ps:
          with ps; [
            gunicorn
            flask
            flask_cors
            librosa
            pydub
            scikit-learn
            matplotlib
          ]);
      in {
        devShells.default = pkgs.mkShell {
          name = "flask-ml-env";

          packages = [
            pythonEnv
          ];

          buildInputs = with pkgs; [
            ffmpeg # for pydub audio processing
          ];

          shellHook = ''
            echo "Flask server environment loaded."
            echo "Run in dev mode: flask --app server run"
            echo "Run in dev mode and allow all hosts: flask --app server run --host=0.0.0.0"
          '';
        };

        packages.default = let
          inputs = [pythonEnv pkgs.ffmpeg];

          sleepspec-server-pkg = pkgs.stdenvNoCC.mkDerivation {
            name = "sleepspec-server-pkg";
            src = ./.;
            buildInputs = inputs;
            installPhase = ''
              mkdir -p $out
              cp -r ./* $out
            '';
          };
        in
          pkgs.writeShellApplication {
            name = "sleepspec-server";
            runtimeInputs = inputs;
            text = ''
              cd ${sleepspec-server-pkg}
              gunicorn -t "''${TIMEOUT:-300}" -w "''${WORKERS:-4}" server:app --bind "''${HOST:-localhost}":"''${PORT:-5000}"
            '';
          };
      };
    };
}
