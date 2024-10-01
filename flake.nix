{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
    flocken = {
      url = "github:mirkolenz/flocken/v2";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs =
    inputs@{
      self,
      nixpkgs,
      flake-parts,
      systems,
      flocken,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.treefmt-nix.flakeModule
      ];
      systems = import systems;
      perSystem =
        {
          pkgs,
          system,
          lib,
          config,
          ...
        }:
        let
          python = pkgs.python311;
          poetry = pkgs.poetry;
          mkPoetryApp =
            args:
            pkgs.poetry2nix.mkPoetryApplication (
              {
                inherit python;
                projectDir = ./.;
                preferWheels = true;
                checkPhase = "pytest";
                meta = {
                  description = "Customizable Case-Based Reasoning (CBR) toolkit for Python with a built-in API and CLI.";
                  license = lib.licenses.mit;
                  maintainers = with lib.maintainers; [ mirkolenz ];
                  platforms = with lib.platforms; darwin ++ linux;
                  homepage = "https://github.com/wi2trier/cbrkit";
                  mainProgram = "cbrkit";
                };
              }
              // args
            );
        in
        {
          _module.args.pkgs = import nixpkgs {
            inherit system;
            overlays = [ inputs.poetry2nix.overlays.default ];
          };
          checks = {
            inherit (config.packages) cbrkit;
          };
          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              ruff-check.enable = true;
              ruff-format.enable = true;
              nixfmt.enable = true;
            };
          };
          packages = {
            default = config.packages.cbrkit;
            cbrkit = mkPoetryApp { };
            docker = pkgs.dockerTools.buildLayeredImage {
              name = "cbrkit";
              tag = "latest";
              created = "now";
              config = {
                entrypoint = [ (lib.getExe config.packages.default) ];
                cmd = [ ];
              };
            };
            releaseEnv = pkgs.buildEnv {
              name = "release-env";
              paths = [
                python
                poetry
              ];
            };
            docs =
              let
                app = mkPoetryApp { groups = [ "docs" ]; };
                env = app.dependencyEnv;
              in
              pkgs.stdenv.mkDerivation {
                name = "docs";
                src = ./.;
                buildPhase = ''
                  mkdir -p "$out"

                  {
                    echo '```txt'
                    COLUMNS=120 ${lib.getExe app} --help
                    echo '```'
                  } > ./cli.md

                  # remove everyting before the first header
                  ${lib.getExe pkgs.gnused} -i '1,/^# /d' ./README.md

                  ${lib.getExe' env "pdoc"} -d google -t pdoc-template --math \
                    --logo https://raw.githubusercontent.com/wi2trier/cbrkit/main/assets/logo.png \
                    -o "$out" ./cbrkit

                  mkdir "$out/assets"
                  cp -rf ./assets/**/{*.png,*.gif} "$out/assets/"
                '';
                dontInstall = true;
              };
          };
          legacyPackages.dockerManifest = flocken.legacyPackages.${system}.mkDockerManifest {
            github = {
              enable = true;
              token = "$GH_TOKEN";
            };
            version = builtins.getEnv "VERSION";
            images = with self.packages; [ x86_64-linux.docker ];
          };
          devShells.default = pkgs.mkShell {
            packages = [
              python
              poetry
              config.treefmt.build.wrapper
            ];
            POETRY_VIRTUALENVS_IN_PROJECT = true;
            shellHook = ''
              ${lib.getExe poetry} env use ${lib.getExe python}
              ${lib.getExe poetry} install --all-extras --no-root --sync
            '';
          };
        };
    };
}
