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
  };
  outputs = inputs @ {
    self,
    nixpkgs,
    flake-parts,
    systems,
    flocken,
    poetry2nix,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = import systems;
      perSystem = {
        pkgs,
        system,
        lib,
        self',
        ...
      }: let
        python = pkgs.python311;
        poetry = pkgs.poetry;
        packages = [python poetry];
      in {
        _module.args.pkgs = import nixpkgs {
          inherit system;
          overlays = [poetry2nix.overlays.default];
        };
        checks = {
          inherit (self'.packages) cbrkit;
        };
        packages = {
          default = pkgs.poetry2nix.mkPoetryApplication {
            inherit python;
            projectDir = ./.;
            preferWheels = true;
            checkPhase = "pytest";
            extras = [];
          };
          cbrkit = self'.packages.default;
          docker = pkgs.dockerTools.buildLayeredImage {
            name = "cbrkit";
            tag = "latest";
            created = "now";
            config = {
              entrypoint = [(lib.getExe self'.packages.default)];
              cmd = [];
            };
          };
          releaseEnv = pkgs.buildEnv {
            name = "release-env";
            paths = packages;
          };
          docs = let
            app = pkgs.poetry2nix.mkPoetryApplication {
              inherit python;
              projectDir = ./.;
              preferWheels = true;
              checkPhase = "pytest";
              extras = ["cli"];
              groups = ["docs"];
            };
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
            token = builtins.getEnv "GH_TOKEN";
          };
          version = builtins.getEnv "VERSION";
          images = with self.packages; [x86_64-linux.docker aarch64-linux.docker];
        };
        devShells.default = pkgs.mkShell {
          inherit packages;
          POETRY_VIRTUALENVS_IN_PROJECT = true;
          shellHook = ''
            ${lib.getExe poetry} env use ${lib.getExe python}
            ${lib.getExe poetry} install --all-extras --no-root --sync
          '';
        };
      };
    };
}
