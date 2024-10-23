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
        {
          _module.args.pkgs = import nixpkgs {
            inherit system;
            overlays = [
              inputs.poetry2nix.overlays.default
              (final: prev: {
                python3 = final.python312;
              })
            ];
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
            cbrkit = pkgs.callPackage ./default.nix { };
            docs = pkgs.callPackage ./docs.nix { inherit (config.packages) cbrkit; };
            docker = pkgs.dockerTools.buildLayeredImage {
              name = "cbrkit";
              tag = "latest";
              created = "now";
              config.Entrypoint = [ (lib.getExe config.packages.default) ];
            };
            release-env = pkgs.buildEnv {
              name = "release-env";
              paths = with pkgs; [
                python3
                poetry
              ];
            };
          };
          apps.docker-manifest.program = flocken.legacyPackages.${system}.mkDockerManifest {
            github = {
              enable = true;
              token = "$GH_TOKEN";
            };
            version = builtins.getEnv "VERSION";
            images = with self.packages; [ x86_64-linux.docker ];
          };
          devShells.default = pkgs.mkShell rec {
            buildInputs = with pkgs; [
              stdenv.cc.cc
              zlib
            ];
            packages = with pkgs; [
              python3
              poetry
              config.treefmt.build.wrapper
            ];
            POETRY_VIRTUALENVS_IN_PROJECT = true;
            LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
            shellHook = ''
              ${lib.getExe pkgs.poetry} env use ${lib.getExe pkgs.python3}
              ${lib.getExe pkgs.poetry} install --all-extras --no-root --sync
            '';
          };
        };
    };
}
