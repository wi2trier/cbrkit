{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
    flocken = {
      url = "github:mirkolenz/flocken/v2";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
      "https://wi2trier.cachix.org"
      "https://pyproject-nix.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "wi2trier.cachix.org-1:8wJvKtRD8XUqYZMdjECTsN1zWxHy9kvp5aoPQiAm1fY="
      "pyproject-nix.cachix.org-1:UNzugsOlQIu2iOz0VyZNBQm2JSrL/kwxeCcFGw+jMe0="
    ];
  };
  outputs =
    inputs@{
      self,
      nixpkgs,
      flake-parts,
      systems,
      flocken,
      uv2nix,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import systems;
      imports = with inputs; [
        flake-parts.flakeModules.easyOverlay
        treefmt-nix.flakeModule
      ];
      perSystem =
        {
          pkgs,
          system,
          lib,
          config,
          ...
        }:
        let
          inherit
            (pkgs.callPackage ./default.nix {
              inherit (inputs) uv2nix pyproject-nix pyproject-build-systems;
            })
            pythonSet
            mkApplication
            ;
        in
        {
          _module.args.pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
            };
            overlays = lib.singleton (
              final: prev: {
                uv = uv2nix.packages.${system}.uv-bin;
              }
            );
          };
          overlayAttrs = {
            inherit (config.packages) cbrkit;
          };
          checks = pythonSet.cbrkit.passthru.tests // {
            inherit (pythonSet.cbrkit.passthru) docs;
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
            inherit (pythonSet.cbrkit.passthru) docs;
            default = config.packages.cbrkit;
            cbrkit = mkApplication {
              venv = pythonSet.mkVirtualEnv "cbrkit-env" {
                cbrkit = [ "all" ];
              };
              package = pythonSet.cbrkit;
            };
            docker = pkgs.dockerTools.streamLayeredImage {
              name = "cbrkit";
              tag = "latest";
              created = "now";
              config.Entrypoint = [ (lib.getExe config.packages.cbrkit) ];
            };
            release-env = pkgs.buildEnv {
              name = "release-env";
              paths = with pkgs; [
                uv
                python3
              ];
            };
          };
          legacyPackages.docker-manifest = flocken.legacyPackages.${system}.mkDockerManifest {
            github = {
              enable = true;
              token = "$GH_TOKEN";
            };
            version = builtins.getEnv "VERSION";
            imageStreams = with self.packages; [ x86_64-linux.docker ];
          };
          devShells.default = pkgs.mkShell {
            packages = with pkgs; [
              uv
              config.treefmt.build.wrapper
              graphviz
            ];
            nativeBuildInputs = with pkgs; [ zlib ];
            LD_LIBRARY_PATH = lib.makeLibraryPath [
              pkgs.stdenv.cc.cc
              pkgs.zlib
              "/run/opengl-driver"
            ];
            UV_PYTHON = lib.getExe pkgs.python312;
            TOKENIZERS_PARALLELISM = true;
            shellHook = ''
              uv sync --all-extras --locked
            '';
          };
        };
    };
}
