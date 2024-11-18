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
      url = "github:nix-community/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:adisbladis/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };
  outputs =
    inputs@{
      self,
      nixpkgs,
      flake-parts,
      systems,
      flocken,
      pyproject-nix,
      uv2nix,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import systems;
      imports = [
        inputs.flake-parts.flakeModules.easyOverlay
        inputs.treefmt-nix.flakeModule
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
          workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
          projectOverlay = workspace.mkPyprojectOverlay {
            sourcePreference = "wheel";
          };
          cudaOverlay =
            final: prev:
            lib.genAttrs
              # uv.lock -> torch -> nvidia deps
              [
                "nvidia-cublas-cu12"
                "nvidia-cuda-cupti-cu12"
                "nvidia-cuda-nvrtc-cu12"
                "nvidia-cuda-runtime-cu12"
                "nvidia-cudnn-cu12"
                "nvidia-cufft-cu12"
                "nvidia-curand-cu12"
                "nvidia-cusolver-cu12"
                "nvidia-cusparse-cu12"
                "nvidia-nccl-cu12"
                "nvidia-nvjitlink-cu12"
                "nvidia-nvtx-cu12"
              ]
              (
                name:
                prev.${name}.overrideAttrs (old: {
                  autoPatchelfIgnoreMissingDeps = true;
                  dontUsePyprojectBytecode = true;
                })
              );
          buildSystemOverlay =
            final: prev:
            lib.mapAttrs
              (
                name: value:
                prev.${name}.overrideAttrs (old: {
                  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ (final.resolveBuildSystem value);
                })
              )
              {
                cbor = {
                  setuptools = [ ];
                };
                dtaidistance = {
                  setuptools = [ ];
                  cython = [ ];
                  numpy = [ ];
                  wheel = [ ];
                };
                warc3-wet-clueweb09 = {
                  setuptools = [ ];
                };
              };
          packageOverlay =
            final: prev:
            lib.mapAttrs (name: value: prev.${name}.overrideAttrs value) {
              torch = old: {
                autoPatchelfIgnoreMissingDeps = true;
              };
              numba = old: {
                buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.tbb_2021_11 ];
              };
              cbrkit = old: {
                passthru = lib.recursiveUpdate (old.passthru or { }) {
                  tests.pytest = pkgs.stdenv.mkDerivation {
                    name = "${final.cbrkit.name}-pytest";
                    inherit (final.cbrkit) src;
                    nativeBuildInputs = [
                      (final.mkVirtualEnv "cbrkit-test-env" {
                        cbrkit = [
                          "all"
                          "test"
                        ];
                      })
                    ];
                    dontConfigure = true;
                    buildPhase = ''
                      runHook preBuild
                      pytest --cov-report=html
                      runHook postBuild
                    '';
                    installPhase = ''
                      runHook preInstall
                      mv htmlcov $out
                      runHook postInstall
                    '';
                  };
                  docs = pkgs.stdenv.mkDerivation {
                    name = "${final.cbrkit.name}-docs";
                    inherit (final.cbrkit) src;
                    nativeBuildInputs = [
                      (final.mkVirtualEnv "cbrkit-docs-env" {
                        cbrkit = [
                          "all"
                          "docs"
                        ];
                      })
                    ];
                    dontConfigure = true;
                    buildPhase = ''
                      runHook preBuild

                      pdoc \
                        -d google \
                        -t pdoc-template \
                        --math \
                        --logo https://raw.githubusercontent.com/wi2trier/cbrkit/main/assets/logo.png \
                        -o "$out" \
                        ./src/cbrkit

                      runHook postBuild
                    '';
                    installPhase = ''
                      runHook preInstall

                      mkdir -p "$out/assets"
                      cp -rf ./assets/**/{*.png,*.gif} "$out/assets"

                      runHook postInstall
                    '';
                  };
                };
              };
            };
          baseSet = pkgs.callPackage pyproject-nix.build.packages {
            python = pkgs.python3;
          };
          pythonSet = baseSet.overrideScope (
            lib.composeManyExtensions [
              projectOverlay
              cudaOverlay
              buildSystemOverlay
              packageOverlay
            ]
          );
          addMeta =
            drv:
            drv.overrideAttrs (old: {
              passthru = lib.recursiveUpdate (old.passthru or { }) {
                inherit (pythonSet.cbrkit.passthru) tests;
              };
              meta = (old.meta or { }) // {
                mainProgram = "cbrkit";
                maintainers = with lib.maintainers; [ mirkolenz ];
                license = lib.licenses.mit;
                homepage = "https://github.com/wi2trier/cbrkit";
                description = "Generate entire directory structures using Jinja templates with support for external data and custom plugins.";
                platforms = with lib.platforms; darwin ++ linux;
              };
            });
        in
        {
          _module.args.pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
            };
            overlays = lib.singleton (
              final: prev: {
                python3 = final.python312;
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
            cbrkit = addMeta (pythonSet.mkVirtualEnv "cbrkit-env" workspace.deps.optionals);
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
          apps.docker-manifest.program = flocken.legacyPackages.${system}.mkDockerManifest {
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
            ];
            nativeBuildInputs = with pkgs; [ zlib ];
            LD_LIBRARY_PATH = lib.makeLibraryPath [
              pkgs.stdenv.cc.cc
              pkgs.zlib
              "/run/opengl-driver"
            ];
            UV_PYTHON = lib.getExe pkgs.python3;
            shellHook = ''
              uv sync --all-extras --locked
            '';
          };
        };
    };
}
