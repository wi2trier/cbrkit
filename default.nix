{
  lib,
  stdenv,
  callPackage,
  fetchFromGitHub,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
  python3,
  onetbb,
  cacert,
  graphviz,
}:
let
  pdocRepo = fetchFromGitHub {
    owner = "mitmproxy";
    repo = "pdoc";
    tag = "v16.0.0";
    hash = "sha256-9amp6CWYIcniVfdlmPKYuRFR7B5JJtuMlOoDxpfvvJA=";
  };
  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
  projectOverlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };
  getCudaPkgs = attrs: lib.filter (name: lib.hasPrefix "nvidia-" name) (lib.attrNames attrs);
  cudaOverlay =
    final: prev:
    lib.genAttrs (getCudaPkgs prev) (
      name:
      prev.${name}.overrideAttrs (old: {
        autoPatchelfIgnoreMissingDeps = true;
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
        pygraphviz = {
          setuptools = [ ];
        };
        cbor = {
          setuptools = [ ];
        };
        warc3-wet-clueweb09 = {
          setuptools = [ ];
        };
      };
  packageOverlay =
    final: prev:
    lib.mapAttrs (name: value: prev.${name}.overrideAttrs value) {
      pygraphviz = old: {
        buildInputs = (old.buildInputs or [ ]) ++ [ graphviz ];
      };
      torch = old: {
        autoPatchelfIgnoreMissingDeps = true;
      };
      numba = old: {
        buildInputs = (old.buildInputs or [ ]) ++ [ onetbb ];
      };
      cbrkit = old: {
        meta = (old.meta or { }) // {
          mainProgram = "cbrkit";
          maintainers = with lib.maintainers; [ mirkolenz ];
          license = lib.licenses.mit;
          homepage = "https://github.com/wi2trier/cbrkit";
          description = "Customizable Case-Based Reasoning (CBR) toolkit for Python with a built-in API and CLI";
          platforms = with lib.platforms; darwin ++ linux;
        };
        passthru = lib.recursiveUpdate (old.passthru or { }) {
          tests.pytest = stdenv.mkDerivation {
            name = "${final.cbrkit.name}-pytest";
            inherit (final.cbrkit) src;
            nativeBuildInputs = [
              cacert
              (mkVenv "cbrkit-test-env" {
                cbrkit = [
                  "all"
                  "test"
                ];
              })
            ];
            dontConfigure = true;
            buildPhase = ''
              runHook preBuild
              export HOME=$(mktemp -d)
              export NUMBA_CACHE_DIR=$HOME/.numba_cache
              pytest --cov-report=html
              runHook postBuild
            '';
            installPhase = ''
              runHook preInstall
              mv htmlcov $out
              runHook postInstall
            '';
          };
          docs = stdenv.mkDerivation {
            name = "${final.cbrkit.name}-docs";
            inherit (final.cbrkit) src;
            nativeBuildInputs = [
              cacert
              (mkVenv "cbrkit-docs-env" {
                cbrkit = [
                  "all"
                  "docs"
                ];
              })
            ];
            dontConfigure = true;
            buildPhase = ''
              runHook preBuild

              typer ./src/cbrkit/cli.py utils docs --name cbrkit --output cli.md

              pdoc \
                -d google \
                -t ${pdocRepo}/examples/dark-mode \
                --math \
                --logo https://raw.githubusercontent.com/wi2trier/cbrkit/main/assets/logo.png \
                -o "$out" \
                ./src/cbrkit/api.py \
                ./src/cbrkit/cli.py \
                ./src/cbrkit

              runHook postBuild
            '';
            installPhase = ''
              runHook preInstall

              cp -rf ./assets "$out/assets"

              runHook postInstall
            '';
          };
        };
      };
    };
  baseSet = callPackage pyproject-nix.build.packages {
    python = python3;
  };
  pythonSet = baseSet.overrideScope (
    lib.composeManyExtensions [
      pyproject-build-systems.overlays.wheel
      projectOverlay
      cudaOverlay
      buildSystemOverlay
      packageOverlay
    ]
  );
  mkVenv =
    name: deps:
    (pythonSet.mkVirtualEnv name deps).overrideAttrs (_: {
      venvIgnoreCollisions = [ "${python3.sitePackages}/griffe/*" ];
    });
  inherit (callPackage pyproject-nix.build.util { }) mkApplication;
in
mkApplication {
  venv = mkVenv "cbrkit-env" { cbrkit = [ "all" ]; };
  package = pythonSet.cbrkit;
}
