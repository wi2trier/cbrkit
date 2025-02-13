{
  lib,
  stdenv,
  callPackage,
  fetchFromGitHub,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
  python3,
  tbb_2021_11,
  cacert,
}:
let
  pdocRepo = fetchFromGitHub {
    owner = "mitmproxy";
    repo = "pdoc";
    tag = "v15.0.1";
    hash = "sha256-HDrDGnK557EWbBQtsvDzTst3oV0NjLRm4ilXaxd6/j8=";
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
      torch = old: {
        autoPatchelfIgnoreMissingDeps = true;
      };
      numba = old: {
        buildInputs = (old.buildInputs or [ ]) ++ [ tbb_2021_11 ];
      };
      cbrkit = old: {
        meta = (old.meta or { }) // {
          mainProgram = "cbrkit";
          maintainers = with lib.maintainers; [ mirkolenz ];
          license = lib.licenses.mit;
          homepage = "https://github.com/wi2trier/cbrkit";
          description = "Generate entire directory structures using Jinja templates with support for external data and custom plugins.";
          platforms = with lib.platforms; darwin ++ linux;
        };
        passthru = lib.recursiveUpdate (old.passthru or { }) {
          tests.pytest = stdenv.mkDerivation {
            name = "${final.cbrkit.name}-pytest";
            inherit (final.cbrkit) src;
            nativeBuildInputs = [
              cacert
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
          docs = stdenv.mkDerivation {
            name = "${final.cbrkit.name}-docs";
            inherit (final.cbrkit) src;
            nativeBuildInputs = [
              cacert
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

              mkdir -p "$out/assets"
              cp -rf ./assets/**/{*.png,*.gif} "$out/assets"

              runHook postInstall
            '';
          };
        };
      };
    };
  baseSet = callPackage pyproject-nix.build.packages {
    python = python3;
  };
in
{
  inherit workspace;
  inherit (callPackage pyproject-nix.build.util { }) mkApplication;
  pythonSet = baseSet.overrideScope (
    lib.composeManyExtensions [
      pyproject-build-systems.overlays.default
      projectOverlay
      cudaOverlay
      buildSystemOverlay
      packageOverlay
    ]
  );
}
