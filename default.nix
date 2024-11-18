{
  lib,
  stdenv,
  callPackage,
  uv2nix,
  pyproject-nix,
  python3,
  tbb_2021_11,
}:
let
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
        buildInputs = (old.buildInputs or [ ]) ++ [ tbb_2021_11 ];
      };
      cbrkit = old: {
        passthru = lib.recursiveUpdate (old.passthru or { }) {
          tests.pytest = stdenv.mkDerivation {
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
          docs = stdenv.mkDerivation {
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
  baseSet = callPackage pyproject-nix.build.packages {
    python = python3;
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
pythonSet
// {
  mkApp = depsName: addMeta (pythonSet.mkVirtualEnv "cbrkit-env" workspace.deps.${depsName});
}
