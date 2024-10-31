{
  stdenv,
  lib,
  cbrkit,
}:
let
  app = cbrkit.override {
    extraGroups = [ "docs" ];
    dontCheck = true;
  };
  env = app.dependencyEnv;
in
stdenv.mkDerivation {
  name = "cbrkit-docs";
  src = ./.;
  dontInstall = true;
  dontCheck = true;
  dontConfigure = true;
  buildPhase = ''
    mkdir -p "$out"

    {
      echo '```txt'
      COLUMNS=120 ${lib.getExe' env "cbrkit"} --help
      echo '```'
    } > ./cli.md

    # remove everyting before the first header
    sed -i '1,/^# /d' ./README.md

    ${lib.getExe' env "pdoc"} -d google -t pdoc-template --math \
      --logo https://raw.githubusercontent.com/wi2trier/cbrkit/main/assets/logo.png \
      -o "$out" ./cbrkit

    mkdir "$out/assets"
    cp -rf ./assets/**/{*.png,*.gif} "$out/assets/"
  '';
}
