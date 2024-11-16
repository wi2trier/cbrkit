{
  poetry2nix,
  python3,
  lib,
  dontCheck ? false,
  extraGroups ? [ ],
}:
let
  injectBuildInputs =
    attrs: final: prev:
    lib.mapAttrs (
      name: value:
      prev.${name}.overridePythonAttrs (old: {
        buildInputs = (old.buildInputs or [ ]) ++ (map (v: prev.${v}) value);
      })
    ) attrs;
in
poetry2nix.mkPoetryApplication {
  inherit dontCheck;
  projectDir = ./.;
  preferWheels = true;
  groups = [ "main" ] ++ extraGroups;
  nativeCheckInputs = with python3.pkgs; [
    pytestCheckHook
    pytest-cov-stub
  ];
  overrides = poetry2nix.overrides.withDefaults (injectBuildInputs {
    warc3-wet-clueweb09 = [ "setuptools" ];
    dtaidistance = [ "setuptools" ];
  });
  meta = with lib; {
    description = "Customizable Case-Based Reasoning (CBR) toolkit for Python with a built-in API and CLI.";
    license = licenses.mit;
    maintainers = with maintainers; [ mirkolenz ];
    platforms = with platforms; darwin ++ linux;
    homepage = "https://github.com/wi2trier/cbrkit";
    mainProgram = "cbrkit";
  };
}
