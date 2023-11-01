{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.05";
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
        packages = {
          default = pkgs.poetry2nix.mkPoetryApplication {
            inherit python;
            projectDir = ./.;
            preferWheels = true;
          };
          cbrkit = self'.packages.default;
          releaseEnv = pkgs.buildEnv {
            name = "release-env";
            paths = packages;
          };
        };
        devShells.default = pkgs.mkShell {
          inherit packages;
          POETRY_VIRTUALENVS_IN_PROJECT = true;
          shellHook = ''
            ${lib.getExe poetry} env use ${lib.getExe python}
            ${lib.getExe poetry} install --all-extras --no-root
          '';
        };
      };
    };
}
