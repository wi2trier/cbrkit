# Changelog

## [0.14.2](https://github.com/wi2trier/cbrkit/compare/v0.14.1...v0.14.2) (2024-10-02)


### Bug Fixes

* **retrieval:** use starmap for parallel execution ([257d941](https://github.com/wi2trier/cbrkit/commit/257d941b2cd071d1b6f31b94a54b1755f4e55291))

## [0.14.1](https://github.com/wi2trier/cbrkit/compare/v0.14.0...v0.14.1) (2024-10-02)


### Bug Fixes

* **cli:** add parallel strategy enum for typer ([8647097](https://github.com/wi2trier/cbrkit/commit/8647097f189714d0a7a4dbd4d7aa46d61d7b9887))

## [0.14.0](https://github.com/wi2trier/cbrkit/compare/v0.13.1...v0.14.0) (2024-10-02)


### Features

* add parallelization to api and cli ([f46f1f4](https://github.com/wi2trier/cbrkit/commit/f46f1f46729921c0c38581d62a27ee20a46fe1a3))
* allow multiple processes for single queries ([45acf0e](https://github.com/wi2trier/cbrkit/commit/45acf0e3bc6f333959ecd7abdad511a752b0fb6b))


### Bug Fixes

* use multiprocess from pathos for mapply ([2cf11f6](https://github.com/wi2trier/cbrkit/commit/2cf11f67acde8075bd7e39f6697917b27ed0c855))

## [0.13.1](https://github.com/wi2trier/cbrkit/compare/v0.13.0...v0.13.1) (2024-10-01)


### Reverts

* replace treefmt with git-hooks ([97d6a3e](https://github.com/wi2trier/cbrkit/commit/97d6a3e8e0f86f8e47965eab9cb0c1dbbee66aba))

## [0.13.0](https://github.com/wi2trier/cbrkit/compare/v0.12.3...v0.13.0) (2024-09-05)


### Features

* **cli:** add json serialization option ([4ec0446](https://github.com/wi2trier/cbrkit/commit/4ec0446d5ad27a1e8b11ccbd8f13b246f629becf))
* introduce `mapply` to run multiple queries ([0c1b1da](https://github.com/wi2trier/cbrkit/commit/0c1b1dac23e93831125d2252a107fab6bd387eff))
* **retrieval:** add `mapply` function for multiple queries ([1619d5c](https://github.com/wi2trier/cbrkit/commit/1619d5c1cda9b1c8bffb0843099b6c66352bde69))
* **sim:** add linear interval function for numbers ([ef5e1fd](https://github.com/wi2trier/cbrkit/commit/ef5e1fde2b7b003a9a3ac27cafc7756ca37b6af2))


### Bug Fixes

* **aggregator:** apply pooling factor for custom weights ([aee427c](https://github.com/wi2trier/cbrkit/commit/aee427cbd32d88fcc79d414f773f3768a0355b6d))
* rename single/intermediate results ([3796133](https://github.com/wi2trier/cbrkit/commit/37961336071eed9218045e2835667d52cef5cc07))
* **sim:** add missing collection functions to __all__ ([a6f6ee7](https://github.com/wi2trier/cbrkit/commit/a6f6ee7fe4cc75d7e4c643ce1e0386b3899fe43a))

## [0.12.3](https://github.com/wi2trier/cbrkit/compare/v0.12.2...v0.12.3) (2024-08-09)


### Bug Fixes

* **api:** add support for response models ([6377c48](https://github.com/wi2trier/cbrkit/commit/6377c481b587200ae7b539af40bf0877eb85e474))

## [0.12.2](https://github.com/wi2trier/cbrkit/compare/v0.12.1...v0.12.2) (2024-07-24)


### Bug Fixes

* **api:** allow overriding more params ([443afab](https://github.com/wi2trier/cbrkit/commit/443afabaef54ddf809a6d27527238ca044bd422c))
* **api:** allow passing multiple retrievers ([2328763](https://github.com/wi2trier/cbrkit/commit/2328763002f2beaace616c9e75c9ffb6e0c1b78a))
* **api:** allow specifying the search path ([ab0802e](https://github.com/wi2trier/cbrkit/commit/ab0802e337f18bcda6d2a277ff8b66dc33fe7d17))
* **cli:** allow silent retrieval ([02d0b3a](https://github.com/wi2trier/cbrkit/commit/02d0b3a57490b43898957668580bb606828ddb38))

## [0.12.1](https://github.com/wi2trier/cbrkit/compare/v0.12.0...v0.12.1) (2024-07-16)


### Bug Fixes

* use "is" for type comparisons ([2a9580a](https://github.com/wi2trier/cbrkit/commit/2a9580a1aa55f1c05ca60b9afdff2618f36fc890))
* use casting for certain argument types ([5b3ebf3](https://github.com/wi2trier/cbrkit/commit/5b3ebf378a536c769b91ae5b1294ae47db06f63d))

## [0.12.0](https://github.com/wi2trier/cbrkit/compare/v0.11.1...v0.12.0) (2024-07-12)


### Features

* **cli:** add serve command ([e397c2a](https://github.com/wi2trier/cbrkit/commit/e397c2a4d8248b9227ca9772e2effad1566518be))


### Bug Fixes

* **api:** remove invalid response model ([159772d](https://github.com/wi2trier/cbrkit/commit/159772d1d045561b53cb9a58d7996e60bc8bf89e))
* **deps:** pin torch to v2.2 to prevent issues on macos ([855ef87](https://github.com/wi2trier/cbrkit/commit/855ef87fd57e68ed442a395e1594090fcf14dcda))
* **nix:** install all extras ([ea54034](https://github.com/wi2trier/cbrkit/commit/ea5403431f70a9b1d52f0e548652fc192fc24b03))

## [0.11.1](https://github.com/wi2trier/cbrkit/compare/v0.11.0...v0.11.1) (2024-06-26)


### Bug Fixes

* **deps:** bump pdoc due to security issue ([219e7dd](https://github.com/wi2trier/cbrkit/commit/219e7dd4a162b1006ba389195431bf549b6dfa49))

## [0.11.0](https://github.com/wi2trier/cbrkit/compare/v0.10.1...v0.11.0) (2024-06-17)


### Features

* add two sequence similarity functions ([#127](https://github.com/wi2trier/cbrkit/issues/127)) ([9860408](https://github.com/wi2trier/cbrkit/commit/986040850ae882c2058865a6c7592b99ee6fced0))

## [0.10.1](https://github.com/wi2trier/cbrkit/compare/v0.10.0...v0.10.1) (2024-05-20)


### Bug Fixes

* add missing rich dependency ([e376fc8](https://github.com/wi2trier/cbrkit/commit/e376fc8ac83fb7d37c03371871ed574ac2f0243e))

## [0.10.0](https://github.com/wi2trier/cbrkit/compare/v0.9.1...v0.10.0) (2024-05-20)


### Features

* **sim:** add mapping-based collection metrics ([#118](https://github.com/wi2trier/cbrkit/issues/118)) ([0857b84](https://github.com/wi2trier/cbrkit/commit/0857b84b08a6b0fcb0292fb4f717061fadfff4dd))

## [0.9.1](https://github.com/wi2trier/cbrkit/compare/v0.9.0...v0.9.1) (2024-04-03)


### Bug Fixes

* **docker:** remove aarch64 images ([fccc4f7](https://github.com/wi2trier/cbrkit/commit/fccc4f7427a54daabedfcad905c73f252706a688))

## [0.9.0](https://github.com/wi2trier/cbrkit/compare/v0.8.0...v0.9.0) (2024-04-02)


### Features

* add pydantic-based validation function ([#87](https://github.com/wi2trier/cbrkit/issues/87)) ([b90f006](https://github.com/wi2trier/cbrkit/commit/b90f006c50f8928054e651acd30a37f2b892d094))

## [0.8.0](https://github.com/wi2trier/cbrkit/compare/v0.7.0...v0.8.0) (2024-04-01)


### Features

* add similarity measures for time-series data ([#80](https://github.com/wi2trier/cbrkit/issues/80)) ([53bb481](https://github.com/wi2trier/cbrkit/commit/53bb48186970c8f9e6e10df0ab3e62ee2216d7a1))

## [0.7.0](https://github.com/wi2trier/cbrkit/compare/v0.6.2...v0.7.0) (2024-03-26)


### Features

* add additional string similarity measures ([#72](https://github.com/wi2trier/cbrkit/issues/72)) ([b687bc4](https://github.com/wi2trier/cbrkit/commit/b687bc40e7b1e43d2478a801776ba4ec5e8b260c))

## [0.6.2](https://github.com/wi2trier/cbrkit/compare/v0.6.1...v0.6.2) (2024-03-26)


### Bug Fixes

* update similarity types ([fa4f201](https://github.com/wi2trier/cbrkit/commit/fa4f20128ee26585ad24de50204fda8665b0ce52))

## [0.6.1](https://github.com/wi2trier/cbrkit/compare/v0.6.0...v0.6.1) (2024-03-11)


### Bug Fixes

* update usage instructions ([ca0a2d5](https://github.com/wi2trier/cbrkit/commit/ca0a2d5504ce709e6b783fdb4b92a23d500cda0a))

## [0.6.0](https://github.com/wi2trier/cbrkit/compare/v0.5.2...v0.6.0) (2024-03-08)


### Features

* add min/max similarity to retrieval ([4a0c31f](https://github.com/wi2trier/cbrkit/commit/4a0c31f9c0424aec1440c1235fcc04af4c2cb086))

## [0.5.2](https://github.com/wi2trier/cbrkit/compare/v0.5.1...v0.5.2) (2024-03-07)


### Bug Fixes

* make taxonomy module publicly visible ([ca5a56e](https://github.com/wi2trier/cbrkit/commit/ca5a56e9a442817eaa6181965e95ddcd89d297c0))
* **sim:** add case_sensitive param to levenshtein ([3e45b5a](https://github.com/wi2trier/cbrkit/commit/3e45b5a925bdbf1bcc80444581a802398f154c67))

## [0.5.1](https://github.com/wi2trier/cbrkit/compare/v0.5.0...v0.5.1) (2024-03-06)


### Bug Fixes

* only compare attributes from the query ([6196c98](https://github.com/wi2trier/cbrkit/commit/6196c98916609d0f87dd38f348c0af921c4ade1c))

## [0.5.0](https://github.com/wi2trier/cbrkit/compare/v0.4.0...v0.5.0) (2024-03-05)


### Features

* move `global_sim` functions to `sim` ([bce823e](https://github.com/wi2trier/cbrkit/commit/bce823ef9fdf03a5b52fb075b801bfb209af7480))


### Bug Fixes

* **api:** update retriever configuration ([24c5726](https://github.com/wi2trier/cbrkit/commit/24c5726c66822a40a15823406d360afe3734f846))

## [0.4.0](https://github.com/wi2trier/cbrkit/compare/v0.3.2...v0.4.0) (2024-03-04)


### Features

* add additional taxonomy measures ([26aee48](https://github.com/wi2trier/cbrkit/commit/26aee48bc5d8051055d43b82d4040ebd5ccd2e5c))

## [0.3.2](https://github.com/wi2trier/cbrkit/compare/v0.3.1...v0.3.2) (2024-03-03)


### Bug Fixes

* add singleton helper function ([4625b6f](https://github.com/wi2trier/cbrkit/commit/4625b6f0104099b6c9045e812de727066fbe112f))

## [0.3.1](https://github.com/wi2trier/cbrkit/compare/v0.3.0...v0.3.1) (2024-03-01)


### Bug Fixes

* add missing sim exports ([4f55c97](https://github.com/wi2trier/cbrkit/commit/4f55c97bbf573af4cec0f888fbe646e97e625d4a))

## [0.3.0](https://github.com/wi2trier/cbrkit/compare/v0.2.1...v0.3.0) (2024-01-12)


### Features

* allow loading json/yaml lists ([daf01da](https://github.com/wi2trier/cbrkit/commit/daf01da070907e4bb7a736b55801264add02ea1f))


### Bug Fixes

* add slots to all classes ([6aae1d7](https://github.com/wi2trier/cbrkit/commit/6aae1d7b8a938b6395596249c655d324368e72a2))

## [0.2.1](https://github.com/wi2trier/cbrkit/compare/v0.2.0...v0.2.1) (2024-01-11)


### Bug Fixes

* correctly use final ([a51c260](https://github.com/wi2trier/cbrkit/commit/a51c260725ccc443eacd66b9a65598dfeae55d99))

## [0.2.0](https://github.com/wi2trier/cbrkit/compare/v0.1.1...v0.2.0) (2024-01-10)


### Features

* add preliminary support for graph structures ([e5e11a2](https://github.com/wi2trier/cbrkit/commit/e5e11a21169313be2d672ace42634b8daafc731a))
* allow accessing similarity by attribute ([45faf6b](https://github.com/wi2trier/cbrkit/commit/45faf6b7c93840cfd2c256e2072fe6ea0834bb45))
* allow annotated float as sim value ([6367e6b](https://github.com/wi2trier/cbrkit/commit/6367e6baedea62ccf55c536a87627717c59b98f6))


### Bug Fixes

* make SimType more generic ([c21b532](https://github.com/wi2trier/cbrkit/commit/c21b5325e408623be402df1d57cac40f0928cf45))
* **sim:** improve numeric.linear logic ([f6b1ddb](https://github.com/wi2trier/cbrkit/commit/f6b1ddb13c67e8c3c0b603ac7488f045e7efebcb))
* **sim:** update module exports ([f1616d0](https://github.com/wi2trier/cbrkit/commit/f1616d0e6702c0feb04da89e44097ff4ea5aa614))
* update retrieval.apply type signature ([82fb4a9](https://github.com/wi2trier/cbrkit/commit/82fb4a9bf38d01d77b2ac8d648f137aea44a5e1a))

## [0.1.1](https://github.com/wi2trier/cbrkit/compare/v0.1.0...v0.1.1) (2023-12-20)


### Bug Fixes

* **sim:** improve typing ([893be6b](https://github.com/wi2trier/cbrkit/commit/893be6b3b2f36711ffaa5a0aeef947dfbb39d407))

## [0.1.0](https://github.com/wi2trier/cbrkit/compare/v0.0.1...v0.1.0) (2023-12-14)


### Features

* initial release ([dcaaedb](https://github.com/wi2trier/cbrkit/commit/dcaaedb0da50e479e75f2818792b4e2a0772c7ca))


### Bug Fixes

* **cli:** handly single command properly ([a0260bb](https://github.com/wi2trier/cbrkit/commit/a0260bb77930ac9554986c3059f701c153440f97))
