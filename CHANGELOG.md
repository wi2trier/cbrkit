# Changelog

## [0.18.3](https://github.com/wi2trier/cbrkit/compare/v0.18.2...v0.18.3) (2024-11-07)

### Bug Fixes

* **sim/graph:** improve networkx integration ([ead9c71](https://github.com/wi2trier/cbrkit/commit/ead9c715f76429807de85820d9b628fa75a9cf21))

## [0.18.2](https://github.com/wi2trier/cbrkit/compare/v0.18.1...v0.18.2) (2024-11-07)

### Bug Fixes

* **sim/graph:** add support for networkx ([#196](https://github.com/wi2trier/cbrkit/issues/196)) ([e682dcd](https://github.com/wi2trier/cbrkit/commit/e682dcd1501902ef7b99af13ceb1be90c53cfbe3))

## [0.18.1](https://github.com/wi2trier/cbrkit/compare/v0.18.0...v0.18.1) (2024-10-31)


### Bug Fixes

* **loaders:** allow getting pandas/polars rows by string ([5d21c3c](https://github.com/wi2trier/cbrkit/commit/5d21c3c8f14cac85482a613c34384936548ce073))
* **loaders:** replace `loaders.dataframe` with `loaders.pandas` ([14cec3a](https://github.com/wi2trier/cbrkit/commit/14cec3aa9c139b8daebf94d1299a29889f814de0))

## [0.18.0](https://github.com/wi2trier/cbrkit/compare/v0.17.0...v0.18.0) (2024-10-23)


### Features

* **retrieval:** add base retrieval function ([e5cddde](https://github.com/wi2trier/cbrkit/commit/e5cddde780c539942c7bc71d84341c2bb5e67bf2))
* **retrieval:** add cohere retrieval function ([839383f](https://github.com/wi2trier/cbrkit/commit/839383fa20ec2384d6be35ba623996c05a78aa48))
* **sim/generic:** add transpose function ([54d34ad](https://github.com/wi2trier/cbrkit/commit/54d34ad909c3764bc81281f2b02e912e695f117a))
* **sim/strings:** add cohere embeddings function ([bb67318](https://github.com/wi2trier/cbrkit/commit/bb673186a7747350f9042b91884566fae6b9cfdb))
* **sim/strings:** add ollama embeddings function ([9da4ef0](https://github.com/wi2trier/cbrkit/commit/9da4ef065ef4978534cd22b395cf28ef6bf773f0))


### Bug Fixes

* **retrieval:** properly call super() for dataclasses ([778d9e5](https://github.com/wi2trier/cbrkit/commit/778d9e5bcdb561390cd507890b2eab94bd248c62))
* **sim/openai:** update model calling ([4caca8b](https://github.com/wi2trier/cbrkit/commit/4caca8bf3064176e3f06abdc6a256b959ed9f208))

## [0.17.0](https://github.com/wi2trier/cbrkit/compare/v0.16.0...v0.17.0) (2024-10-17)


### Features

* add initial version of eval module ([cfb1b2d](https://github.com/wi2trier/cbrkit/commit/cfb1b2dc8d6192e5650279a6a4059fd8543925a9))


### Bug Fixes

* **eval:** import ranx in function instead of globally ([ff17643](https://github.com/wi2trier/cbrkit/commit/ff17643d8db946c92f2f824eedf726f7cd223d3b))
* **eval:** use functional approach instead of classes ([4a8d97e](https://github.com/wi2trier/cbrkit/commit/4a8d97ef15b473bb2cbdc6b604bf0f5eb075ffa0))
* **retrieval:** add query key type for mapply ([3e8ac6b](https://github.com/wi2trier/cbrkit/commit/3e8ac6bf4eda2e13eee38e2de2e115fbce160b23))
* **sim:** export graph sim ([edb3ef0](https://github.com/wi2trier/cbrkit/commit/edb3ef0b4b18f099df45b35590b905d209d5bcab))

## [0.16.0](https://github.com/wi2trier/cbrkit/compare/v0.15.0...v0.16.0) (2024-10-15)


### Features

* **sim/graph:** add support for data-based similarity ([803dc1c](https://github.com/wi2trier/cbrkit/commit/803dc1c95de701f321faa271172f3e6e8d495308))
* **sim/table:** allow function as values and arbitrary key transformations ([a918931](https://github.com/wi2trier/cbrkit/commit/a91893112fe496368e61dc247510d3b5914af4a4))
* **sim/table:** convert to SimSeqFunc for faster computation ([14ab90f](https://github.com/wi2trier/cbrkit/commit/14ab90f2980251741cf466380bd8a16072896999))
* **sim/table:** separate static and dynamic table func ([7cdf24f](https://github.com/wi2trier/cbrkit/commit/7cdf24fd5b168fb9464dca94f11c716b3aaf9c58))
* **sim:** add static function ([effcb87](https://github.com/wi2trier/cbrkit/commit/effcb87a5241037d4a087049187385ce876ff512))


### Bug Fixes

* **sim:** improve metadata for sim wrappers ([fa30d8b](https://github.com/wi2trier/cbrkit/commit/fa30d8b138e6e8d9822e73936021a41ce4e47cd0))
* **sim:** update pandas series type ([01096fe](https://github.com/wi2trier/cbrkit/commit/01096febe09c2ef3baca30aa13a053913460019b))
* update AttributeValueData type ([f30cc4d](https://github.com/wi2trier/cbrkit/commit/f30cc4d2519ac012b67556f33603013c96035f40))

## [0.15.0](https://github.com/wi2trier/cbrkit/compare/v0.14.2...v0.15.0) (2024-10-15)


### Features

* **helpers:** add sim2pair function ([76f8595](https://github.com/wi2trier/cbrkit/commit/76f8595b461543cb69ab4f54219441d6998cc4a7))
* **sim/graph:** add support for rustworkx ([c30bc01](https://github.com/wi2trier/cbrkit/commit/c30bc0170b832fe33dd95100cbf94fe7a427ff70))
* **sim/graph:** rewrite interface completely ([7d6b965](https://github.com/wi2trier/cbrkit/commit/7d6b965bd8140db3213fd45b257b44e941240a97))
* **sim/strings:** add support for custom model instances ([c185504](https://github.com/wi2trier/cbrkit/commit/c18550439356626c8d18a138c7c4e932361e13ea))
* **sim:** convert SimMapFuncs to SimSeqFuncs for better compat ([17cd6b3](https://github.com/wi2trier/cbrkit/commit/17cd6b3ceeefd3120a915b84caea542615f8cfaf))
* switch to new 3.12 generics and add support for retrieval metadata ([30912a8](https://github.com/wi2trier/cbrkit/commit/30912a8f093773c8c29bf6d55b61ded6dd2d91db))


### Bug Fixes

* **retrieval:** freeze result dataclass ([00cc0fd](https://github.com/wi2trier/cbrkit/commit/00cc0fd59b09bb5a6d2f9f5e48515c6a19c4d033))
* **sim/graph:** improve algorithms ([3072670](https://github.com/wi2trier/cbrkit/commit/3072670748e868febd73ebff3afd0900a17e4878))
* **sim/graph:** improve model and trim down exports ([ce61b9d](https://github.com/wi2trier/cbrkit/commit/ce61b9daeadf852a679db515ed1c92e2513b9b0b))
* **sim/strings:** add missing exported functions ([7a409aa](https://github.com/wi2trier/cbrkit/commit/7a409aaad19d6176e1033fa54519fe2d97ae2781))

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
