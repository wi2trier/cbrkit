# Changelog

## [0.23.2](https://github.com/wi2trier/cbrkit/compare/v0.23.1...v0.23.2) (2025-03-31)

### Bug Fixes

* **retrieval/build:** if no chunksize given, do not compute pairwise batches in advance ([0e21714](https://github.com/wi2trier/cbrkit/commit/0e2171465c87c23d073149c9b832e408a6ea952f))

## [0.23.1](https://github.com/wi2trier/cbrkit/compare/v0.23.0...v0.23.1) (2025-03-27)

### Bug Fixes

* **deps:** add graphviz library ([273a807](https://github.com/wi2trier/cbrkit/commit/273a807a0e0f4eb0e05250d87323e42bd6f50509))
* **model:** add graphviz export ([2e003fd](https://github.com/wi2trier/cbrkit/commit/2e003fde941e9ac3b24ba8702276def8e4738c25))
* only deconstruct casebases once for multiprocessing if they are identical ([0b11f0a](https://github.com/wi2trier/cbrkit/commit/0b11f0a076b2a5fc88c9f425927f8276ddab9a31))

## [0.23.0](https://github.com/wi2trier/cbrkit/compare/v0.22.0...v0.23.0) (2025-03-22)

### Features

* **synthesis:** allow responses to be structured to pass down metadata like usage ([a9dc237](https://github.com/wi2trier/cbrkit/commit/a9dc23709a32dc1881933fc49092eb39cfdfafb3))

### Bug Fixes

* **eval:** add missing auto scale parameter ([3e2f55a](https://github.com/wi2trier/cbrkit/commit/3e2f55a39b747c8ea282efe755fbfcc396a7db93))
* **eval:** handle single-valued metric return values from ranx ([29b5344](https://github.com/wi2trier/cbrkit/commit/29b53447e0902cebd7d94ab934356d5cb306a524))
* **sim/collections:** update exported members ([0b1bfe7](https://github.com/wi2trier/cbrkit/commit/0b1bfe78d40874785abe157036d49ddb9adbe498))
* **sim/embed:** correctly check if cache file already exists ([aace36d](https://github.com/wi2trier/cbrkit/commit/aace36d19d3396aaf8782ccd557a605611169ae0))
* **synthesis/openai:** handle finish reasons ([531f903](https://github.com/wi2trier/cbrkit/commit/531f9034219adc24de25b0037960c7482e6c6b49))
* **synthesis/openai:** properly handle response type ([762ce0f](https://github.com/wi2trier/cbrkit/commit/762ce0f3f8e3061b838132e7ba9e67182343ab5d))
* **synthesis:** make chat message type generic ([5f1a1d4](https://github.com/wi2trier/cbrkit/commit/5f1a1d458ecf76ac58c71f831c0bbd7e7290bc9b))

## [0.22.0](https://github.com/wi2trier/cbrkit/compare/v0.21.3...v0.22.0) (2025-03-07)

### Features

* **model:** add duration to result of retrieval/reuse/synthesis/cycle ([b1c5071](https://github.com/wi2trier/cbrkit/commit/b1c5071cc367103e3630af6490f1e1f804117c08))
* **synthesis/providers:** add instructor provider ([b831426](https://github.com/wi2trier/cbrkit/commit/b831426593ff846efe871f0d59beac256bcaab3e))
* **synthesis/providers:** add parameters retried and default_response ([c019ebe](https://github.com/wi2trier/cbrkit/commit/c019ebec18e359abd6a3c03b2a6ac501320d025a))
* **synthesis/providers:** print response in case of an exception ([209a358](https://github.com/wi2trier/cbrkit/commit/209a358bd53c349be7e6b902f96f6fb2ff001c23))

### Bug Fixes

* **synthesis/openai:** add missing parameters ([4c4b8d4](https://github.com/wi2trier/cbrkit/commit/4c4b8d45bc046111edff9e4310dfb40d7567a64d))
* **synthesis/openai:** improve error handling ([c77ef56](https://github.com/wi2trier/cbrkit/commit/c77ef56ade5666188fd963e2bad56717d331ae14))

## [0.21.3](https://github.com/wi2trier/cbrkit/compare/v0.21.2...v0.21.3) (2025-02-21)

### Bug Fixes

* **helpers:** add chain_map_chunks function ([a567e30](https://github.com/wi2trier/cbrkit/commit/a567e304547a297aad4d62a54933a42e3a73b7e4))
* **helpers:** allow omitting default values when converting functions to pydantic ([047f3a7](https://github.com/wi2trier/cbrkit/commit/047f3a7b3a0945daa7892ec0469953fc87eead6a))
* **synthesis/openai:** improve tool usage logic ([549511c](https://github.com/wi2trier/cbrkit/commit/549511c6b0719d448cb17b04f2939f80cb9a1491))
* **synthesis/prompts:** add concat wrapper function ([55ac06d](https://github.com/wi2trier/cbrkit/commit/55ac06de02f116941b080e8a728d4e7e014f02cf))
* **synthesis:** allow arbitrary chunk conversion functions ([49b07bc](https://github.com/wi2trier/cbrkit/commit/49b07bc8db808e28693654f311fa056ab636921a))
* **synthesis:** allow non-batched pooling functions ([4a95350](https://github.com/wi2trier/cbrkit/commit/4a95350808c9b505447234531d3a17df02a6c1e6))
* **typing:** generalize positional/named function types ([b68e822](https://github.com/wi2trier/cbrkit/commit/b68e822888ccc3f49b0a5f58d4cac6672be62725))

## [0.21.2](https://github.com/wi2trier/cbrkit/compare/v0.21.1...v0.21.2) (2025-02-13)

### Bug Fixes

* **synthesis:** improve encoding of data for prompts ([d37c70f](https://github.com/wi2trier/cbrkit/commit/d37c70f5c9663317a3651a9c9ca9e638159a18f2))

## [0.21.1](https://github.com/wi2trier/cbrkit/compare/v0.21.0...v0.21.1) (2025-02-06)

### Bug Fixes

* **sim/embed:** add `load_spacy` to download models on demand ([babc594](https://github.com/wi2trier/cbrkit/commit/babc5947619381a8c9e20bf00b2e0df09ba3f911))

## [0.21.0](https://github.com/wi2trier/cbrkit/compare/v0.20.4...v0.21.0) (2025-02-05)

### ⚠ BREAKING CHANGES

* The entire library has largely been rewritten, so there will be additional breaking changes. Please refer to the Readme and the tests for more information.
* The function `cbrkit.reuse.build` now expects a retriever function instead of a similarity function so that more logic can be shared between the phases.
* To better support the new retrieval functions, the arguments `limit`, `min_similarity`, and `max_similarity` of the function `cbrkit.retrieval.build` have been removed. Instead, wrap your call of `cbrkit.retrieval.build` with the new function `cbrkit.retrieval.dropout` that now exposes these arguments.
* The functions `apply` and `mapply` have been removed to better support processing multiple queries at once. They have been replaced by the functions `apply_query` and `apply_queries`. Both return the same result object, so the return value of `apply_queries` is not identical to the one of the previous `mapply` function. The functions `apply` and `apply_query` however share the same return type.
* The number of processes to use for retrieval is no longer passed to the `apply` functions, but instead given to the `build` function.
* To better support the new retrieval functions, the arguments `limit`, `min_similarity`, and `max_similarity` of the function `cbrkit.retrieval.build` have been removed. Instead, wrap your call of `cbrkit.retrieval.build` with the new function `cbrkit.retrieval.dropout` that now exposes these arguments.
* CBRkit now provides additional modules for `adapt`, `reuse`, `cycle`, and `eval`.
* We added support for logging via the standard library.
* There is a new `synthesis` module that provides tight integration with various LLM providers. This can for instance be used to develop RAG applications using CBR.
* Loading and dumping cases has been reworked, we now provide generators to construct serialization and deserialization functions.
* Caching of similarity values has been added, simply wrap your existing similarity function with the new `cbrkit.sim.cache` wrapper.
* A new embedding module `cbrkit.sim.embed` has been added that provides a better interface to compose string-based similarity functions that rely on vectors. It also includes a cache that can be stored on disk.
* Similarity functions for graphs have been overhauled and now provide a more consistent interface.

### Features

* **adapt:** add openai function to adapt cases ([38cbb26](https://github.com/wi2trier/cbrkit/commit/38cbb26cf85785c4bb12df28ebbd07007fc26925))
* **adapt:** add similarity delta to pipe function ([9d58252](https://github.com/wi2trier/cbrkit/commit/9d58252c25405b3d47b24d9569ff0e9bbeba5d03))
* add docstrings to export ([3991a51](https://github.com/wi2trier/cbrkit/commit/3991a518bd5bbd14ece2e839834f09492da05be6))
* add dumpers module for serializing casebases ([475f532](https://github.com/wi2trier/cbrkit/commit/475f5322fdd47a9ed5dc3d8a822adc41c5a966ff))
* add dumpers, anthropic provider, update docs ([#215](https://github.com/wi2trier/cbrkit/issues/215)) ([0f440c5](https://github.com/wi2trier/cbrkit/commit/0f440c55dad870977e25bc09b479875b53c43f45))
* add generation submodule to handle provider-specific code ([068b6ff](https://github.com/wi2trier/cbrkit/commit/068b6ff62d8443ba92060826dbc76cd1994af89c))
* add global handling of asyncio event loop ([0ed704a](https://github.com/wi2trier/cbrkit/commit/0ed704ad06a1bb8e161af2fb73d48fd7acdcb446))
* add initial version of rag module ([f803b3b](https://github.com/wi2trier/cbrkit/commit/f803b3b62b0aef9b9eefd68c3344b905c0f1f336))
* add integration with voyageai ([6d7b4eb](https://github.com/wi2trier/cbrkit/commit/6d7b4eb04181bcb42834a82203f4c6400b6e79c6))
* add logging ([32fde3d](https://github.com/wi2trier/cbrkit/commit/32fde3d2a6ef6a956d2c204432bd8490f7b38de7))
* add methods to perform entire r4 cycles more easily ([eb08557](https://github.com/wi2trier/cbrkit/commit/eb08557c2db7db970ff5917f142900e380ff1fa8))
* add openapi schema generator ([611370d](https://github.com/wi2trier/cbrkit/commit/611370d0175468c25a7c33f13422201b9fa0eb15))
* add rag support to api and cli ([57a0334](https://github.com/wi2trier/cbrkit/commit/57a0334f2738b8dd47581737b329fb70efbdfb5a))
* add support for factories ([529aa1a](https://github.com/wi2trier/cbrkit/commit/529aa1a1ecde297b24c2c37b1ad70b9ed79fe4cd))
* add support for factories to more functions ([2794ffc](https://github.com/wi2trier/cbrkit/commit/2794ffc2f06bec58c8363307d423a725ec6dac70))
* add transpose_value wrappers ([ebbabc9](https://github.com/wi2trier/cbrkit/commit/ebbabc989304bcaf74ae896c44626c31b7c98214))
* **api:** allow passing paths for casebase/query ([8cdcbb4](https://github.com/wi2trier/cbrkit/commit/8cdcbb4e0d71d69bc9332af99d7527e604c9dd1d))
* **api:** support passing files ([2993fac](https://github.com/wi2trier/cbrkit/commit/2993fac6ce72948bac4000ec3d9a36836bb71e15))
* **api:** switch query parameters to request body ([9604428](https://github.com/wi2trier/cbrkit/commit/960442817323776982a2876eda16a4331c68529f))
* convert results to pydantic models ([3b1c5e0](https://github.com/wi2trier/cbrkit/commit/3b1c5e0b631266728c2b08d679d27ccf25e07b27))
* **dumpers:** make markdown function generic ([33d627c](https://github.com/wi2trier/cbrkit/commit/33d627c91ca65c12eade0bed6546585efbc771fd))
* **embed/openai:** add lazy loading ([7e5c783](https://github.com/wi2trier/cbrkit/commit/7e5c78334c84bfaab804199cfd6b658a01b8f826))
* **embed:** add lazy loading for cache ([98790cc](https://github.com/wi2trier/cbrkit/commit/98790ccd9229ad69fc999e1677017948b7cf4b05))
* **eval:** add helper for arbitrary scores ([79ce192](https://github.com/wi2trier/cbrkit/commit/79ce192c99d2ed7f2925ad1a81d694150a34bbc5))
* **eval:** add proper support for relevance levels ([c414112](https://github.com/wi2trier/cbrkit/commit/c414112918daf51e7c60524655547e9ea355cf26))
* **eval:** allow conversion of retrieval result to qrels ([c8a7cb5](https://github.com/wi2trier/cbrkit/commit/c8a7cb5fda495061f2bf710b5cee915302af9bbd))
* **eval:** allow custom metric functions ([fde869a](https://github.com/wi2trier/cbrkit/commit/fde869a51e624bf4445e764ea7c856a99d01290f))
* **generate:** add memory to openai ([e645fde](https://github.com/wi2trier/cbrkit/commit/e645fde52fe784bf37b19971b81b9abdb241abe4))
* **helpers:** add getitem_or_getattr ([0b94774](https://github.com/wi2trier/cbrkit/commit/0b947747ed8d5455032c8b14b793e51d5050edfd))
* **helpers:** allow conversion of functions to base models ([b206c1f](https://github.com/wi2trier/cbrkit/commit/b206c1f7234c7f111978d20a28de3fa55a3b2b7d))
* improve genai providers ([8da6269](https://github.com/wi2trier/cbrkit/commit/8da6269e077cf60b33ae8da4f58ea0e7dd8fb84f))
* improve handling of multiprocessing ([81ac32c](https://github.com/wi2trier/cbrkit/commit/81ac32c24142ccd9885ec998eb76f079eca399d5))
* improve logging and multiprocessing ([3b78deb](https://github.com/wi2trier/cbrkit/commit/3b78deb3521bb16580479bfa2b32d40080cf25de))
* integrate processing of query collections into the core of cbrkit ([b8df8ee](https://github.com/wi2trier/cbrkit/commit/b8df8ee79d43827f79e5eceec0ccc16f104e22e1))
* make cbrkit project layout more consistent ([b738d6f](https://github.com/wi2trier/cbrkit/commit/b738d6f1ea69e6e316e4cb29e2f523ad3eab165c))
* **multiprocessing:** allow boolean values ([e9e3827](https://github.com/wi2trier/cbrkit/commit/e9e382755b288e76a9f2a00ae00c04ef82fe6efb))
* **openai:** add support for tool calling via unions ([0b3e29e](https://github.com/wi2trier/cbrkit/commit/0b3e29e9abaabd65d15ecfe390fcb87c13b762bd))
* optimize multiprocessing ([87ee55f](https://github.com/wi2trier/cbrkit/commit/87ee55f9196139818e83f48b6a9c9b35aef8e81c))
* **rag:** add model similar to retrieval/reuse ([ca76c73](https://github.com/wi2trier/cbrkit/commit/ca76c732d48cce2014882755cc9103b33b806a36))
* **retrieval:** add dropout function ([3f50dbf](https://github.com/wi2trier/cbrkit/commit/3f50dbff9fdf829f26d224a58bca99af23e450ba))
* **retrieval:** add openai function for estimating the similarity ([803ff75](https://github.com/wi2trier/cbrkit/commit/803ff755e998df657d9c3b95a33e0172bf67e61f))
* **retrieval:** add sentence transformers reranker ([bd05b2e](https://github.com/wi2trier/cbrkit/commit/bd05b2eac87e3f8591e3bef352dc2c03f4f27085))
* **retrieval:** add transpose helper to simplify conversion of cases ([216eca3](https://github.com/wi2trier/cbrkit/commit/216eca3abee7f04eb22bbd070ceaa4e0bb5cf231))
* **retrieval:** use async clients for cohere and voyage ai ([6b37814](https://github.com/wi2trier/cbrkit/commit/6b37814a412771bba165c7fe0211c8c381bb3f47))
* **reuse:** allow passing multiple adaptation functions to builder ([6493923](https://github.com/wi2trier/cbrkit/commit/64939233498eb4ceda8d29cf63d02f3e0080da57))
* **reuse:** allow passing similarities from earlier steps ([6732b38](https://github.com/wi2trier/cbrkit/commit/6732b380efbc37b9bc373267bbc8727cc3ff1ff5))
* **reuse:** introduce dropout function similar to retrieval ([c3254c6](https://github.com/wi2trier/cbrkit/commit/c3254c6d738557641c52e6ed76af3a12abb535c4))
* rework reuse phase and update apply helpers ([f4a11e8](https://github.com/wi2trier/cbrkit/commit/f4a11e850c79db0f0ef1f1a4e3d4e7f3e3952c9c))
* rework type structure and improve genai/rag modules ([b94e898](https://github.com/wi2trier/cbrkit/commit/b94e898a0338c5926292229aa4730f4575772e63))
* **sim/embed:** add chunking/truncation for openai ([406c7bc](https://github.com/wi2trier/cbrkit/commit/406c7bc318d19bc7e6cee1f248c74ff723198bff))
* **sim/graphs:** add dtw alignment ([#214](https://github.com/wi2trier/cbrkit/issues/214)) ([bc44cb7](https://github.com/wi2trier/cbrkit/commit/bc44cb75d2332e7fdd54504d1da6b3ef8ebdbe31))
* **sim/graphs:** add dtw and smith-waterman functions ([#201](https://github.com/wi2trier/cbrkit/issues/201)) ([040b702](https://github.com/wi2trier/cbrkit/commit/040b70217d9bb7ca76004689351260aceeb4db55))
* **sim/graphs:** add initial version of exhaustive mapping ([818a356](https://github.com/wi2trier/cbrkit/commit/818a356b69d8956129728243823aacd5177db5ab))
* **sim/graphs:** add local sims, update astar heuristics ([fe51441](https://github.com/wi2trier/cbrkit/commit/fe51441d2fc0f5302670f0870d78673c3d1ffc4b))
* **sim/graphs:** add precompute function ([5ba52d7](https://github.com/wi2trier/cbrkit/commit/5ba52d718e1a24a18cb23aed192035f8ca939b14))
* **sim/graphs:** add smith ([#218](https://github.com/wi2trier/cbrkit/issues/218)) ([771fe3b](https://github.com/wi2trier/cbrkit/commit/771fe3b0ec93400c3bf9c5530bd6d00d15e24a9d))
* **sim/graphs:** make it easier to define node similarities ([0c95d3a](https://github.com/wi2trier/cbrkit/commit/0c95d3a8641961205a4ad2bc747c01481f43a9ea))
* **sim/graphs:** rewrite astar algorithm ([9c788f6](https://github.com/wi2trier/cbrkit/commit/9c788f67946a8dab1cb0573d9e70086003a44f48))
* **sim/strings:** add vector database ([07727f0](https://github.com/wi2trier/cbrkit/commit/07727f0a5dd4310408983469274f136873c819d7))
* **sim:** add cache method ([8c9992f](https://github.com/wi2trier/cbrkit/commit/8c9992f00fcfcec0c13bfb5096d81ea819192019))
* **sim:** add default sim for attribute value ([686f270](https://github.com/wi2trier/cbrkit/commit/686f270ce38e3ade2de28f7d3f878ff3153c8211))
* **sim:** update interface for embed and taxonomy functions ([6af97f9](https://github.com/wi2trier/cbrkit/commit/6af97f945ac1e626befdf2e3adc576132e7b2955))
* **sim:** update table helpers and move to wrappers ([868bd8d](https://github.com/wi2trier/cbrkit/commit/868bd8dcb1fdd7cda6b3605b58c9d9ec08828ba7))
* **synthesis/providers:** add delay parameter ([0fad3c5](https://github.com/wi2trier/cbrkit/commit/0fad3c5d36defd9483d9150ef38fe4cf1d625d4b))
* **synthesis:** add google provider ([1724ad0](https://github.com/wi2trier/cbrkit/commit/1724ad0f230dc72d1b47a5a53908096f7cbdbbd3))
* **synthesis:** allow chunking with overlap ([ee5bebd](https://github.com/wi2trier/cbrkit/commit/ee5bebd6415740b2a2045ce535cc635788e90f7d))
* **synthesis:** run chunk helper in parallel ([b91c9c0](https://github.com/wi2trier/cbrkit/commit/b91c9c0529bdeba7961720da86236475929d761e))
* various improvements ([da308f2](https://github.com/wi2trier/cbrkit/commit/da308f225de21d8851dc4ebed54c91f0d513447f))

### Bug Fixes

* **adapt/generic:** add strategy to pipe function ([d943f59](https://github.com/wi2trier/cbrkit/commit/d943f59b9fe76026c67b6786d0d264eafe29b1ee))
* **api:** response types failed to validate ([a5fe024](https://github.com/wi2trier/cbrkit/commit/a5fe0247628c9972bb9cc3719f290de265360822))
* **api:** simplify definition of retrievers/reusers ([371cd78](https://github.com/wi2trier/cbrkit/commit/371cd78553efdb44b0cfb2be32f49f1675bae271))
* **astar:** convert to batch sim func ([cfc0d42](https://github.com/wi2trier/cbrkit/commit/cfc0d424957a692948f077e9768fa6224238014e))
* **astar:** improve logic ([1643549](https://github.com/wi2trier/cbrkit/commit/164354955e5d191d0b7ddf2dc7bd5c8ea25350e3))
* **astar:** make naming and exports more consistent ([007dfb0](https://github.com/wi2trier/cbrkit/commit/007dfb04f074a81bbcc74f21550af5f580f65618))
* **astar:** restructure legal mapping funcs, add sim precomputer ([8d54903](https://github.com/wi2trier/cbrkit/commit/8d549036fadd8a89810247f9f8a98e791137fca6))
* **chunkify:** check arguments ([90b8f38](https://github.com/wi2trier/cbrkit/commit/90b8f38509d640fa36b4a2c1f45a629214e1c142))
* **cli:** disable pretty exceptions ([215070b](https://github.com/wi2trier/cbrkit/commit/215070b73e11e4ec43cf26b3b46e64954c8f532d))
* **cli:** use dumpers for exporting ([2293948](https://github.com/wi2trier/cbrkit/commit/2293948815e14b1725289b875c8bb216c8a32b16))
* convert some lambdas to real functions ([10994c3](https://github.com/wi2trier/cbrkit/commit/10994c3bc65d1fdfdc110895d212affe5a9c08a4))
* correctly construct pydantic models ([fc49b1a](https://github.com/wi2trier/cbrkit/commit/fc49b1a473177971034907f1098b13e4ebc89f59))
* correctly set sentence_transformers metadata ([d3511cc](https://github.com/wi2trier/cbrkit/commit/d3511ccfefc967bd8a2a8f567cf6a1315cee82d2))
* default to structured outputs for openai ([247bb05](https://github.com/wi2trier/cbrkit/commit/247bb057e47956ab41ee871d42baf5e0170425db))
* **dumpers:** properly get name for markdown code block ([40ba442](https://github.com/wi2trier/cbrkit/commit/40ba442385b0c22f5e55ee5ba726fcf5baef8d43))
* **embed:** add autodump to cache ([f7dc949](https://github.com/wi2trier/cbrkit/commit/f7dc949547938a6e447c8e11c901ffa7dd15c329))
* **embed:** add lazy loading to sentence transformers ([0cf74ec](https://github.com/wi2trier/cbrkit/commit/0cf74ecfce6180f893381242598c2524a9934db3))
* **embed:** add logging ([9156846](https://github.com/wi2trier/cbrkit/commit/9156846227dfac07624f87ae65dbf8e741e93857))
* **embed:** autodump only if new texts are found ([bd9d05b](https://github.com/wi2trier/cbrkit/commit/bd9d05b13e266eb60113564b265257afa48c6eef))
* **embed:** check hash before dumping cache ([118611b](https://github.com/wi2trier/cbrkit/commit/118611b56f40cf393f99d8552eecf22cc723a639))
* **embed:** remove unneeded lazy loading ([45738c0](https://github.com/wi2trier/cbrkit/commit/45738c0c9752578eaa4524159099319d38641d39))
* **embed:** use modified time instead of hash to detect changes ([4ee3963](https://github.com/wi2trier/cbrkit/commit/4ee39634f87452781946e59f93449dbade9bc819))
* **eval:** add kendall tau ([95eda58](https://github.com/wi2trier/cbrkit/commit/95eda584ebbb64d85fca4cb110b8bb6869ce8c6c))
* **eval:** add mean_score function ([9ac7c30](https://github.com/wi2trier/cbrkit/commit/9ac7c30a5945fe20e18252ebeba1e1f6224f87d0))
* **eval:** improve conversion of scores to qrels ([579de7f](https://github.com/wi2trier/cbrkit/commit/579de7faa26df9a74dad249dda338236cddf93a7))
* **eval:** improve metric generation ([c14f48e](https://github.com/wi2trier/cbrkit/commit/c14f48ed6c10e738ebd720f406f54dd96b730401))
* export default aggregator ([f4d4473](https://github.com/wi2trier/cbrkit/commit/f4d44736a8bbcb733694153e827f4b73346f38f4))
* extend support for lazy loading ([645cad4](https://github.com/wi2trier/cbrkit/commit/645cad44f4de738874ce63753cee578d4e69d030))
* formatting and typing improvements ([d49f3e5](https://github.com/wi2trier/cbrkit/commit/d49f3e5e36b2aece52516886dc52fe5d3223fc41))
* **genai/prompts:** add transpose function ([dc02582](https://github.com/wi2trier/cbrkit/commit/dc02582af351316953cd021ff74fa9fc78a1e527))
* **graph:** enhance serialization ([d6111a4](https://github.com/wi2trier/cbrkit/commit/d6111a44acf2daf14c1956a93df7b819ef53e542))
* **graphs:** add converter callbacks to dump/load ([d96e9d8](https://github.com/wi2trier/cbrkit/commit/d96e9d8d1bab7a23f1496f2f97766980fc7f9910))
* **graphs:** add load/dump ([e61bd92](https://github.com/wi2trier/cbrkit/commit/e61bd92591d676be3bcb20ccd63e550407ef670c))
* **graphs:** drop SerializedNode ([b4e0939](https://github.com/wi2trier/cbrkit/commit/b4e0939b4f9a7889973e36b81a2daf5e531f46ad))
* **helpers:** add log_batch ([cf17591](https://github.com/wi2trier/cbrkit/commit/cf17591b3c429acad9b24b40b7c39ca9232d1937))
* **helpers:** correctly handle bool values for multiprocessing ([b02e749](https://github.com/wi2trier/cbrkit/commit/b02e749176c255cf5190cd6508c18ba2e9270726))
* **helpers:** optimize loading of callable maps ([22e9443](https://github.com/wi2trier/cbrkit/commit/22e9443e3eca0864d43033a20e45ec3a10254e39))
* improve dumpers, especially for graphs ([cb9f6df](https://github.com/wi2trier/cbrkit/commit/cb9f6df31fdd88c8be3fe6d52d993a1e04dfc5f3))
* improve eval module ([a85f817](https://github.com/wi2trier/cbrkit/commit/a85f817e8d3aa14d6d8b189c66f8450329cc0aad))
* improve handling of defaults for tables ([f082511](https://github.com/wi2trier/cbrkit/commit/f0825117f3d01250b2b86b747d91334dd5719a1d))
* improve logging during multiprocessing ([fb038ba](https://github.com/wi2trier/cbrkit/commit/fb038ba3a863e84f09ea44863c031fc027c21801))
* improve synthesis ([00e95c2](https://github.com/wi2trier/cbrkit/commit/00e95c23ddc6df9cb5a68f9595d56ce6df1ba22a))
* improve typing of generic tables ([7ca8740](https://github.com/wi2trier/cbrkit/commit/7ca87406a97859d2b1a467979ffad8cbc88e6f75))
* improve vector db ([31afcc1](https://github.com/wi2trier/cbrkit/commit/31afcc1d77ffb044b2d7daeff2cdd6b585f10a09))
* keep casebase/query in result object when dumping ([3cfd623](https://github.com/wi2trier/cbrkit/commit/3cfd623e032d37165e3330d6498950633ff63aff))
* **loaders:** correctly handle files in directories ([afa26f2](https://github.com/wi2trier/cbrkit/commit/afa26f26c99fd6fbba69133488878119e33f04d1))
* **loaders:** properly handle io ([088f3c9](https://github.com/wi2trier/cbrkit/commit/088f3c9f9d385db5da8fbe55239e466a052192fb))
* **loaders:** properly load binary data ([33c3424](https://github.com/wi2trier/cbrkit/commit/33c34248f1af01649cbb23e3b11406363ead3ada))
* log only if more than one batches are processed ([10d8371](https://github.com/wi2trier/cbrkit/commit/10d8371533e1f10dd3ea5efd347ba29aa7bd51f0))
* make dumper argument ordering more consistent ([5e90214](https://github.com/wi2trier/cbrkit/commit/5e90214721bdd3ffcf89e43c2e343b66a0f44039))
* make reuse/retrieval functions more robust ([e7437b9](https://github.com/wi2trier/cbrkit/commit/e7437b989b68aedf151e307d05c10cde5d0ed770))
* minor improvements ([f1c25b0](https://github.com/wi2trier/cbrkit/commit/f1c25b0011f37e5a18bba749ec2f84bf601b5f43))
* **model:** add default_query to top result class ([9d33c17](https://github.com/wi2trier/cbrkit/commit/9d33c170ced32e28f359d118d26a0f0c88feb1f7))
* **model:** store unfiltered casebase as well ([73ccb69](https://github.com/wi2trier/cbrkit/commit/73ccb69ef6533c40185972bff0871b04e5efe2a0))
* move from TypedDict to BaseModel/dataclass ([2afce5c](https://github.com/wi2trier/cbrkit/commit/2afce5c01d1d962ec41325a4d230a5394f04c89c))
* **openai:** use not_given where necessary ([e39f297](https://github.com/wi2trier/cbrkit/commit/e39f29789b876898dcc11574efa3c40dd89b64d9))
* **prompts:** allow giving functions as instructions ([1646548](https://github.com/wi2trier/cbrkit/commit/16465480239082a7dd37c63e315d406037ac0a03))
* **prompts:** remove dedent ([8abce29](https://github.com/wi2trier/cbrkit/commit/8abce2937e4ab3ae6e3e6fb187cc64cf6b510e7b))
* re-add logging to astar ([aa99c92](https://github.com/wi2trier/cbrkit/commit/aa99c92bff18b027bc22cb047d794ceb1af59b81))
* remove factories that are no longer needed ([a420695](https://github.com/wi2trier/cbrkit/commit/a420695704c8a833734be27cba228be648b974a7))
* remove synthesis-based retriever/reuser until a better interface is defined ([7d813c5](https://github.com/wi2trier/cbrkit/commit/7d813c5d8c80e584b23d56e75fa76b0168c8e587))
* restore similarity filtering behavior ([7ec627d](https://github.com/wi2trier/cbrkit/commit/7ec627d9e296ac03cacbc3f3386a8dae94decf8a))
* result export ([58ba034](https://github.com/wi2trier/cbrkit/commit/58ba034e3d1bcb46122155db571f2e700f61cce6))
* **retrieval:** improve metadata ([76d90e4](https://github.com/wi2trier/cbrkit/commit/76d90e411e561c9ad2b30fbc5e8e77d4ff579fb3))
* **retrieval:** optimize sentence transformers ([553a018](https://github.com/wi2trier/cbrkit/commit/553a018a132b68061b7a55eeb6aaea00f7848e4f))
* **sim/astar:** add default to max-calls ([e2fc133](https://github.com/wi2trier/cbrkit/commit/e2fc133fc6ad21c9fa6cf0bc3f9e66486f7caa2f))
* **sim/astar:** correctly compute sim and loop over the open set ([28cb22b](https://github.com/wi2trier/cbrkit/commit/28cb22b755b6639d0a9cffbef7e5c6f318094025))
* **sim/astar:** force to map all edges in select2 ([87ea326](https://github.com/wi2trier/cbrkit/commit/87ea3262272030f355523b9f2b2948faccf21a80))
* **sim/astar:** remove optimization for edge expansion ([f8c1409](https://github.com/wi2trier/cbrkit/commit/f8c14094df91a6bb4a0b7c827b03b3f461a9b919))
* **sim/collections:** allow dtw for arbitrary types ([#204](https://github.com/wi2trier/cbrkit/issues/204)) ([5f7585e](https://github.com/wi2trier/cbrkit/commit/5f7585e636e9ae71287e5fce59d93b5b65a26b1b))
* **sim/collections:** update types for dtw ([74737f1](https://github.com/wi2trier/cbrkit/commit/74737f1322ec2348c4d321975ae555b12e566ceb))
* **sim/embed:** correctly convert to float ([32925d5](https://github.com/wi2trier/cbrkit/commit/32925d5aa984b8944322ab69a87c1452b5677602))
* **sim/embed:** correctly load/dump cached store ([22c8ffc](https://github.com/wi2trier/cbrkit/commit/22c8ffc8ea7b7ae6b5caa431294739774a748015))
* **sim/embed:** generalize helper functions ([3bb7c10](https://github.com/wi2trier/cbrkit/commit/3bb7c10aa44e1b90ad2af4eba7509e58185ab9a8))
* **sim/graphs:** generalize graph sim ([7212929](https://github.com/wi2trier/cbrkit/commit/721292993a813a7f5ffab192c212bc7b1958d5b1))
* **sim/graphs:** improve is_sequential and conditionally import alignment metrics ([33b25e4](https://github.com/wi2trier/cbrkit/commit/33b25e42b12d403a2aa0bbb74f0b4a7be56ea7d7))
* **sim/graphs:** improve isomorphism ([a6bac6b](https://github.com/wi2trier/cbrkit/commit/a6bac6bee2846548491ce2ae65cc74c0fb8def45))
* **sim/graphs:** merge node_data_sim and node_obj_sim ([9b912ad](https://github.com/wi2trier/cbrkit/commit/9b912ad79cc89bf5359fae98a01134ff73958986))
* **sim/graphs:** swap x and y in some cases ([070e8fb](https://github.com/wi2trier/cbrkit/commit/070e8fb241e2813abaccb15c71165540a00a4d06))
* **sim/graphs:** use dicts for graph sim return value ([4fffbc0](https://github.com/wi2trier/cbrkit/commit/4fffbc072dabb630085f962fc4800c6beb7cca49))
* **sim/graphs:** use optional dependencies for alignment ([4ea0005](https://github.com/wi2trier/cbrkit/commit/4ea0005670b1bc6f79dc9d1ccaeddafcadf8de2b))
* **sim/strings:** gracefully handle empty batches ([6f7e86e](https://github.com/wi2trier/cbrkit/commit/6f7e86e640fc0828c6916b8e38c5301d517d0237))
* **sim/strings:** optimize computation of semantic similarities ([cfa6505](https://github.com/wi2trier/cbrkit/commit/cfa6505c460503b6053b5e7e3efc7211ac335fc5))
* **sim:** add type_equality function ([c49a3af](https://github.com/wi2trier/cbrkit/commit/c49a3af44999d5605df9a2e4a8a3d9f77f9d5618))
* **sim:** do not serialize cache ([9ef0b26](https://github.com/wi2trier/cbrkit/commit/9ef0b26387239d86f0f89935565c0ec410c27ca6))
* **sim:** expand functionality of dynamic table ([e79cb33](https://github.com/wi2trier/cbrkit/commit/e79cb33867c753814aa7bba243b3103fa3eeabce))
* **sim:** improve table similarities ([977d798](https://github.com/wi2trier/cbrkit/commit/977d798f1480fc93b019d20b5b5fcb7856fa85a4))
* small improvements for sentence transformers ([7877c27](https://github.com/wi2trier/cbrkit/commit/7877c2703f7a4300a2942222f5e419e758ec3730))
* **synthesis:** add logging ([c59839a](https://github.com/wi2trier/cbrkit/commit/c59839a721f19579554caf599677c62edd198425))
* **synthesis:** openai message construction ([959d33e](https://github.com/wi2trier/cbrkit/commit/959d33e0d822cfc2c07cfb66bc50f9388abc0344))
* **synthesis:** properly use init vars ([dd4ddf1](https://github.com/wi2trier/cbrkit/commit/dd4ddf1da7a64b79e7fc7d07e1ca3ac708677e90))
* **synthesis:** update openai parameters ([9eac912](https://github.com/wi2trier/cbrkit/commit/9eac912e1abbf145dd66d3a4ae3d79fad14e2def))
* **taxonomy:** allow paths ([b35e488](https://github.com/wi2trier/cbrkit/commit/b35e4884987ebef3d90d781bbd5c233c7030177e))
* **typing:** use np.float64 ([fdbd45b](https://github.com/wi2trier/cbrkit/commit/fdbd45b02676d91c4d443ebb901027e8d932c706))
* update text loaders ([c60219c](https://github.com/wi2trier/cbrkit/commit/c60219c5cd73318196a03fbf683a30aa61dada39))
* use rag functions in retrieve/adapt and add chunking ([f737a2b](https://github.com/wi2trier/cbrkit/commit/f737a2b9b2183f0ffef0b95b62e9b654b927f998))

### Miscellaneous Chores

* add notable changes ([1f1bd17](https://github.com/wi2trier/cbrkit/commit/1f1bd17d087ef99813a5d75e33f89b8a5d911ff0))

## [0.20.4](https://github.com/wi2trier/cbrkit/compare/v0.20.3...v0.20.4) (2024-11-19)

### Bug Fixes

* **docs:** generate cli documentation automatically ([3a9fae4](https://github.com/wi2trier/cbrkit/commit/3a9fae4b2ba4b7a71c2721c391650654b53a8cbd))

## [0.20.3](https://github.com/wi2trier/cbrkit/compare/v0.20.2...v0.20.3) (2024-11-18)

### Bug Fixes

* **deps:** update nix flake lock ([40e8a66](https://github.com/wi2trier/cbrkit/commit/40e8a66fa9ea6fce50d9f2cfdf89572e3ca2dab6))

## [0.20.2](https://github.com/wi2trier/cbrkit/compare/v0.20.1...v0.20.2) (2024-11-18)

### Bug Fixes

* make docker image smaller to fix CI failures ([4405fd3](https://github.com/wi2trier/cbrkit/commit/4405fd32214f17f09ecfebe147e3fd440a97d576))

## [0.20.1](https://github.com/wi2trier/cbrkit/compare/v0.20.0...v0.20.1) (2024-11-17)

### Bug Fixes

* adjust paths in docs to fix build ([59e1f87](https://github.com/wi2trier/cbrkit/commit/59e1f87ff20d400c6d3c7853e2332d5263cb9e2a))

## [0.20.0](https://github.com/wi2trier/cbrkit/compare/v0.19.2...v0.20.0) (2024-11-17)

### Features

* **adapt:** add null adaptation ([e477975](https://github.com/wi2trier/cbrkit/commit/e477975aa59a1e0d02230813824697a6a4bb262b))
* **adapt:** add number aggregator ([ef9a6be](https://github.com/wi2trier/cbrkit/commit/ef9a6bea30c4f8581a1ec92c287bd62003dfc7eb))
* add built-in adaptation functions ([8072481](https://github.com/wi2trier/cbrkit/commit/8072481e48341c78ea41f599ff14c6f137fd5bfd))
* add compositional adaptation ([e5b8634](https://github.com/wi2trier/cbrkit/commit/e5b8634b089f6e4e52c00da255b4cfea09137546))
* add reuse/adaptation interface with build/apply helpers ([fdd54c0](https://github.com/wi2trier/cbrkit/commit/fdd54c080aecbc18d6f4e2bb6627db7d8754044c))
* **api:** add reuse endpoints ([3e728f7](https://github.com/wi2trier/cbrkit/commit/3e728f7b4a71408c943dcd85bb5f17a042a2ae78))
* **cli:** add reuse command ([0e5c6bf](https://github.com/wi2trier/cbrkit/commit/0e5c6bf2589903b8b71b7cf42b7b5b25c9340bb2))
* migrate from poetry to uv ([794455c](https://github.com/wi2trier/cbrkit/commit/794455cda9c9f4bea61dc5f7fabedd8d5a7069d7))
* prefer polars over pandas ([773bed8](https://github.com/wi2trier/cbrkit/commit/773bed82346d72081ef9fc647259a91ce531cc12))
* **reuse:** make interface more generic ([119ce3b](https://github.com/wi2trier/cbrkit/commit/119ce3b61a426bd32122f41313a33e513c2af77e))

### Bug Fixes

* **adapt/strings:** improve regex ([84acce6](https://github.com/wi2trier/cbrkit/commit/84acce6a9abf953cd4358cf7b03a1e7652b3e94e))
* **adapt/strings:** swap order of query and case pattern ([84a1a7a](https://github.com/wi2trier/cbrkit/commit/84a1a7ad617f045e1fbdaa44683e85c398b272d4))
* **adapt:** rename rules to pipe ([c60b0d3](https://github.com/wi2trier/cbrkit/commit/c60b0d31fdeda78be4b6e598597412d15d468797))
* **aggregator:** export pooling functions ([49644bd](https://github.com/wi2trier/cbrkit/commit/49644bde1fc70837c9cdfc14a59a3d2875294177))
* **api:** properly extend result classes ([b622610](https://github.com/wi2trier/cbrkit/commit/b62261074825438502e66612572bd64cba4a52b3))
* **attribute-value:** remove unused type parameter ([6e6e0cf](https://github.com/wi2trier/cbrkit/commit/6e6e0cf979c54827405a7147d6c424fa1da67202))
* **cli:** add reusers to serve function ([43aeb29](https://github.com/wi2trier/cbrkit/commit/43aeb29ae205531613e716151a44dddf53e921cb))
* **helpers:** add generics to sim wrappers ([dcf4007](https://github.com/wi2trier/cbrkit/commit/dcf40075b49bec106203e36ca329b85f19403a9c))
* **loaders:** make pydantic optional ([5ea54d6](https://github.com/wi2trier/cbrkit/commit/5ea54d6d19662258bb06460c5b782901752a4ad8))
* **reuse:** update typing of apply_single ([79983bd](https://github.com/wi2trier/cbrkit/commit/79983bde7bb1e6864bf141750a654ad98c605ebe))
* **sim/graphs:** improve similarity types ([176fe2b](https://github.com/wi2trier/cbrkit/commit/176fe2be88ac78b112a879636782b508f40958d1))

## [0.19.2](https://github.com/wi2trier/cbrkit/compare/v0.19.1...v0.19.2) (2024-11-11)

### Bug Fixes

* **eval:** ensure inequality of two qrels for completeness/correctness ([d2bd202](https://github.com/wi2trier/cbrkit/commit/d2bd2026234aab12cbb3ec3e9baa4104cc0f910a))

## [0.19.1](https://github.com/wi2trier/cbrkit/compare/v0.19.0...v0.19.1) (2024-11-11)

### Bug Fixes

* **cli:** allow printing of similarities ([9fc1595](https://github.com/wi2trier/cbrkit/commit/9fc1595b8e24257638f5f34adb472c487d67cb4d))

## [0.19.0](https://github.com/wi2trier/cbrkit/compare/v0.18.3...v0.19.0) (2024-11-08)

### Features

* **sim/graph:** add isomorphism-based approach ([8de9597](https://github.com/wi2trier/cbrkit/commit/8de9597ff422534d87f22fef5ab365625c6238dc))

### Bug Fixes

* **eval:** overhaul correctness completeness ([fc530bf](https://github.com/wi2trier/cbrkit/commit/fc530bf3ea5fb22ca699895f8e1ec201b19a1781))
* **eval:** update correctness completeness ([04ffab0](https://github.com/wi2trier/cbrkit/commit/04ffab0d94ae534ba6be3b5412e25fbe220acc90))

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
