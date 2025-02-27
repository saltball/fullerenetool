# Changelog

## [0.2.0](https://github.com/saltball/fullerenetool/compare/v0.1.0...v0.2.0) (2025-02-27)


### Features

* **derivative_group:** add DerivativeGroupType enum and coordinate generation for substituents ([5b24bd9](https://github.com/saltball/fullerenetool/commit/5b24bd93be58fafb35332d0117ff30d18a9354a6))
* **dev_fullerene_flow:** update workflow and add new functionality ([586b7f0](https://github.com/saltball/fullerenetool/commit/586b7f0db99fa18ce89515a112a40b110e3d70cb))


### Bug Fixes

* **cage:** enhance error messages with neighbor information ([5b24bd9](https://github.com/saltball/fullerenetool/commit/5b24bd93be58fafb35332d0117ff30d18a9354a6))
* **operator:** redirect stderr to stdout in gaussian and orca calculations ([31497c8](https://github.com/saltball/fullerenetool/commit/31497c8b003e84a73d7ee4ceb8f57494d6c34667))

## 0.1.0 (2025-02-17)


### Features

* add addons outside of fullerenefamily done. ([1552bb6](https://github.com/saltball/fullerenetool/commit/1552bb656726154e022abc1463c0bec591415b94))
* **addexo:** addons_to_fullerene function and class. ([09a6310](https://github.com/saltball/fullerenetool/commit/09a6310a3a7cf577b8f6bb721bddca641edb6c88))
* **algorithm:** add verbose logging to cycle finder ([bb9d347](https://github.com/saltball/fullerenetool/commit/bb9d347a8d463e80b09773be3e943ee6b694c5e5))
* **algorithm:** implement cycle finding algorithm with TBB support ([f73f4a8](https://github.com/saltball/fullerenetool/commit/f73f4a8a8680f7e23068978d39075c2be5c61a10))
* **flow:** implement nonisomorphic addon generation and energy calculation ([4c70cf5](https://github.com/saltball/fullerenetool/commit/4c70cf5e45c5b32bf0c552a17f0b8ecdace89e09))
* **fp:** enhance ABACUS and VASP calculation features ([b51b100](https://github.com/saltball/fullerenetool/commit/b51b100b7fb27dff0af7ba1ee9d0ee60d57a14d8))
* **fullerene:** enhance addon generation, improve error handling, and add visualization support ([8b7eed4](https://github.com/saltball/fullerenetool/commit/8b7eed4df6208fe464d4b75700da499ce95e57a3))
* **fullerene:** implement addon generation with certification and sorting ([530fba2](https://github.com/saltball/fullerenetool/commit/530fba25d7ac7ac39f2379e43a9ff962ebb786d2))
* **fullerene:** improve addon generation and handling ([6c0617a](https://github.com/saltball/fullerenetool/commit/6c0617afbb4b1f584ce8d4ca80f192e28a6c99d7))
* **fullerene:** improve non-isomorphic addon generation and energy calculation ([1e606ea](https://github.com/saltball/fullerenetool/commit/1e606eabd9a85c453f05bf81b4c4911484428538))
* **operator:** add support for fragmented molecular calculations in Gaussian ([ed333b0](https://github.com/saltball/fullerenetool/commit/ed333b084ecf2149d696e2dca6d9b105d61daedd))
* **operator:** add xtb calculation ops and example ([1aa6f03](https://github.com/saltball/fullerenetool/commit/1aa6f03afa5a32d2d11dad940f35509636d56a8d))


### Bug Fixes

* addon_generator.py use fullerene_dev directly ([c6a57ad](https://github.com/saltball/fullerenetool/commit/c6a57adf73dc1c3635238c9c137dfd3db5146c38))
* **fullerene:** handle ImportError for DEFAULT_MAX_STEPS and add progress bar ([a9510c3](https://github.com/saltball/fullerenetool/commit/a9510c318d591933c95dc000bbbdf34720d7feaa))
* **operator:** verify canon_graph output via certificate comparison ([cd243e9](https://github.com/saltball/fullerenetool/commit/cd243e99607c827c8a8b7cdd27ea8fee4732adb8))
* optimize C60 derivative generation process ([bd82208](https://github.com/saltball/fullerenetool/commit/bd82208ec59b477a7d90cad88d2c50e2173a8043))
