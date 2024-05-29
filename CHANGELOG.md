# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2024-05-29

### Added
- Added canonicalization with optimization approach.
- Added evaluating transfer learning capabilities of canonicalizer.
- Added pull request template.
- Added test for discrete invert canonicalization.

### Fixed
- Fixed segmentation evaluation for non-identity canonicalizers.
- Fixed minor bugs in inverse canonicalization for discrete groups.

### Changed
- Updated `README.md` with [Improved Canonicalization for Model Agnostic Equivariance](https://arxiv.org/abs/2405.14089) ([EquiVision](https://equivision.github.io/), CVPR 2024 workshop) paper details.
- Updated `CONTRIBUTING.md` with more information on how to run the code checks.
- Changed the OS used to test Python 3.7 on GitHub actions (macos-latest -> macos-13).

## [0.1.1] - 2024-03-15

### Changed
- Operating system classifier in `setup.cfg`.
- Replaced `escnn` dependency with `e2cnn`.

## [0.1.0] - 2024-03-14

### Added
- Initial version of project on GitHub.
