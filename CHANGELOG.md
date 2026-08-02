# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-02

Initial release.

### Added

- RV32IM functional model (`isa.py`): full RV32I base set except FENCE/EBREAK/CSR,
  plus the M extension. ECALL halts.
- Two-pass RV32IM assembler (`asm.py`) with labels and pseudo-instructions,
  removing any dependency on a cross-toolchain.
- Microarchitectural model (`uarch.py`): set-associative LRU cache, gshare and
  bimodal branch predictors, load-use interlock, configurable multiply/divide
  latency with an optional early-terminating divider, and a first-order
  Hamming-weight/Hamming-distance activity model with additive Gaussian noise.
- Per-cycle instruction attribution: every activity sample carries the PC that
  produced it.
- Analysis layer (`analysis.py`): TVLA fixed-vs-random Welch t-test, remote
  timing test, correlation power analysis with signed ranking, HW and HD
  hypothesis generators, PC-based attribution, and per-trace point-of-interest
  extraction.
- Six victim programs as three leaky/hardened pairs: table lookup, modular
  exponentiation, and tag comparison.
- Campaign drivers (`campaign.py`) including a fixed-vs-fixed false-positive
  control and full 16-byte key recovery.
- ASCII and matplotlib reporting, and a CLI with `list`, `disasm`, `tvla`,
  `null`, `cpa`, `fullkey`, and `sweep` subcommands.
- 80-test suite, `scripts/reproduce.py` for regenerating all documented results,
  Dockerfile, and GitHub Actions CI across Python 3.10-3.12.

### Fixed during initial development

These were found by running the tool and are retained here because each is a
silent failure mode worth knowing about. See the README for detail.

- Shared noise seed across traces drove within-group variance to zero and the
  t statistic to ~1e15, making every victim appear to leak overwhelmingly.
- Perfect separation between two deterministic groups was reported as `t = 0`
  ("no leak") rather than as the strongest possible leak.
- Majority-vote point-of-interest windows collapsed from 21 samples to 1 as
  traces diverged in length, reducing full-key recovery to 3/16.
- The Hamming-weight hypothesis was mismatched against a Hamming-distance bus
  model, returning candidates one or two Hamming steps from the true key.
