# Required frozen belief checkpoint

The included `calibrated_checkpoint.pt` is the calibrated frozen ABMIL belief
checkpoint used by the measured experiment. It is not trained or updated by
this benchmark and is distributed under the repository's Apache-2.0 license.

Measured checkpoint metadata:

- filename: `calibrated_checkpoint.pt`
- SHA-256: `71fb6677a5307a53424e84a111e5b88c426e845065b85f8bf09c03f7d0d239fc`
- size: 4,215,860 bytes
- embedding dimension: 1,536
- hidden dimension: 512
- attention dimension: 256

`scripts/configure.py` uses this file by default. An explicit replacement can
be supplied with `--belief-checkpoint`; configuration refuses a file with a
different hash or size.
