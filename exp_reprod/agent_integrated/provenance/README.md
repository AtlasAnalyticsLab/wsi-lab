# Provenance boundary

The harness records the public source revisions and protocol/input identities
needed to reproduce the experiment. Author-machine paths, command histories,
measured outputs, and machine/storage inventories are intentionally excluded.

The copied WSI-Lab runner resolves the library through
`WSI_LAB_SOURCE_ROOT`, which setup points at a detached pinned worktree. The
manifest generator requires the reproducer to supply a dataset root; no local
default is embedded. `checksums.sha256` covers every redistributed experiment
file except the checksum list itself.
