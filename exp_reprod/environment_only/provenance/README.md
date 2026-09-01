# Provenance boundary

The harness records the source and protocol/input identities needed to
reproduce the experiment. Author-machine paths, command histories, measured
outputs, and machine/storage inventories are intentionally excluded.

The copied WSI-Lab runner resolves the library through `WSI_LAB_SOURCE_ROOT`,
which setup points at the extracted submitted source tree. The shared
`../../provenance/wsi_lab_source.sha256` manifest identifies its executable Python
snapshot without Git; the historical measured revision remains recorded as
provenance. The manifest generator requires the reproducer to supply a dataset
root; no local default is embedded. `checksums.sha256` covers every
redistributed experiment file except the checksum list itself.
