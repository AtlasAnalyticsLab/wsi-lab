# Provenance boundary

The workflow records protocol/input identities, software versions, and the
pinned HistoGym revision needed to reproduce the experiment. Author-machine
paths, command histories, measured outputs, and machine/storage inventories
are excluded from the committed package.

Set `WSI_LAB_SOURCE_ROOT` to the WSI-Lab repository's absolute path before
running setup. The harness uses that local source tree. Configuration and
validation check the dataset cohort, workload, and protocol inputs. The
manifest generator requires an explicit dataset root; no machine-specific
default is embedded.
