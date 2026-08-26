# Design

## System boundary

The selected native or manifest dataset adapter owns discovery, labels, patient groups, and source-cohort metadata. The HDF5 provider owns lazy sparse feature access. Preprocessing owns dense WSI geometry, tissue projection, and source-row mappings. The Gym environment consumes processed geometry plus RGB/embedding providers. ABMIL consumes only labels, the persisted split, and embeddings. PPO consumes the processed environment and a frozen calibrated ABMIL accumulator.

This separation is intentional: `train-belief` cannot trigger tissue detection, geometric preprocessing, environment construction, or PPO.

Geometric preprocessing is development-only in the normal pipeline. Dataset-
global cell IDs use a fixed, non-overlapping int64 block per deterministic
adapter slide index. The persisted offset table may therefore be sparse, and
preprocessing an arbitrary bounded development subset never opens any other
HDF5 file. Only an explicitly authorized `eval-rl --split test --allow-test`
operation may add test entries; it receives a separate manifest and cannot
become a belief/PPO training dependency.

## Geometry before evidence

The six levels are fixed at 0.625×, 1.25×, 2.5×, 5×, 10×, and 20×. Their nominal level-0 cell footprints are 16,384, 8,192, 4,096, 2,048, 1,024, and 512 pixels. Cells start at every row-major origin inside the WSI. Nominal bounds may overhang a right or bottom edge; clipped bounds are used for maps and the RGB reader pads the rest white.

Topology is computed before sparse HDF5 rows are attached. Parent/child relations are exact quadtree geometry. Same-scale pan neighbors span the entire level. A missing `embedding_index` never removes a cell or relationship.

## Belief

Gated ABMIL receives `[UNI embedding; four-way magnification one-hot]`. Partial bags are sampled dynamically after the slide split. Magnification composition is chosen before unique rows, so dense scales do not dominate by row count. The trained model is selected on deterministic validation NLL, then scalar-temperature calibrated on validation only.

The frozen accumulator caches each hidden projection and attention logit. Its normalized pooled representation is mathematically identical to a full forward pass over the same evidence set.

## Environment

The actor sees only a five-channel aligned overview, a local RGB viewport, and normalized magnification/budget/belief scalars. The historical boolean action mask stays in `info` and rollout storage. Invalid external actions are no-ops with normal step cost and no budget, visit, or evidence change.

The final available action is reserved for an explicit diagnosis. Gym termination therefore represents diagnosis; the normal episode path does not time out.

## Recurrent PPO

Separate residual CNNs encode overview and viewport. A scalar MLP joins them before a one-layer GRU and actor/critic heads. Rollouts retain exact masks and recurrent boundary state. PPO minibatches shuffle temporal sequences, never individual timesteps, and reset GRU state at episode starts.
