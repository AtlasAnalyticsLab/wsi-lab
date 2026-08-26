"""Strict typed configuration models."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    model_validator,
)

from wsi_lab.constants import BELIEF_MAGNIFICATIONS


class ConfigModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class DatasetConfig(ConfigModel):
    name: Literal["camelyon16", "camelyon17", "tcga_nsclc", "generic"] = "camelyon16"
    root: Path
    slides_dir: Path = Path("images")
    labels: Path = Path("evaluation/reference.csv")
    tissue_masks_dir: Path = Path("background_tissue")
    tumor_masks_dir: Path = Path("masks")
    annotations_dir: Path = Path("annotations")
    luad_dir: Path = Path("TCGA-LUAD")
    lusc_dir: Path = Path("TCGA-LUSC")

    @model_validator(mode="after")
    def native_metadata_default(self) -> DatasetConfig:
        if self.name == "camelyon17" and self.labels == Path("evaluation/reference.csv"):
            self.labels = Path("stages.csv")
        elif self.name == "generic" and self.labels == Path("evaluation/reference.csv"):
            self.labels = Path("metadata.csv")
        return self


class EmbeddingsConfig(ConfigModel):
    root: Path = Path("Embeddings")
    encoder: Literal["uni_v2"] = "uni_v2"
    magnifications: tuple[StrictFloat, ...] = BELIEF_MAGNIFICATIONS
    directory_template: str = "UNI_v2_{magnification}x_256/patches"
    feature_key: str = "features/uni_v2"
    coordinate_key: str = "coords"
    passport_key: str = "passports"
    mapping_tolerance: StrictInt = Field(default=0, ge=0)
    validation_chunk_rows: StrictInt = Field(default=8192, ge=1)

    @model_validator(mode="after")
    def fixed_protocol_scales(self) -> EmbeddingsConfig:
        if tuple(float(x) for x in self.magnifications) != BELIEF_MAGNIFICATIONS:
            raise ValueError(f"WSILab-v0 requires magnifications {BELIEF_MAGNIFICATIONS}")
        try:
            rendered = self.directory_template.format(
                magnification="2.5",
                encoder=self.encoder,
                slide_id="normal_001",
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "embedding directory_template contains an unsupported placeholder"
            ) from exc
        relative = Path(rendered)
        if relative.is_absolute():
            raise ValueError("embedding directory_template must be relative")
        if ".." in relative.parts:
            raise ValueError("embedding directory_template cannot contain parent traversal")
        excluded = {"pean", "bench_embeddings"}
        if any(part.casefold() in excluded for part in relative.parts):
            raise ValueError("PEAN and bench_embeddings are excluded embedding sources")
        return self


class WorkspaceConfig(ConfigModel):
    root: Path = Path("workspace")


class AtlasPatchConfig(ConfigModel):
    executable: str = "atlaspatch"
    device: str = "cuda:0"
    segmentation_batch_size: StrictInt = Field(default=1, ge=1)
    keep_overlay: StrictBool = False
    timeout_seconds: StrictInt = Field(default=7200, ge=1)


class TissueSource(StrEnum):
    AUTO = "auto"
    EXISTING = "existing"
    ATLASPATCH = "atlaspatch"


class TissueConfig(ConfigModel):
    source: TissueSource = TissueSource.ATLASPATCH
    threshold: StrictFloat = Field(default=0.01, ge=0.0, le=1.0)
    rebuild: StrictBool = False
    atlaspatch: AtlasPatchConfig = Field(default_factory=AtlasPatchConfig)

    @model_validator(mode="after")
    def validate_source_rebuild(self) -> TissueConfig:
        if self.source is TissueSource.EXISTING and self.rebuild:
            raise ValueError("tissue.source='existing' cannot be combined with rebuild=true")
        return self


class RewardType(StrEnum):
    DEFAULT = "default"
    BELIEF = "belief"


class EnvironmentConfig(ConfigModel):
    overview_size: StrictInt = Field(default=224, ge=16)
    viewport_size: StrictInt = Field(default=224, ge=16)
    max_episode_steps: StrictInt = Field(default=500, ge=2)
    min_diagnostic_evidence: StrictInt = Field(default=4, ge=0)
    max_visit_count: StrictInt = Field(default=10, ge=1)
    reward_type: RewardType = RewardType.DEFAULT
    step_cost: StrictFloat = Field(default=0.001, ge=0.0)
    belief_reward_beta: StrictFloat = 1.0
    slide_handle_cache_size: StrictInt = Field(default=4, ge=1)
    viewport_cache_size: StrictInt = Field(default=256, ge=0)
    persistent_rgb_cache: Path | None = None


class BeliefConfig(ConfigModel):
    seed: StrictInt = 1
    max_bag_size: StrictInt = Field(default=500, ge=1)
    hidden_dim: StrictInt = Field(default=512, ge=1)
    attention_dim: StrictInt = Field(default=256, ge=1)
    dropout: StrictFloat = Field(default=0.1, ge=0.0, lt=1.0)
    learning_rate: StrictFloat = Field(default=1e-4, gt=0.0)
    weight_decay: StrictFloat = Field(default=1e-4, ge=0.0)
    max_grad_norm: StrictFloat = Field(default=5.0, gt=0.0)
    batch_size: StrictInt = Field(default=8, ge=1)
    epochs: StrictInt = Field(default=100, ge=1)
    bags_per_slide_per_epoch: StrictInt = Field(default=8, ge=1)
    validation_bags_per_slide: StrictInt = Field(default=8, ge=1)
    validation_bag_sizes: tuple[StrictInt, ...] = (1, 4, 8, 16, 32, 64, 128, 256, 500)
    early_stopping_patience: StrictInt = Field(default=10, ge=1)
    device: str = "cuda:0"
    num_workers: StrictInt = Field(default=4, ge=0)
    ece_bins: StrictInt = Field(default=15, ge=2)
    resume: StrictBool = True

    @model_validator(mode="after")
    def bag_sizes_fit(self) -> BeliefConfig:
        if not self.validation_bag_sizes:
            raise ValueError("validation bag sizes cannot be empty")
        if any(value <= 0 for value in self.validation_bag_sizes):
            raise ValueError("validation bag sizes must be positive")
        if len(set(self.validation_bag_sizes)) != len(self.validation_bag_sizes):
            raise ValueError("validation bag sizes must be unique")
        if max(self.validation_bag_sizes) > self.max_bag_size:
            raise ValueError("validation bag sizes cannot exceed max_bag_size")
        return self


class VectorBackend(StrEnum):
    SYNC = "sync"
    ASYNC = "async"


class RLConfig(ConfigModel):
    seed: StrictInt = 1
    num_envs: StrictInt = Field(default=8, ge=1)
    rollout_steps: StrictInt = Field(default=128, ge=1)
    sequence_length: StrictInt = Field(default=32, ge=1)
    learning_rate: StrictFloat = Field(default=2.5e-4, gt=0.0)
    gamma: StrictFloat = Field(default=0.99, ge=0.0, le=1.0)
    gae_lambda: StrictFloat = Field(default=0.95, ge=0.0, le=1.0)
    clip_coef: StrictFloat = Field(default=0.2, gt=0.0)
    update_epochs: StrictInt = Field(default=4, ge=1)
    entropy_coef: StrictFloat = Field(default=0.01, ge=0.0)
    value_coef: StrictFloat = Field(default=0.5, ge=0.0)
    max_grad_norm: StrictFloat = Field(default=0.5, gt=0.0)
    num_minibatches: StrictInt = Field(default=4, ge=1)
    total_timesteps: StrictInt = Field(default=10_000_000, ge=1)
    anneal_learning_rate: StrictBool = True
    clip_value_loss: StrictBool = True
    target_kl: StrictFloat | None = Field(default=None, gt=0.0)
    checkpoint_interval: StrictInt = Field(default=100_000, ge=1)
    validation_interval: StrictInt = Field(default=100_000, ge=1)
    vector_backend: VectorBackend = VectorBackend.SYNC
    device: str = "cuda:0"
    belief_device: str = "cpu"
    evaluation_episodes: StrictInt = Field(default=54, ge=1)

    @model_validator(mode="after")
    def recurrent_batch_shape(self) -> RLConfig:
        if self.rollout_steps % self.sequence_length:
            raise ValueError("rollout_steps must be divisible by sequence_length")
        sequences = self.num_envs * (self.rollout_steps // self.sequence_length)
        if sequences % self.num_minibatches:
            raise ValueError("rollout sequence count must be divisible by num_minibatches")
        return self


class SplitConfig(ConfigModel):
    seed: StrictInt = 1
    validation_fraction: StrictFloat = Field(default=0.2, gt=0.0, lt=1.0)
    test_fraction: StrictFloat = Field(default=0.2, gt=0.0, lt=1.0)
    stratify_by_center: StrictBool = True

    @model_validator(mode="after")
    def generated_fraction_budget(self) -> SplitConfig:
        if self.validation_fraction + self.test_fraction >= 1.0:
            raise ValueError("validation_fraction + test_fraction must be less than 1")
        return self


class LoggingConfig(ConfigModel):
    tensorboard: StrictBool = True
    wandb: StrictBool = False
    wandb_project: str = "wsi-lab"


class AppConfig(ConfigModel):
    dataset: DatasetConfig
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    tissue: TissueConfig = Field(default_factory=TissueConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    belief: BeliefConfig = Field(default_factory=BeliefConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
