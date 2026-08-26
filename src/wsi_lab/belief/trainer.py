"""Resumable ABMIL training, deterministic validation and calibration."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from wsi_lab.belief.calibration import (
    CalibrationResult,
    fit_temperature,
    save_calibration,
)
from wsi_lab.belief.dataset import (
    BeliefBatch,
    PartialBagDataset,
    ValidationBagDataset,
    collate_belief_bags,
    validate_slide_partition,
)
from wsi_lab.belief.evaluation import (
    BeliefEvaluation,
    collect_belief_predictions,
    evaluate_belief,
    evaluate_predictions,
)
from wsi_lab.belief.model import GatedAttentionMIL
from wsi_lab.belief.sampler import SlideEmbeddingRecord
from wsi_lab.errors import ArtifactCompatibilityError, TrainingError
from wsi_lab.utils.atomic import atomic_json_dump, atomic_torch_save
from wsi_lab.utils.checkpoints import require_dependencies, save_training_checkpoint
from wsi_lab.utils.fingerprints import fingerprint
from wsi_lab.utils.seeds import restore_rng_state, seed_everything

MetricCallback = Callable[[str, Mapping[str, Any]], None]


@dataclass(frozen=True, slots=True)
class BeliefDataLoaders:
    train: DataLoader[BeliefBatch]
    validation: DataLoader[BeliefBatch]


@dataclass(frozen=True, slots=True)
class BeliefTrainingResult:
    artifact_dir: Path
    checkpoint_path: Path
    calibrated_checkpoint_path: Path
    temperature: float
    best_epoch: int
    epochs_completed: int
    global_step: int
    stopped_early: bool
    validation: BeliefEvaluation
    calibration: CalibrationResult


def _config_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _config_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "model_dump"):
        return dict(config.model_dump(mode="json"))
    if hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    fields = (
        "seed",
        "max_bag_size",
        "hidden_dim",
        "attention_dim",
        "dropout",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "epochs",
        "bags_per_slide_per_epoch",
        "validation_bags_per_slide",
        "validation_bag_sizes",
        "early_stopping_patience",
        "device",
        "num_workers",
        "ece_bins",
        "resume",
        "max_grad_norm",
    )
    return {name: getattr(config, name) for name in fields if hasattr(config, name)}


def build_belief_loaders(
    train_slides: Sequence[SlideEmbeddingRecord],
    validation_slides: Sequence[SlideEmbeddingRecord],
    config: Any,
) -> BeliefDataLoaders:
    """Construct worker-safe dynamic training and fixed validation loaders."""

    validate_slide_partition(train_slides, expected="train")
    validate_slide_partition(validation_slides, expected="validation")
    training_dataset = PartialBagDataset.from_config(train_slides, config)
    validation_dataset = ValidationBagDataset.from_config(validation_slides, config)
    if training_dataset.embedding_dim != validation_dataset.slides[0].embedding_dim:
        raise ValueError("Training and validation embedding dimensions differ")
    batch_size = int(_config_value(config, "batch_size", 8))
    workers = int(_config_value(config, "num_workers", 0))
    device = str(_config_value(config, "device", "cpu"))
    common: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": workers,
        "collate_fn": collate_belief_bags,
        "pin_memory": device.startswith("cuda"),
        # Worker copies must be recreated each epoch so set_epoch is visible.
        "persistent_workers": False,
    }
    return BeliefDataLoaders(
        train=cast(
            DataLoader[BeliefBatch],
            DataLoader(training_dataset, shuffle=False, **common),
        ),
        validation=cast(
            DataLoader[BeliefBatch],
            DataLoader(validation_dataset, shuffle=False, **common),
        ),
    )


class BeliefTrainer:
    """Own one compatible, atomic, resumable belief-model artifact directory."""

    def __init__(
        self,
        model: GatedAttentionMIL,
        config: Any,
        artifact_dir: str | Path,
        *,
        dependencies: Mapping[str, str],
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        metric_callback: MetricCallback | None = None,
        manifest_name: str = "manifest.json",
    ) -> None:
        self.model = model
        self.config = config
        self.config_snapshot = _config_dict(config)
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        if not dependencies:
            raise ValueError(
                "BeliefTrainer requires dependency fingerprints for artifact compatibility"
            )
        self.dependencies = {str(key): str(value) for key, value in dependencies.items()}
        self.dependencies["belief_architecture"] = fingerprint(model.architecture_config())
        self.dependencies["belief_training_config"] = fingerprint(self.config_snapshot)
        self.metric_callback = metric_callback
        if Path(manifest_name).name != manifest_name or not manifest_name.endswith(".json"):
            raise ValueError("manifest_name must be a JSON filename without directories")
        self.manifest_name = manifest_name
        self.device = self._resolve_device(str(_config_value(config, "device", "cpu")))
        # Configure cuBLAS determinism before moving parameters to CUDA or
        # constructing an optimizer that may initialize device state.
        seed_everything(int(_config_value(config, "seed", 1)), deterministic=True)
        self.model.to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=float(_config_value(config, "learning_rate", 1e-4)),
            weight_decay=float(_config_value(config, "weight_decay", 1e-4)),
        )
        self.scheduler = scheduler
        self.global_step = 0
        self.start_epoch = 0
        self.best_epoch = -1
        self.best_nll = float("inf")
        self.stale_epochs = 0
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise TrainingError(
                f"Belief training requested {requested}, but CUDA is not available; "
                "set belief.device to 'cpu' for a CPU run"
            )
        return device

    @property
    def checkpoint_path(self) -> Path:
        return self.artifact_dir / "checkpoint.pt"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.artifact_dir / "best_checkpoint.pt"

    @property
    def calibrated_checkpoint_path(self) -> Path:
        return self.artifact_dir / "calibrated_checkpoint.pt"

    @property
    def manifest_path(self) -> Path:
        return self.artifact_dir / self.manifest_name

    def _emit(self, event: str, payload: Mapping[str, Any]) -> None:
        if self.metric_callback is not None:
            self.metric_callback(event, payload)

    def _manifest_status(self) -> str | None:
        if not self.manifest_path.exists():
            return None
        try:
            return str(json.loads(self.manifest_path.read_text(encoding="utf-8")).get("status"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_manifest(self, status: str, **metadata: Any) -> None:
        now = datetime.now(UTC).isoformat()
        existing: dict[str, Any] = {}
        if self.manifest_path.exists():
            try:
                existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
        manifest = {
            "schema_version": 1,
            "artifact_type": "belief_model",
            "fingerprint": fingerprint(
                {
                    "dependencies": self.dependencies,
                    "architecture": self.model.architecture_config(),
                }
            ),
            "status": status,
            "created_at": existing.get("created_at", now),
            "completed_at": now if status == "complete" else None,
            "dependencies": self.dependencies,
            "metadata": {
                "architecture": self.model.architecture_config(),
                "embedding_dimension": self.model.embedding_dim,
                "config": self.config_snapshot,
                **metadata,
            },
        }
        atomic_json_dump(self.manifest_path, manifest)

    @staticmethod
    def _torch_load(path: Path, *, map_location: torch.device) -> dict[str, Any]:
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:  # PyTorch before weights_only was introduced.
            checkpoint = torch.load(path, map_location=map_location)
        if not isinstance(checkpoint, dict):
            raise ArtifactCompatibilityError(f"Checkpoint {path} is not a mapping")
        return checkpoint

    def _save_checkpoint(self, path: Path, *, epoch: int, complete: bool) -> None:
        save_training_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            epoch=epoch,
            config=self.config_snapshot,
            dependencies=self.dependencies,
            extra={
                "complete": complete,
                "architecture": self.model.architecture_config(),
                "best_epoch": self.best_epoch,
                "best_nll": self.best_nll,
                "stale_epochs": self.stale_epochs,
                "history": self.history,
            },
        )

    def _load_checkpoint(self, path: Path, *, restore_rng: bool) -> dict[str, Any]:
        checkpoint = self._torch_load(path, map_location=self.device)
        require_dependencies(checkpoint, self.dependencies)
        architecture = checkpoint.get("extra", {}).get("architecture")
        if architecture != self.model.architecture_config():
            raise ArtifactCompatibilityError(
                f"Belief architecture mismatch in {path}: {architecture} != "
                f"{self.model.architecture_config()}"
            )
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.global_step = int(checkpoint.get("global_step", 0))
        extra = checkpoint.get("extra", {})
        self.best_epoch = int(extra.get("best_epoch", -1))
        self.best_nll = float(extra.get("best_nll", float("inf")))
        self.stale_epochs = int(extra.get("stale_epochs", 0))
        self.history = list(extra.get("history", []))
        if restore_rng and "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])
        return checkpoint

    def resume_latest_incomplete(self) -> bool:
        """Restore ``checkpoint.pt`` only when the stage is not marked complete."""

        if self._manifest_status() == "complete" or not self.checkpoint_path.exists():
            return False
        checkpoint = self._load_checkpoint(self.checkpoint_path, restore_rng=True)
        if bool(checkpoint.get("extra", {}).get("complete", False)):
            return False
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        return True

    def _train_epoch(self, loader: Iterable[BeliefBatch | Mapping[str, Any]]) -> float:
        self.model.train()
        losses: list[float] = []
        max_grad_norm = float(_config_value(self.config, "max_grad_norm", 5.0))
        for raw_batch in loader:
            if not isinstance(raw_batch, BeliefBatch):
                from wsi_lab.belief.evaluation import _mapping_to_batch

                batch = _mapping_to_batch(raw_batch)
            else:
                batch = raw_batch
            batch = batch.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(batch.features, batch.magnifications, mask=batch.mask)
            if not isinstance(logits, Tensor):
                logits = logits.logits
            loss = F.binary_cross_entropy_with_logits(logits, batch.labels)
            if not bool(torch.isfinite(loss)):
                raise TrainingError(
                    f"Belief loss became non-finite at global step {self.global_step}"
                )
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            self.global_step += 1
            losses.append(float(loss.detach().cpu().item()))
        if not losses:
            raise TrainingError("Belief training loader produced no batches")
        return float(np.mean(losses))

    def _restore_best(self) -> None:
        if not self.best_checkpoint_path.exists():
            raise TrainingError("Belief training completed without a best checkpoint")
        checkpoint = self._torch_load(self.best_checkpoint_path, map_location=self.device)
        require_dependencies(checkpoint, self.dependencies)
        architecture = checkpoint.get("extra", {}).get("architecture")
        if architecture != self.model.architecture_config():
            raise ArtifactCompatibilityError(
                f"Belief architecture mismatch in {self.best_checkpoint_path}"
            )
        # Model selection must not roll back the latest optimizer/global-step/
        # history bookkeeping.  Only selected weights are restored.
        self.model.load_state_dict(checkpoint["model"])

    def fit(
        self,
        train_loader: DataLoader[BeliefBatch],
        validation_loader: DataLoader[BeliefBatch],
        *,
        resume: bool | None = None,
    ) -> BeliefTrainingResult:
        """Train/select on validation, restore best weights, calibrate and finalize."""

        train_slides = getattr(getattr(train_loader, "dataset", None), "slides", None)
        validation_slides = getattr(getattr(validation_loader, "dataset", None), "slides", None)
        if train_slides is not None:
            validate_slide_partition(train_slides, expected="train")
        if validation_slides is not None:
            validate_slide_partition(validation_slides, expected="validation")
        if self._manifest_status() == "complete":
            raise TrainingError(
                f"Belief artifact {self.artifact_dir} is already complete; load its "
                "calibrated checkpoint instead of retraining"
            )
        should_resume = (
            bool(_config_value(self.config, "resume", True)) if resume is None else resume
        )
        seed_everything(int(_config_value(self.config, "seed", 1)), deterministic=True)
        if should_resume:
            resumed = self.resume_latest_incomplete()
            if resumed:
                self._emit(
                    "belief/resumed",
                    {"epoch": self.start_epoch, "step": self.global_step},
                )
        self._write_manifest("incomplete", global_step=self.global_step)

        maximum_epochs = int(_config_value(self.config, "epochs", 100))
        patience = int(_config_value(self.config, "early_stopping_patience", 10))
        ece_bins = int(_config_value(self.config, "ece_bins", 15))
        stopped_early = False
        last_epoch = self.start_epoch - 1
        try:
            for epoch in range(self.start_epoch, maximum_epochs):
                dataset = getattr(train_loader, "dataset", None)
                set_epoch = getattr(dataset, "set_epoch", None)
                if callable(set_epoch):
                    set_epoch(epoch)
                training_loss = self._train_epoch(train_loader)
                validation = evaluate_belief(
                    self.model,
                    validation_loader,
                    temperature=1.0,
                    device=self.device,
                    ece_bins=ece_bins,
                )
                validation_nll = validation.overall.nll
                improved = validation_nll < self.best_nll
                if improved:
                    self.best_nll = validation_nll
                    self.best_epoch = epoch
                    self.stale_epochs = 0
                else:
                    self.stale_epochs += 1
                record = {
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "train_loss": training_loss,
                    "validation": validation.to_dict(),
                    "improved": improved,
                }
                self.history.append(record)
                if self.scheduler is not None:
                    self.scheduler.step()
                self._save_checkpoint(self.checkpoint_path, epoch=epoch, complete=False)
                if improved:
                    self._save_checkpoint(self.best_checkpoint_path, epoch=epoch, complete=False)
                atomic_json_dump(self.artifact_dir / "history.json", self.history)
                self._emit("belief/epoch", record)
                last_epoch = epoch
                if self.stale_epochs >= patience:
                    stopped_early = True
                    break

            self._restore_best()
            raw_predictions = collect_belief_predictions(
                self.model, validation_loader, temperature=1.0, device=self.device
            )
            calibration = fit_temperature(
                torch.from_numpy(raw_predictions.logits),
                torch.from_numpy(raw_predictions.labels),
            )
            scaled_logits = raw_predictions.logits / calibration.temperature
            calibrated_probabilities = np.exp(-np.logaddexp(0.0, -scaled_logits))
            calibrated_predictions = type(raw_predictions)(
                logits=scaled_logits,
                probabilities=calibrated_probabilities,
                labels=raw_predictions.labels,
                requested_sizes=raw_predictions.requested_sizes,
                actual_sizes=raw_predictions.actual_sizes,
                slide_ids=raw_predictions.slide_ids,
            )
            final_validation = evaluate_predictions(calibrated_predictions, ece_bins=ece_bins)
            save_calibration(self.artifact_dir / "temperature.json", calibration)
            atomic_json_dump(self.artifact_dir / "metrics.json", final_validation.to_dict())
            atomic_torch_save(
                self.calibrated_checkpoint_path,
                {
                    "schema_version": 1,
                    "model": self.model.state_dict(),
                    "architecture": self.model.architecture_config(),
                    "temperature": calibration.temperature,
                    "dependencies": self.dependencies,
                    "validation": final_validation.to_dict(),
                },
            )
            # checkpoint.pt becomes the selected, complete uncalibrated model;
            # calibrated_checkpoint.pt adds only validation temperature/metrics.
            self._save_checkpoint(
                self.checkpoint_path,
                epoch=self.best_epoch,
                complete=True,
            )
            self._write_manifest(
                "complete",
                global_step=self.global_step,
                best_epoch=self.best_epoch,
                epochs_completed=last_epoch + 1,
                stopped_early=stopped_early,
                temperature=calibration.temperature,
                validation=final_validation.to_dict(),
            )
            self._emit(
                "belief/complete",
                {
                    "best_epoch": self.best_epoch,
                    "temperature": calibration.temperature,
                    "validation": final_validation.to_dict(),
                },
            )
            return BeliefTrainingResult(
                artifact_dir=self.artifact_dir,
                checkpoint_path=self.checkpoint_path,
                calibrated_checkpoint_path=self.calibrated_checkpoint_path,
                temperature=calibration.temperature,
                best_epoch=self.best_epoch,
                epochs_completed=last_epoch + 1,
                global_step=self.global_step,
                stopped_early=stopped_early,
                validation=final_validation,
                calibration=calibration,
            )
        except BaseException as exc:
            self._write_manifest(
                "failed",
                global_step=self.global_step,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise


def load_calibrated_belief(
    path: str | Path,
    *,
    expected_dependencies: Mapping[str, str] | None = None,
    device: str | torch.device = "cpu",
) -> tuple[GatedAttentionMIL, float, dict[str, Any]]:
    """Load a frozen calibrated model after strict dependency verification."""

    source = Path(path)
    try:
        checkpoint = torch.load(source, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(source, map_location=device)
    if expected_dependencies is not None:
        require_dependencies(checkpoint, {str(k): str(v) for k, v in expected_dependencies.items()})
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ArtifactCompatibilityError(f"Calibrated checkpoint {source} lacks architecture")
    model = GatedAttentionMIL(
        int(architecture["embedding_dim"]),
        hidden_dim=int(architecture["hidden_dim"]),
        attention_dim=int(architecture["attention_dim"]),
        dropout=float(architecture["dropout"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    temperature = float(checkpoint.get("temperature", 1.0))
    if temperature <= 0:
        raise ArtifactCompatibilityError(f"Invalid temperature {temperature} in {source}")
    return model, temperature, checkpoint


__all__ = [
    "BeliefDataLoaders",
    "BeliefTrainer",
    "BeliefTrainingResult",
    "build_belief_loaders",
    "load_calibrated_belief",
]
