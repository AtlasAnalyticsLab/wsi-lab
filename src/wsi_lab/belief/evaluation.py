"""Deterministic partial-bag prediction and calibration metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from wsi_lab.belief.accumulator import BeliefAccumulator
from wsi_lab.belief.dataset import BeliefBatch
from wsi_lab.belief.model import GatedAttentionMIL
from wsi_lab.belief.sampler import SlideEmbeddingRecord


@dataclass(frozen=True, slots=True)
class BeliefMetrics:
    auroc: float
    accuracy: float
    nll: float
    brier_score: float
    ece: float
    sensitivity: float
    specificity: float
    sample_count: int
    positive_count: int
    negative_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BeliefPredictions:
    logits: np.ndarray
    probabilities: np.ndarray
    labels: np.ndarray
    requested_sizes: np.ndarray
    actual_sizes: np.ndarray
    slide_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        count = int(self.logits.size)
        arrays = (
            self.probabilities,
            self.labels,
            self.requested_sizes,
            self.actual_sizes,
        )
        if any(int(array.size) != count for array in arrays) or len(self.slide_ids) != count:
            raise ValueError("Prediction fields must have the same number of samples")


@dataclass(frozen=True, slots=True)
class BeliefEvaluation:
    overall: BeliefMetrics
    by_requested_size: Mapping[int, BeliefMetrics]
    predictions: BeliefPredictions

    def to_dict(self, *, include_predictions: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "overall": self.overall.to_dict(),
            "by_requested_size": {
                str(size): metrics.to_dict()
                for size, metrics in sorted(self.by_requested_size.items())
            },
        }
        if include_predictions:
            result["predictions"] = {
                "logits": self.predictions.logits.tolist(),
                "probabilities": self.predictions.probabilities.tolist(),
                "labels": self.predictions.labels.tolist(),
                "requested_sizes": self.predictions.requested_sizes.tolist(),
                "actual_sizes": self.predictions.actual_sizes.tolist(),
                "slide_ids": list(self.predictions.slide_ids),
            }
        return result


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Exact rank AUROC with average ranks for ties and no sklearn dependency."""

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or labels.size == 0:
        raise ValueError("AUROC labels and scores must be non-empty and equally sized")
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Ranks are one-based; all ties receive the mean occupied rank.
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    if bins < 2:
        raise ValueError("ECE requires at least two bins")
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    # right=True puts exactly 1.0 in the final bin after clipping.
    assignments = np.clip(np.digitize(probabilities, boundaries[1:-1], right=True), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum())
        if count:
            confidence = float(probabilities[selected].mean())
            accuracy = float(labels[selected].mean())
            error += count / labels.size * abs(accuracy - confidence)
    return float(error)


def compute_belief_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    ece_bins: int = 15,
    threshold: float = 0.5,
) -> BeliefMetrics:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.shape != probabilities.shape or labels.size == 0:
        raise ValueError("Metric labels and probabilities must be non-empty and equally sized")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("Labels must be binary")
    predictions = probabilities >= threshold
    positive = labels == 1
    negative = ~positive
    true_positive = int(np.sum(predictions & positive))
    true_negative = int(np.sum(~predictions & negative))
    positives = int(positive.sum())
    negatives = int(negative.sum())
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    nll = -float(np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)))
    return BeliefMetrics(
        auroc=binary_auroc(labels, probabilities),
        accuracy=float(np.mean(predictions == positive)),
        nll=nll,
        brier_score=float(np.mean((probabilities - labels) ** 2)),
        ece=expected_calibration_error(labels, probabilities, bins=ece_bins),
        sensitivity=true_positive / positives if positives else float("nan"),
        specificity=true_negative / negatives if negatives else float("nan"),
        sample_count=int(labels.size),
        positive_count=positives,
        negative_count=negatives,
    )


def _mapping_to_batch(batch: Mapping[str, Any]) -> BeliefBatch:
    required = {"features", "magnifications", "mask", "labels"}
    missing = required - set(batch)
    if missing:
        raise ValueError(f"Belief batch mapping is missing fields: {sorted(missing)}")
    labels = torch.as_tensor(batch["labels"])
    count = int(labels.shape[0])
    mask = torch.as_tensor(batch["mask"])
    bag_sizes = torch.as_tensor(batch.get("bag_sizes", mask.sum(dim=1)), dtype=torch.long)
    requested = torch.as_tensor(batch.get("requested_sizes", bag_sizes), dtype=torch.long)
    return BeliefBatch(
        features=torch.as_tensor(batch["features"]),
        magnifications=torch.as_tensor(batch["magnifications"]),
        mask=mask,
        labels=labels,
        slide_ids=tuple(str(value) for value in batch.get("slide_ids", range(count))),
        bag_sizes=bag_sizes,
        requested_sizes=requested,
    )


def collect_belief_predictions(
    model: GatedAttentionMIL,
    loader: Iterable[BeliefBatch | Mapping[str, Any]],
    *,
    temperature: float = 1.0,
    device: str | torch.device | None = None,
) -> BeliefPredictions:
    """Run a deterministic, dropout-free validation pass."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    logits_parts: list[Tensor] = []
    labels_parts: list[Tensor] = []
    requested_parts: list[Tensor] = []
    actual_parts: list[Tensor] = []
    slide_ids: list[str] = []
    try:
        with torch.inference_mode():
            for raw_batch in loader:
                batch = (
                    raw_batch
                    if isinstance(raw_batch, BeliefBatch)
                    else _mapping_to_batch(raw_batch)
                )
                device_batch = batch.to(target_device)
                logits = model(
                    device_batch.features,
                    device_batch.magnifications,
                    mask=device_batch.mask,
                )
                if not isinstance(logits, Tensor):  # Defensive for alternate model wrappers.
                    logits = logits.logits
                logits_parts.append((logits / temperature).detach().cpu())
                labels_parts.append(batch.labels.detach().cpu())
                requested_parts.append(batch.requested_sizes.detach().cpu())
                actual_parts.append(batch.bag_sizes.detach().cpu())
                slide_ids.extend(batch.slide_ids)
    finally:
        model.train(was_training)
    if not logits_parts:
        raise ValueError("Cannot evaluate an empty belief loader")
    logits_tensor = torch.cat(logits_parts).to(dtype=torch.float64)
    probabilities = torch.sigmoid(logits_tensor)
    return BeliefPredictions(
        logits=logits_tensor.numpy(),
        probabilities=probabilities.numpy(),
        labels=torch.cat(labels_parts).to(dtype=torch.int64).numpy(),
        requested_sizes=torch.cat(requested_parts).to(dtype=torch.int64).numpy(),
        actual_sizes=torch.cat(actual_parts).to(dtype=torch.int64).numpy(),
        slide_ids=tuple(slide_ids),
    )


def evaluate_predictions(
    predictions: BeliefPredictions,
    *,
    ece_bins: int = 15,
) -> BeliefEvaluation:
    overall = compute_belief_metrics(
        predictions.labels, predictions.probabilities, ece_bins=ece_bins
    )
    grouped: dict[int, BeliefMetrics] = {}
    for requested_size in sorted(set(int(value) for value in predictions.requested_sizes)):
        selected = predictions.requested_sizes == requested_size
        grouped[requested_size] = compute_belief_metrics(
            predictions.labels[selected],
            predictions.probabilities[selected],
            ece_bins=ece_bins,
        )
    return BeliefEvaluation(overall=overall, by_requested_size=grouped, predictions=predictions)


def evaluate_belief(
    model: GatedAttentionMIL,
    loader: Iterable[BeliefBatch | Mapping[str, Any]],
    *,
    temperature: float = 1.0,
    device: str | torch.device | None = None,
    ece_bins: int = 15,
) -> BeliefEvaluation:
    predictions = collect_belief_predictions(model, loader, temperature=temperature, device=device)
    return evaluate_predictions(predictions, ece_bins=ece_bins)


def evaluate_full_bags_streaming(
    model: GatedAttentionMIL,
    slides: Iterable[SlideEmbeddingRecord],
    *,
    temperature: float = 1.0,
    device: str | torch.device | None = None,
    ece_bins: int = 15,
    chunk_rows: int = 4096,
) -> BeliefEvaluation:
    """Evaluate complete slides in bounded memory via the exact accumulator.

    Feature providers are sliced a chunk at a time, but evidence is accumulated
    instance-by-instance in canonical magnification/row order.  The result is
    algebraically equivalent to a single full-bag ABMIL forward pass while
    avoiding a potentially enormous padded batch.
    """

    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    slides = tuple(slides)
    if not slides:
        raise ValueError("Full-bag evaluation requires at least one slide")
    original_device = next(model.parameters()).device
    target_device = torch.device(device) if device is not None else original_device
    was_training = model.training
    model.to(target_device).eval()
    logits: list[float] = []
    probabilities: list[float] = []
    labels: list[int] = []
    sizes: list[int] = []
    slide_ids: list[str] = []
    try:
        for slide in slides:
            if slide.embedding_dim != model.embedding_dim:
                raise ValueError(
                    f"Slide {slide.slide_id!r} dimension {slide.embedding_dim} does not "
                    f"match model dimension {model.embedding_dim}"
                )
            accumulator = BeliefAccumulator(
                model,
                temperature=temperature,
                freeze=False,
            )
            for magnification in sorted(slide.features_by_magnification):
                matrix = slide.features_by_magnification[magnification]
                for start in range(0, int(matrix.shape[0]), chunk_rows):
                    stop = min(start + chunk_rows, int(matrix.shape[0]))
                    raw_rows = matrix[start:stop]
                    if isinstance(raw_rows, Tensor):
                        rows = raw_rows.detach().to(device="cpu", dtype=torch.float32)
                    else:
                        rows = torch.from_numpy(np.array(raw_rows, dtype=np.float32, copy=True))
                    if rows.ndim != 2 or rows.shape[1] != model.embedding_dim:
                        raise ValueError(
                            f"Provider for slide {slide.slide_id!r} at {magnification}x "
                            f"returned {tuple(rows.shape)}; expected [N,{model.embedding_dim}]"
                        )
                    for offset, feature in enumerate(rows):
                        accumulator.add(
                            feature,
                            magnification,
                            instance_key=(magnification, start + offset),
                        )
            raw_logit = accumulator.logit
            logits.append(raw_logit / temperature)
            probabilities.append(accumulator.probability)
            labels.append(slide.label)
            sizes.append(accumulator.evidence_count)
            slide_ids.append(slide.slide_id)
    finally:
        model.to(original_device).train(was_training)
    predictions = BeliefPredictions(
        logits=np.asarray(logits, dtype=np.float64),
        probabilities=np.asarray(probabilities, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        requested_sizes=np.asarray(sizes, dtype=np.int64),
        actual_sizes=np.asarray(sizes, dtype=np.int64),
        slide_ids=tuple(slide_ids),
    )
    return evaluate_predictions(predictions, ece_bins=ece_bins)


__all__ = [
    "BeliefEvaluation",
    "BeliefMetrics",
    "BeliefPredictions",
    "binary_auroc",
    "collect_belief_predictions",
    "compute_belief_metrics",
    "evaluate_belief",
    "evaluate_full_bags_streaming",
    "evaluate_predictions",
    "expected_calibration_error",
]
