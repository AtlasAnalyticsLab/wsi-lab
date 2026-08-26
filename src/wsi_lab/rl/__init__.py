"""Recurrent PPO components for WSI Lab.

The implementation is local to :mod:`wsi_lab`; CleanRL is a design
reference, not a runtime dependency.
"""

from wsi_lab.rl.checkpoints import (
    find_latest_checkpoint,
    load_ppo_checkpoint,
    save_ppo_checkpoint,
)
from wsi_lab.rl.distributions import MaskedCategorical, mask_logits
from wsi_lab.rl.evaluation import evaluate_baseline, evaluate_policy
from wsi_lab.rl.networks import RecurrentActorCritic
from wsi_lab.rl.recurrent_ppo import PPOConfig, PPOUpdateMetrics, ppo_update
from wsi_lab.rl.rollout import RecurrentRolloutBuffer, SequenceBatch, compute_gae
from wsi_lab.rl.trainer import PPOTrainer, make_vector_env

__all__ = [
    "MaskedCategorical",
    "PPOConfig",
    "PPOTrainer",
    "PPOUpdateMetrics",
    "RecurrentActorCritic",
    "RecurrentRolloutBuffer",
    "SequenceBatch",
    "compute_gae",
    "evaluate_baseline",
    "evaluate_policy",
    "find_latest_checkpoint",
    "load_ppo_checkpoint",
    "make_vector_env",
    "mask_logits",
    "ppo_update",
    "save_ppo_checkpoint",
]
