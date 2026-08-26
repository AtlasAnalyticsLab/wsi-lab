"""Recurrent PPO components for RL4XS WSI.

The implementation is local to :mod:`rl4xs_wsi`; CleanRL is a design
reference, not a runtime dependency.
"""

from rl4xs_wsi.rl.checkpoints import (
    find_latest_checkpoint,
    load_ppo_checkpoint,
    save_ppo_checkpoint,
)
from rl4xs_wsi.rl.distributions import MaskedCategorical, mask_logits
from rl4xs_wsi.rl.evaluation import evaluate_baseline, evaluate_policy
from rl4xs_wsi.rl.networks import RecurrentActorCritic
from rl4xs_wsi.rl.recurrent_ppo import PPOConfig, PPOUpdateMetrics, ppo_update
from rl4xs_wsi.rl.rollout import RecurrentRolloutBuffer, SequenceBatch, compute_gae
from rl4xs_wsi.rl.trainer import PPOTrainer, make_vector_env

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
