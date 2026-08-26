"""WSI Lab public package."""

from __future__ import annotations

from wsi_lab.constants import ENV_ID, PACKAGE_VERSION

__version__ = PACKAGE_VERSION


def register_environment() -> None:
    try:
        import gymnasium as gym
    except ImportError:
        return
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point="wsi_lab.envs:make_env")


register_environment()

__all__ = ["__version__", "register_environment"]
