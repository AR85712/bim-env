"""BIM Clash/Conflict Resolution Environment."""

from .models import BimAction, BimObservation, ClashInfo, ElementInfo

__all__ = [
    "BimAction",
    "BimObservation",
    "ClashInfo",
    "ElementInfo",
    "BimEnv",
]


def __getattr__(name: str):
    # Lazy import so that gym_env / train.py work without openenv installed.
    if name == "BimEnv":
        from .client import BimEnv
        return BimEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
