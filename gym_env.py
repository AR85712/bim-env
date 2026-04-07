"""
Gymnasium wrapper for BimEnvironment.

Wraps BimEnvironment directly (no HTTP) for fast single-process RL training.
Compatible with stable-baselines3, RLlib, and any Gymnasium-based framework.

Observation vector (112-dim float32, all values normalised to ~[-1, 1]):
    [0   : 35 ]  Movable elements    (MAX_MOVABLE=5):    bbox_min/max (×3×2), displacement
    [35  : 59 ]  Structural elements (MAX_STRUCTURAL=4): bbox_min/max (×3×2)
    [59  : 109]  Top clashes         (MAX_CLASHES=10):   pvec (×3), depth, volume
    [109 : 112]  Global stats:        total_clash_volume, num_clashes, step_fraction

Action space — Box(-1, 1, shape=(MAX_MOVABLE * 3,)):
    Reshaped to (MAX_MOVABLE, 3). The row with the largest L2 norm selects
    which movable element moves; that row × 1000 mm is the translation.
    This keeps the space purely continuous so standard MLP policies work.
"""

from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from .server.bim_env_environment import BimEnvironment
    from .models import BimAction, BimObservation
except ImportError:
    from server.bim_env_environment import BimEnvironment
    from models import BimAction, BimObservation

# ---------------------------------------------------------------------------
# Fixed-size caps (determines observation vector length)
# ---------------------------------------------------------------------------
MAX_MOVABLE    = 5
MAX_STRUCTURAL = 4
MAX_CLASHES    = 10

# OBS_DIM = 5×7 + 4×6 + 10×5 + 3 = 35 + 24 + 50 + 3 = 112
OBS_DIM = MAX_MOVABLE * 7 + MAX_STRUCTURAL * 6 + MAX_CLASHES * 5 + 3

# Normalisation denominators (all in mm / mm³)
_COORD_SCALE = 10_000.0   # 10 m
_VOL_SCALE   = 1e9         # ~1 m³ in mm³
_DEPTH_SCALE = 1_000.0    # 1 m
_DISP_SCALE  = 10_000.0   # 10 m


# ---------------------------------------------------------------------------
# Observation encoder
# ---------------------------------------------------------------------------

def _obs_to_array(obs: BimObservation, max_steps: int) -> np.ndarray:
    """Convert a BimObservation to a fixed-size float32 numpy array."""
    vec = np.zeros(OBS_DIM, dtype=np.float32)
    ptr = 0

    movable    = [e for e in obs.elements if e.movable]
    structural = [e for e in obs.elements if not e.movable]

    # ── Movable elements (MAX_MOVABLE × 7) ─────────────────────────────────
    for i in range(MAX_MOVABLE):
        if i < len(movable):
            e = movable[i]
            vec[ptr:ptr+3] = np.array(e.bbox_min, dtype=np.float32) / _COORD_SCALE
            vec[ptr+3:ptr+6] = np.array(e.bbox_max, dtype=np.float32) / _COORD_SCALE
            vec[ptr+6] = e.displacement / _DISP_SCALE
        ptr += 7

    # ── Structural elements (MAX_STRUCTURAL × 6) ────────────────────────────
    for i in range(MAX_STRUCTURAL):
        if i < len(structural):
            e = structural[i]
            vec[ptr:ptr+3] = np.array(e.bbox_min, dtype=np.float32) / _COORD_SCALE
            vec[ptr+3:ptr+6] = np.array(e.bbox_max, dtype=np.float32) / _COORD_SCALE
        ptr += 6

    # ── Top clashes (MAX_CLASHES × 5) ───────────────────────────────────────
    for i in range(MAX_CLASHES):
        if i < len(obs.clashes):
            c = obs.clashes[i]
            vec[ptr:ptr+3] = np.array(c.penetration_vector, dtype=np.float32)
            vec[ptr+3] = min(c.penetration_depth / _DEPTH_SCALE, 2.0)
            vec[ptr+4] = min(c.overlap_volume / _VOL_SCALE, 1.0)
        ptr += 5

    # ── Global stats (3) ────────────────────────────────────────────────────
    vec[ptr]   = min(obs.total_clash_volume / _VOL_SCALE, 1.0)
    vec[ptr+1] = min(obs.num_clashes / 20.0, 1.0)
    vec[ptr+2] = obs.step_number / max(max_steps, 1)

    return vec


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class BimClashEnv(gym.Env):
    """
    Gymnasium-compatible BIM Clash/Conflict Resolution environment.

    Wraps BimEnvironment directly (no HTTP overhead) for fast RL training.

    Parameters
    ----------
    task : str
        Difficulty level: "easy" | "medium" | "hard"
    render_mode : str | None
        Present for Gymnasium API compliance; no visual rendering supported.

    Examples
    --------
    >>> import gymnasium as gym
    >>> env = BimClashEnv(task="easy")
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": []}

    def __init__(self, task: str = "medium", render_mode: Optional[str] = None):
        super().__init__()
        self.task = task
        self._env = BimEnvironment(task)
        self._max_steps = self._env._max_steps
        self._obs: Optional[BimObservation] = None

        # Observation: 112-dim float32, loosely bounded in [-2, 2]
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # Action: MAX_MOVABLE × 3 continuous values in [-1, 1]
        # Row with max L2 norm → selected element; row × 1000 mm → translation
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(MAX_MOVABLE * 3,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Action decoder
    # ------------------------------------------------------------------

    def _decode_action(self, action: np.ndarray) -> BimAction:
        """Convert normalised action vector to BimAction."""
        groups = action.reshape(MAX_MOVABLE, 3)
        norms = np.linalg.norm(groups, axis=1)

        movable = [e for e in self._obs.elements if e.movable]
        n_valid = len(movable)
        if n_valid == 0:
            return BimAction(element_index=0, translation=[0.0, 0.0, 0.0])

        # Only consider rows for elements that actually exist
        valid_norms = norms[:n_valid]
        elem_idx = int(np.argmax(valid_norms))
        tx = (groups[elem_idx] * 1000.0).tolist()  # scale [-1,1] → [-1000, 1000] mm
        return BimAction(element_index=elem_idx, translation=tx)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._obs = self._env.reset()
        return _obs_to_array(self._obs, self._max_steps), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        bim_action = self._decode_action(action)
        self._obs = self._env.step(bim_action)

        obs_array  = _obs_to_array(self._obs, self._max_steps)
        reward     = float(self._obs.reward)
        terminated = bool(self._obs.all_clashes_resolved)
        truncated  = bool(self._obs.done and not terminated)
        info = {
            "grade":       self._obs.grade,
            "num_clashes": self._obs.num_clashes,
            "step":        self._obs.step_number,
            "task":        self._obs.task,
        }
        return obs_array, reward, terminated, truncated, info

    def render(self):
        pass  # no visual rendering

    def close(self):
        pass
