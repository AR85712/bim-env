"""
BIM Clash/Conflict Resolution Environment  —  v2  (3 tasks + grader).

Three difficulty tasks
-----------------------
easy   1 IfcBeam, 1 IfcPipe, 1 guaranteed clash,                  MAX_STEPS=15
medium 2 structural (IfcBeam + IfcColumn), 2 MEP, 2-4 clashes,   MAX_STEPS=30
hard   3 IfcBeams + 1 IfcColumn, 4-5 MEP elements, many clashes,  MAX_STEPS=50

Each reset() creates a new randomised scenario at the current task level.
The RNG is seeded with the episode_id UUID so each episode is reproducible.

Grader (0.0 – 1.0)
-------------------
The `grade` field in every BimObservation reports the current episode score:

    fully resolved  ->  0.70 + 0.30 * (1 - steps / max_steps)   [0.70 - 1.00]
    partial         ->  0.70 * (vol_reduced / initial_vol)         [0.00 - 0.69]

Reward (per-step, dense partial-progress signal)
-------------------------------------------------
    r = progress_fraction                        # dVol / initial_vol in [0,1]
      + 0.05 * d_num_clashes                    # per-clash-resolved bonus
      - 0.000005 * displacement_magnitude        # keep moves minimal
      - 0.5 / max_steps                         # time cost (sums to -0.5 max)
    on full resolution:  + 0.5 + 0.3 * efficiency
     where efficiency = 1 - steps / max_steps

Setting the task
----------------
Option A -- env var before starting the server:
    TASK=easy uvicorn bim_env.server.app:app --port 8001

Option B -- configure action at runtime (no geometry change, no reward):
    BimAction(element_index=-1, translation=[0,0,0], task="hard")
    # then call reset()
"""

import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import BimAction, BimObservation, ClashInfo, ElementInfo
except ImportError:
    from models import BimAction, BimObservation, ClashInfo, ElementInfo

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

_TASK_CONFIG = {
    "easy":   {"max_steps": 15, "n_beams": 1, "n_mep_min": 1, "n_mep_max": 1},
    "medium": {"max_steps": 30, "n_beams": 2, "n_mep_min": 2, "n_mep_max": 2},
    "hard":   {"max_steps": 50, "n_beams": 3, "n_mep_min": 4, "n_mep_max": 5},
}

VALID_TASKS = frozenset(_TASK_CONFIG.keys())


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------


@dataclass
class _Element:
    guid: str
    ifc_class: str
    discipline: str
    movable: bool
    bbox_min: List[float]
    bbox_max: List[float]
    original_min: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.original_min:
            self.original_min = list(self.bbox_min)

    def center(self) -> List[float]:
        return [(self.bbox_min[i] + self.bbox_max[i]) / 2.0 for i in range(3)]

    def translate(self, dx: float, dy: float, dz: float) -> None:
        for i, d in enumerate([dx, dy, dz]):
            self.bbox_min[i] += d
            self.bbox_max[i] += d

    def displacement(self) -> float:
        return math.sqrt(
            sum((self.bbox_min[i] - self.original_min[i]) ** 2 for i in range(3))
        )


def _aabb_overlap_volume(
    mn_a: List[float], mx_a: List[float],
    mn_b: List[float], mx_b: List[float],
) -> float:
    ox = max(0.0, min(mx_a[0], mx_b[0]) - max(mn_a[0], mn_b[0]))
    oy = max(0.0, min(mx_a[1], mx_b[1]) - max(mn_a[1], mn_b[1]))
    oz = max(0.0, min(mx_a[2], mx_b[2]) - max(mn_a[2], mn_b[2]))
    return ox * oy * oz


def _aabb_penetration(
    mn_a: List[float], mx_a: List[float],
    mn_b: List[float], mx_b: List[float],
) -> Tuple[float, List[float]]:
    """
    Return (penetration_depth, unit_vector) where the vector points the
    direction element_a should move to resolve the clash along the
    minimum-resistance axis.
    """
    overlaps = [min(mx_a[i], mx_b[i]) - max(mn_a[i], mn_b[i]) for i in range(3)]
    valid = [(abs(o), i) for i, o in enumerate(overlaps) if o > 0]
    if not valid:
        return 0.0, [0.0, 0.0, 0.0]

    depth, axis = min(valid)
    center_a = [(mn_a[i] + mx_a[i]) / 2.0 for i in range(3)]
    center_b = [(mn_b[i] + mx_b[i]) / 2.0 for i in range(3)]
    direction = [0.0, 0.0, 0.0]
    direction[axis] = 1.0 if center_a[axis] >= center_b[axis] else -1.0
    return depth, direction


# ---------------------------------------------------------------------------
# Scenario generators  (easy / medium / hard)
# ---------------------------------------------------------------------------

_MEP_SPECS = [
    # (ifc_class, discipline, half_xy, half_z)  -- all in mm
    ("IfcDuctSegment",         "mechanical", 250.0, 150.0),
    ("IfcPipe",                "plumbing",    70.0,  70.0),
    ("IfcPipe",                "plumbing",    50.0,  50.0),
    ("IfcCableCarrierSegment", "electrical", 150.0,  60.0),
    ("IfcDuctSegment",         "mechanical", 200.0, 120.0),
]

_BEAM_HW = 150.0   # half web-width  (mm)
_BEAM_HH = 250.0   # half depth      (mm)
_COLUMN_HW = 175.0  # half side of a 350×350 mm column cross-section (mm)


def _add_beam_x(elements: List[_Element], y: float, z: float) -> _Element:
    e = _Element(
        guid=str(uuid4()), ifc_class="IfcBeam",
        discipline="structural", movable=False,
        bbox_min=[0.0, y - _BEAM_HW, z - _BEAM_HH],
        bbox_max=[10000.0, y + _BEAM_HW, z + _BEAM_HH],
    )
    elements.append(e)
    return e


def _add_beam_y(elements: List[_Element], x: float, z: float) -> _Element:
    e = _Element(
        guid=str(uuid4()), ifc_class="IfcBeam",
        discipline="structural", movable=False,
        bbox_min=[x - _BEAM_HW, 0.0, z - _BEAM_HH],
        bbox_max=[x + _BEAM_HW, 10000.0, z + _BEAM_HH],
    )
    elements.append(e)
    return e


def _add_column(elements: List[_Element], x: float, y: float,
                z_bottom: float = 0.0, z_top: float = 4000.0) -> _Element:
    """Adds a vertical IfcColumn at (x, y) spanning z_bottom→z_top."""
    e = _Element(
        guid=str(uuid4()), ifc_class="IfcColumn",
        discipline="structural", movable=False,
        bbox_min=[x - _COLUMN_HW, y - _COLUMN_HW, z_bottom],
        bbox_max=[x + _COLUMN_HW, y + _COLUMN_HW, z_top],
    )
    elements.append(e)
    return e


def _add_mep_through_structural(
    elements: List[_Element],
    rng: random.Random,
    structural: _Element,
    spec_idx: int,
    run_vertical: bool,
) -> None:
    """
    Adds a MEP element guaranteed to clash with `structural`.

    Works for both IfcBeam (horizontal) and IfcColumn (vertical) by
    centring the MEP placement on the structural element's bounding box.
    """
    cls, disc, r_xy, r_z = _MEP_SPECS[spec_idx % len(_MEP_SPECS)]
    bc = structural.center()
    is_column = structural.ifc_class == "IfcColumn"

    if is_column:
        # MEP runs horizontally through the column's footprint.
        # run_vertical flag selects X-run vs Y-run to vary geometry.
        z = bc[2] + rng.uniform(-500.0, 500.0)   # somewhere mid-column
        if run_vertical:  # "run along Y"
            x = bc[0] + rng.uniform(-_COLUMN_HW * 0.4, _COLUMN_HW * 0.4)
            mn = [x - r_xy, 0.0, z - r_z]
            mx = [x + r_xy, 10000.0, z + r_z]
        else:             # "run along X"
            y = bc[1] + rng.uniform(-_COLUMN_HW * 0.4, _COLUMN_HW * 0.4)
            mn = [0.0, y - r_xy, z - r_z]
            mx = [10000.0, y + r_xy, z + r_z]
    elif run_vertical:
        # Vertical MEP (riser) through a horizontal beam
        x = rng.uniform(bc[0] - 400.0, bc[0] + 400.0)
        y = bc[1] + rng.uniform(-_BEAM_HW * 0.4, _BEAM_HW * 0.4)
        mn = [x - r_xy, y - r_xy, 300.0]
        mx = [x + r_xy, y + r_xy, 4000.0]
    else:
        # Horizontal MEP through a horizontal beam
        y = bc[1] + rng.uniform(-_BEAM_HW * 0.4, _BEAM_HW * 0.4)
        z = bc[2] + rng.uniform(-_BEAM_HH * 0.4, _BEAM_HH * 0.4)
        mn = [0.0, y - r_xy, z - r_z]
        mx = [10000.0, y + r_xy, z + r_z]

    elements.append(_Element(
        guid=str(uuid4()), ifc_class=cls, discipline=disc, movable=True,
        bbox_min=mn, bbox_max=mx,
    ))


def _gen_easy_scenario(rng: random.Random) -> List[_Element]:
    """1 IfcBeam (X-axis) + 1 IfcPipe. ~1 clash. Agent budget: 15 steps."""
    elements: List[_Element] = []
    y = rng.uniform(4500.0, 5500.0)
    z = rng.uniform(2800.0, 3200.0)
    beam = _add_beam_x(elements, y, z)
    _add_mep_through_structural(elements, rng, beam, spec_idx=1, run_vertical=False)
    return elements


def _gen_medium_scenario(rng: random.Random) -> List[_Element]:
    """1 IfcBeam + 1 IfcColumn + 2 MEP elements. 2-4 clashes. Budget: 30 steps."""
    elements: List[_Element] = []
    beam = _add_beam_x(elements, rng.uniform(4500.0, 5500.0), rng.uniform(2800.0, 3200.0))
    column = _add_column(
        elements,
        x=rng.uniform(4500.0, 5500.0),
        y=rng.uniform(4500.0, 5500.0),
        z_bottom=0.0,
        z_top=rng.uniform(3500.0, 4500.0),
    )
    _add_mep_through_structural(elements, rng, beam,   spec_idx=0, run_vertical=False)
    _add_mep_through_structural(elements, rng, column, spec_idx=1, run_vertical=False)
    return elements


def _gen_hard_scenario(rng: random.Random) -> List[_Element]:
    """3 IfcBeams + 1 IfcColumn + 4-5 MEP elements. Many clashes. Budget: 50 steps."""
    elements: List[_Element] = []
    beam1 = _add_beam_x(elements, rng.uniform(4500.0, 5500.0), rng.uniform(2800.0, 3200.0))
    beam2 = _add_beam_y(elements, rng.uniform(4500.0, 5500.0), rng.uniform(2800.0, 3200.0))
    beam3 = _add_beam_x(elements, rng.uniform(2500.0, 3500.0), rng.uniform(1800.0, 2400.0))
    column = _add_column(
        elements,
        x=rng.uniform(3000.0, 7000.0),
        y=rng.uniform(3000.0, 7000.0),
        z_bottom=0.0,
        z_top=rng.uniform(3500.0, 4500.0),
    )

    n_mep = rng.randint(4, 5)
    structurals = [beam1, beam2, beam3, column]
    for i in range(n_mep):
        _add_mep_through_structural(
            elements, rng, structurals[i % len(structurals)],
            spec_idx=i, run_vertical=(i % 2 == 0),
        )
    return elements


_SCENARIO_FN = {
    "easy":   _gen_easy_scenario,
    "medium": _gen_medium_scenario,
    "hard":   _gen_hard_scenario,
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class BimEnvironment(Environment):
    """
    BIM Clash/Conflict Resolution RL Environment -- 3-task variant.

    Three difficulty levels: easy / medium / hard.
    Select via TASK env-var or runtime configure action (element_index=-1).
    Each episode is seeded by its episode_id for full reproducibility.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_TRANSLATION_PER_COMPONENT: float = 1000.0  # mm, hard ceiling per component

    def __init__(self, task: Optional[str] = None) -> None:
        raw = task or os.environ.get("TASK", "medium")
        self._task: str = raw if raw in VALID_TASKS else "medium"
        self._max_steps: int = _TASK_CONFIG[self._task]["max_steps"]

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._elements: List[_Element] = []
        self._initial_clash_volume: float = 0.0
        self._initial_clash_count: int = 0
        self._prev_clash_volume: float = 0.0
        self._prev_clash_count: int = 0

    # ------------------------------------------------------------------
    # Public helper -- allows direct task selection without HTTP
    # ------------------------------------------------------------------

    def set_task(self, task: str) -> None:
        """Switch task difficulty; takes effect on the next reset()."""
        if task not in VALID_TASKS:
            raise ValueError(f"Unknown task '{task}'. Use one of {sorted(VALID_TASKS)}")
        self._task = task
        self._max_steps = _TASK_CONFIG[task]["max_steps"]

    # ------------------------------------------------------------------
    # OpenEnv core interface
    # ------------------------------------------------------------------

    def reset(self) -> BimObservation:  # type: ignore[override]
        episode_id = str(uuid4())
        self._state = State(episode_id=episode_id, step_count=0)

        rng = random.Random(episode_id)
        self._elements = _SCENARIO_FN[self._task](rng)

        clashes = self._compute_clashes()
        total_vol = sum(c.overlap_volume for c in clashes)

        self._initial_clash_volume = total_vol
        self._initial_clash_count = len(clashes)
        self._prev_clash_volume = total_vol
        self._prev_clash_count = len(clashes)

        movable = [e for e in self._elements if e.movable]
        grade = self._compute_grade(total_vol, 0)

        return BimObservation(
            elements=self._to_element_infos(),
            clashes=clashes,
            total_clash_volume=round(total_vol, 6),
            num_clashes=len(clashes),
            num_movable_elements=len(movable),
            step_number=0,
            all_clashes_resolved=(total_vol == 0.0),
            task=self._task,
            initial_clash_volume=round(self._initial_clash_volume, 6),
            grade=grade,
            done=False,
            reward=0.0,
            message=(
                f"[{self._task.upper()}] Episode started. "
                f"{len(clashes)} clash(es) detected "
                f"(total overlap: {total_vol:.0f} mm\u00b3). "
                f"{len(movable)} movable MEP element(s). "
                f"Budget: {self._max_steps} steps."
            ),
        )

    def step(self, action: BimAction) -> BimObservation:  # type: ignore[override]
        # -- Special configure action (element_index < 0) --------------------
        if action.element_index < 0:
            if action.task in VALID_TASKS:
                self.set_task(action.task)
            clashes = self._compute_clashes()
            vol = sum(c.overlap_volume for c in clashes)
            movable = [e for e in self._elements if e.movable]
            grade = self._compute_grade(vol, self._state.step_count)
            return BimObservation(
                elements=self._to_element_infos(),
                clashes=clashes,
                total_clash_volume=round(vol, 6),
                num_clashes=len(clashes),
                num_movable_elements=len(movable),
                step_number=self._state.step_count,
                all_clashes_resolved=(vol == 0.0),
                task=self._task,
                initial_clash_volume=round(self._initial_clash_volume, 6),
                grade=grade,
                done=False,
                reward=0.0,
                message=f"Task configured to '{self._task}'. Call reset() to start episode.",
            )

        # -- Normal step ------------------------------------------------------
        self._state.step_count += 1
        step = self._state.step_count

        movable = [e for e in self._elements if e.movable]
        n_movable = len(movable)

        if not movable:
            return self._error_obs("No movable elements in scene.", step)

        # Clamp and apply translation
        idx = int(action.element_index) % n_movable
        raw = list(action.translation)[:3]
        while len(raw) < 3:
            raw.append(0.0)
        c = self.MAX_TRANSLATION_PER_COMPONENT
        dx, dy, dz = (max(-c, min(c, v)) for v in raw)

        target = movable[idx]
        target.translate(dx, dy, dz)

        # Recompute clashes
        clashes = self._compute_clashes()
        total_vol = sum(cl.overlap_volume for cl in clashes)
        num_clashes = len(clashes)

        # -- Reward: normalized, multi-signal --------------------------------
        denom = max(self._initial_clash_volume, 1e-9)

        # 1. Dense progress fraction: fraction of initial volume resolved THIS step
        progress_fraction = (self._prev_clash_volume - total_vol) / denom

        # 2. Per-clash bonus: each fully resolved clash gets 0.05
        clash_delta_bonus = 0.05 * max(0, self._prev_clash_count - num_clashes)

        # 3. Displacement penalty: discourages unnecessarily large moves
        displacement_penalty = 0.000005 * math.sqrt(dx**2 + dy**2 + dz**2)

        # 4. Time cost: distributed evenly so total time cost = -0.5 if unused
        time_cost = 0.5 / self._max_steps

        reward = progress_fraction + clash_delta_bonus - displacement_penalty - time_cost

        # 5. Terminal bonus for full resolution (reward: +0.5 to +0.8)
        done = (total_vol == 0.0) or (step >= self._max_steps)
        if total_vol == 0.0:
            efficiency = max(0.0, 1.0 - step / self._max_steps)
            reward += 0.5 + 0.3 * efficiency

        self._prev_clash_volume = total_vol
        self._prev_clash_count = num_clashes

        grade = self._compute_grade(total_vol, step)

        # Build message
        if total_vol == 0.0:
            msg = (
                f"[{self._task.upper()}] All clashes resolved in {step} step(s)! "
                f"Grade: {grade:.2f}. "
                f"Element displacement: {target.displacement():.1f} mm."
            )
        elif step >= self._max_steps:
            msg = (
                f"[{self._task.upper()}] Budget exhausted ({self._max_steps} steps). "
                f"{num_clashes} clash(es) remaining. Grade: {grade:.2f}."
            )
        else:
            pct = 100.0 * (1.0 - total_vol / denom)
            msg = (
                f"[{self._task.upper()}] Step {step}/{self._max_steps}: "
                f"moved {target.ifc_class} by ({dx:.0f},{dy:.0f},{dz:.0f}) mm. "
                f"{num_clashes} clash(es), {pct:.1f}% volume resolved."
            )

        return BimObservation(
            elements=self._to_element_infos(),
            clashes=clashes,
            total_clash_volume=round(total_vol, 6),
            num_clashes=num_clashes,
            num_movable_elements=n_movable,
            step_number=step,
            all_clashes_resolved=(total_vol == 0.0),
            task=self._task,
            initial_clash_volume=round(self._initial_clash_volume, 6),
            grade=grade,
            done=done,
            reward=round(reward, 6),
            message=msg,
        )

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Grader  (0.0 - 1.0)
    # ------------------------------------------------------------------

    def _compute_grade(self, current_vol: float, steps_taken: int) -> float:
        """
        0.0-1.0 score for the current episode state.

        Grade tiers
        -----------
        Tier A  fully resolved           0.70 + 0.30 * efficiency  -> [0.70, 1.00]
        Tier B  partial progress         0.70 * progress_ratio     -> [0.00, 0.69]

        Grade thresholds
        ----------------
        ~0.00  no improvement
        ~0.35  50% clash volume removed
        ~0.69  98%+ volume removed (1 small clash remains)
        0.70   all resolved at max steps
        0.85   all resolved at 50% of budget
        1.00   all resolved in 1 step
        """
        denom = max(self._initial_clash_volume, 1e-9)
        if current_vol <= 0.0:
            efficiency = max(0.0, 1.0 - steps_taken / max(self._max_steps, 1))
            g = 0.70 + 0.30 * efficiency
        else:
            progress = (self._initial_clash_volume - current_vol) / denom
            g = 0.70 * max(0.0, min(1.0, progress))
        return round(g, 4)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_clashes(self) -> List[ClashInfo]:
        clashes: List[ClashInfo] = []
        n = len(self._elements)
        for i in range(n):
            a = self._elements[i]
            for j in range(i + 1, n):
                b = self._elements[j]
                # Skip structural-structural clashes: neither element is movable,
                # so the agent cannot resolve them.  Crossing beams always overlap
                # in medium/hard scenarios and would permanently pollute the
                # initial_clash_volume, making full resolution impossible.
                if not a.movable and not b.movable:
                    continue
                vol = _aabb_overlap_volume(
                    a.bbox_min, a.bbox_max, b.bbox_min, b.bbox_max
                )
                if vol > 0.0:
                    depth, pvec = _aabb_penetration(
                        a.bbox_min, a.bbox_max, b.bbox_min, b.bbox_max
                    )
                    clashes.append(ClashInfo(
                        element_a_guid=a.guid,
                        element_b_guid=b.guid,
                        element_a_class=a.ifc_class,
                        element_b_class=b.ifc_class,
                        overlap_volume=round(vol, 6),
                        penetration_depth=round(depth, 4),
                        penetration_vector=[round(v, 4) for v in pvec],
                    ))
        return clashes

    def _to_element_infos(self) -> List[ElementInfo]:
        infos: List[ElementInfo] = []
        movable_idx = 0
        for e in self._elements:
            idx: Optional[int] = None
            if e.movable:
                idx = movable_idx
                movable_idx += 1
            infos.append(ElementInfo(
                guid=e.guid,
                ifc_class=e.ifc_class,
                discipline=e.discipline,
                movable=e.movable,
                movable_index=idx,
                bbox_min=[round(v, 4) for v in e.bbox_min],
                bbox_max=[round(v, 4) for v in e.bbox_max],
                original_bbox_min=[round(v, 4) for v in e.original_min],
                displacement=round(e.displacement(), 4),
            ))
        return infos

    def _error_obs(self, msg: str, step: int) -> BimObservation:
        grade = self._compute_grade(self._prev_clash_volume, step)
        return BimObservation(
            elements=self._to_element_infos(),
            clashes=[], total_clash_volume=0.0, num_clashes=0,
            num_movable_elements=0, step_number=step,
            all_clashes_resolved=False,
            task=self._task,
            initial_clash_volume=round(self._initial_clash_volume, 6),
            grade=grade,
            done=False, reward=-1.0, message=msg,
        )
