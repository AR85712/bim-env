"""
Data models for the BIM Clash/Conflict Resolution Environment.

An RL environment where an agent learns to resolve geometric clashes between
MEP (Mechanical, Electrical, Plumbing) and structural elements in a BIM model.

The agent observes bounding-box geometry of conflicting elements and their
IFC classes, then decides which element to translate and by how much.
"""

from typing import Dict, List, Optional

import json

from openenv.core import Action, Observation
from pydantic import BaseModel, Field, field_validator


class ElementInfo(BaseModel):
    """Represents a BIM element with its current geometry and metadata."""

    guid: str = Field(..., description="Unique element identifier (GUID)")
    ifc_class: str = Field(
        ..., description="IFC class name (e.g. IfcBeam, IfcDuctSegment, IfcPipe)"
    )
    discipline: str = Field(
        ...,
        description="Engineering discipline: structural | mechanical | plumbing | electrical",
    )
    movable: bool = Field(
        ...,
        description="True for MEP elements (can be repositioned); False for structural",
    )
    movable_index: Optional[int] = Field(
        default=None,
        description="Index within movable elements list (None for fixed elements)",
    )
    bbox_min: List[float] = Field(
        ..., description="Bounding box minimum corner [x, y, z] in millimetres"
    )
    bbox_max: List[float] = Field(
        ..., description="Bounding box maximum corner [x, y, z] in millimetres"
    )
    original_bbox_min: List[float] = Field(
        ..., description="Original bbox_min before any agent moves"
    )
    displacement: float = Field(
        default=0.0,
        description="Total Euclidean displacement from original position (millimetres)",
    )


class ClashInfo(BaseModel):
    """Represents a geometric clash between two BIM elements."""

    element_a_guid: str = Field(..., description="GUID of first clashing element")
    element_b_guid: str = Field(..., description="GUID of second clashing element")
    element_a_class: str = Field(..., description="IFC class of first element")
    element_b_class: str = Field(..., description="IFC class of second element")
    overlap_volume: float = Field(
        ..., description="Overlapping AABB volume in cubic millimetres"
    )
    penetration_depth: float = Field(
        ..., description="Maximum penetration depth across any single axis (millimetres)"
    )
    penetration_vector: List[float] = Field(
        ...,
        description="Suggested unit displacement [dx, dy, dz] to move element_a away from element_b",
    )


class BimAction(Action):
    """
    Action to resolve a BIM geometric clash.

    The agent selects a *movable* element (MEP element) by its index in the
    movable-elements list and applies a translation vector.  The translation is
    clamped server-side to ±1000.0 mm per component so the agent can safely output
    raw continuous values.

    Special action — set task difficulty before the next reset::

        BimAction(element_index=-1, translation=[0,0,0], task="easy")

    Normal move action::

        # Move movable element 0 upward by 600 mm on the medium task
        BimAction(element_index=0, translation=[0.0, 0.0, 600.0])
    """

    element_index: int = Field(
        ...,
        description=(
            "0-based index into the movable (MEP) element list. "
            "Use -1 to set the task difficulty via the `task` field (no geometry change)."
        ),
    )
    translation: List[float] = Field(
        ..., description="[dx, dy, dz] translation in millimetres (clamped to ±1000.0 per step)"
    )
    task: str = Field(
        default="medium",
        description="Task difficulty level: 'easy' | 'medium' | 'hard'. "
                    "Effective on next reset() or when element_index == -1.",
    )

    @field_validator("translation", mode="before")
    @classmethod
    def _coerce_translation(cls, v):
        """Accept a JSON string '[0,0,600]' in addition to a native list.
        The playground UI submits list-type fields as strings."""
        if isinstance(v, str):
            v = v.strip()
            parsed = json.loads(v)
            if not isinstance(parsed, list):
                raise ValueError("translation must be a list of three numbers")
            return parsed
        return v


class BimObservation(Observation):
    """
    Full observation of the BIM model state after a step.

    Contains the current bounding boxes of every element, a list of active
    clashes, scalar clash statistics, and a human-readable status message.
    """

    elements: List[ElementInfo] = Field(
        default_factory=list, description="All elements in the scene (fixed + movable)"
    )
    clashes: List[ClashInfo] = Field(
        default_factory=list, description="Active geometric clashes"
    )
    total_clash_volume: float = Field(
        default=0.0, description="Sum of all overlap volumes in mm³"
    )
    num_clashes: int = Field(default=0, description="Number of active clashes")
    num_movable_elements: int = Field(
        default=0, description="Number of movable MEP elements"
    )
    step_number: int = Field(default=0, description="Episode step counter")
    all_clashes_resolved: bool = Field(
        default=False, description="True when total_clash_volume == 0"
    )
    message: str = Field(default="", description="Human-readable status message")
    # ── Grading fields ──────────────────────────────────────────────────────
    task: str = Field(
        default="medium",
        description="Active task difficulty: 'easy' | 'medium' | 'hard'",
    )
    initial_clash_volume: float = Field(
        default=0.0,
        description="Total clash volume at episode start (used by grader).",
    )
    grade: float = Field(
        default=0.0,
        description=(
            "Grader score for this episode: 0.0 (no progress) → 1.0 (perfect). "
            "0.0–0.69 = partial, 0.70–1.0 = fully resolved (efficiency bonus)."
        ),
    )
