---
title: BIM Clash Conflict Resolution RL Environment
emoji: 🏗️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# BIM Clash/Conflict Resolution Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible reinforcement learning environment where an agent learns to resolve geometric clashes between MEP (Mechanical, Electrical, Plumbing) and structural elements in a Building Information Model (BIM).

---

## Overview

In real-world construction projects, MEP services (ducts, pipes, cable trays) frequently clash with structural members (beams, columns) during the coordination phase. Detecting and resolving these clashes manually is time-consuming and error-prone.

This environment simulates that process as a sequential decision-making problem. The agent observes axis-aligned bounding boxes (AABB) of all BIM elements, identifies active clashes, and decides which MEP element to translate and by how much — all in millimetre units matching real IFC model coordinates.

---

## Environment Details

| Property | Value |
|---|---|
| **Coordinate unit** | Millimetres (mm) |
| **Volume unit** | Cubic millimetres (mm³) |
| **Action space** | Continuous — element index + 3D translation vector |
| **Observation space** | Element geometries, clash list, scalar statistics |
| **Episode seeding** | UUID-based (fully reproducible) |
| **Max concurrent sessions** | 4 |
| **Framework** | OpenEnv (`openenv-core >= 0.2.2`) |

### BIM Element Types

| IFC Class | Discipline | Movable |
|---|---|---|
| `IfcBeam` | Structural | No (fixed) |
| `IfcDuctSegment` | Mechanical | Yes |
| `IfcPipe` | Plumbing | Yes |
| `IfcCableCarrierSegment` | Electrical | Yes |

---

## Task Difficulties

Three difficulty levels are available, selected via the `TASK` environment variable or a runtime configure action.

| Task | Beams | MEP Elements | Guaranteed Clashes | Max Steps |
|---|---|---|---|---|
| `easy` | 1 (X-axis) | 1 pipe | 1 | 15 |
| `medium` | 2 (X + Y crossing) | 2 elements | 2–4 | 30 |
| `hard` | 3 | 4–5 elements | many | 50 |

---

## Action

```python
BimAction(
    element_index: int,      # 0-based index into movable (MEP) elements
    translation: [dx, dy, dz],  # mm, clamped to ±1000 mm per component
    task: str,               # "easy" | "medium" | "hard" (for configure action)
)
```

**Special configure action** — switch task difficulty without applying geometry changes:

```python
BimAction(element_index=-1, translation=[0, 0, 0], task="hard")
# then call reset()
```

---

## Observation

```python
BimObservation(
    elements: List[ElementInfo],     # all elements with current bboxes (mm)
    clashes: List[ClashInfo],        # active geometric clashes
    total_clash_volume: float,       # sum of overlap volumes (mm³)
    num_clashes: int,                # number of active clashes
    num_movable_elements: int,       # count of MEP elements
    step_number: int,                # current step in episode
    all_clashes_resolved: bool,      # True when total_clash_volume == 0
    task: str,                       # active difficulty level
    initial_clash_volume: float,     # clash volume at episode start (mm³)
    grade: float,                    # 0.0–1.0 episode score (see Grader)
    message: str,                    # human-readable status
    done: bool,
    reward: float,
)
```

Each `ClashInfo` includes `overlap_volume` (mm³), `penetration_depth` (mm), and a `penetration_vector` hint pointing the escape direction.

---

## Reward Function

A dense, multi-signal reward is computed every step:

```
r = (ΔVolume / initial_volume)          # progress fraction [0, 1]
  + 0.05 × Δnum_clashes                 # per-clash resolved bonus
  - 0.000005 × ‖translation‖            # displacement penalty
  - 0.5 / max_steps                     # time cost

On full resolution (all_clashes_resolved):
  r += 0.5 + 0.3 × efficiency           # terminal bonus [+0.5, +0.8]
  where efficiency = 1 - steps / max_steps
```

---

## Grader (Episode Score)

The `grade` field in every observation reports the current 0.0–1.0 episode score:

| Tier | Condition | Formula | Range |
|---|---|---|---|
| **A** — Fully resolved | `total_clash_volume == 0` | `0.70 + 0.30 × efficiency` | 0.70–1.00 |
| **B** — Partial progress | clashes remain | `0.70 × (vol_reduced / initial_vol)` | 0.00–0.69 |

A grade ≥ `0.70` means all clashes were resolved.

---

## Quick Start

### 1. Install locally

```bash
cd d:/HF-Hackathon
pip install -e bim_env
```

### 2. Run the server

```bash
# From the HF-Hackathon root directory
uvicorn bim_env.server.app:app --host 0.0.0.0 --port 8001

# Or with a specific task difficulty
TASK=hard uvicorn bim_env.server.app:app --host 0.0.0.0 --port 8001
```

### 3. Use the Python client

```python
import asyncio
from bim_env import BimAction, BimEnv

async def main():
    async with BimEnv(base_url="http://localhost:8001") as env:
        obs = await env.reset()
        print(f"Task: {obs.task} | Clashes: {obs.num_clashes}")
        print(f"Initial clash volume: {obs.initial_clash_volume:.0f} mm³")

        for step in range(30):
            if obs.all_clashes_resolved or obs.done:
                break

            # Move element 0 along the first clash's penetration vector
            if obs.clashes:
                clash = obs.clashes[0]
                pvec = clash.penetration_vector
                tx = [v * 400.0 for v in pvec]  # 400 mm step
            else:
                tx = [0.0, 0.0, 400.0]

            result = await env.step(BimAction(element_index=0, translation=tx))
            obs = result.observation
            print(f"Step {obs.step_number}: reward={result.reward:.3f}, "
                  f"clashes={obs.num_clashes}, grade={obs.grade:.2f}")

asyncio.run(main())
```

---

## Docker

### Build

```bash
cd d:/HF-Hackathon/bim_env
docker build -t bim-env:latest .
```

### Run

```bash
docker run -d -p 8002:8000 --name bim-env bim-env:latest
```

### Health check

```bash
curl http://localhost:8002/health
# {"status":"healthy"}
```

### Use from Docker via client

```python
from bim_env import BimEnv

client = BimEnv.from_docker_image("bim-env:latest")
try:
    obs = await client.reset()
    print(obs.message)
finally:
    await client.close()
```

---

## Inference Script

Run a heuristic or LLM-guided agent against the environment:

```bash
# Heuristic agent (no API key needed)
python -m bim_env.inference

# LLM agent (Hugging Face)
$env:HF_TOKEN = "hf_..."
$env:TASK = "medium"
$env:MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
python -m bim_env.inference
```

The script outputs the mandatory `[START]` / `[STEP]` / `[END]` format:

```
[START] task=medium env=bim_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"element_index":0,"translation":[0,0,400]} reward=0.42 done=false error=null
...
[END] success=true steps=12 score=0.880 rewards=0.42,0.38,...
```

### Inference environment variables

| Variable | Default | Description |
|---|---|---|
| `LOCAL_IMAGE_NAME` | `bim-env:latest` | Docker image to launch |
| `HF_TOKEN` / `API_KEY` | — | API key for LLM inference |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | OpenAI-compatible base URL |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model to use |
| `TASK` | `medium` | Task difficulty: `easy`, `medium`, `hard` |
| `MAX_STEPS` | `30` | Override episode step budget |
| `STEP_SIZE` | `400.0` | Heuristic fallback translation magnitude (mm) |

---

## Project Structure

```
bim_env/
├── __init__.py                  # Package exports (BimEnv, BimAction, BimObservation)
├── models.py                    # Pydantic data models (ElementInfo, ClashInfo, BimAction, BimObservation)
├── client.py                    # BimEnv(EnvClient) — async/sync HTTP+WebSocket client
├── inference.py                 # LLM / heuristic inference script
├── openenv.yaml                 # OpenEnv deployment manifest
├── pyproject.toml               # Package metadata and dependencies
├── Dockerfile                   # Multi-stage Docker build
└── server/
    ├── __init__.py
    ├── app.py                   # FastAPI app factory (create_app)
    ├── bim_env_environment.py   # Core MDP engine — scenarios, reward, grader
    └── requirements.txt
```

---

## API Endpoints

Served automatically by `openenv-core`:

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Apply an action |
| `GET` | `/state` | Current episode state |
| `GET` | `/schema` | Action/Observation JSON schemas |
| `WS` | `/ws` | WebSocket stream |

---

## Dependencies

- `openenv-core[core] >= 0.2.2`
- `numpy >= 1.24.0`
- Python `>= 3.10`
