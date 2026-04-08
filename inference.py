"""
BIM Clash/Conflict Resolution — Inference Script
=================================================
MANDATORY environment variables:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.
    LOCAL_IMAGE_NAME    Docker image name for the BIM environment.
                        (used by BimEnv.from_docker_image())

Defaults (override via env vars):
    API_BASE_URL  = "https://router.huggingface.co/v1"
    MODEL_NAME    = "Qwen/Qwen2.5-72B-Instruct"
    TASK          = "medium"   (easy | medium | hard)

The LLM agent receives the current clash state as a structured prompt and must
reply with a JSON action:  {"element_index": <int>, "translation": [dx, dy, dz]}
The heuristic fallback is used when the model output cannot be parsed.

STDOUT FORMAT (mandatory):
    [START] task=<task> env=bim_env model=<model>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Score is the final episode `grade` (0.0 – 1.0) from the environment grader.
"""

import asyncio
import json
import math
import os
import textwrap
from typing import Dict, List, Optional

from openai import OpenAI

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from bim_env import BimAction, BimEnv, BimObservation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME", "bim-env:latest")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
TASK_NAME    = os.getenv("TASK", "medium")
BENCHMARK    = "bim_env"

MAX_STEPS           = int(os.getenv("MAX_STEPS", "30"))
TEMPERATURE         = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS          = int(os.getenv("MAX_TOKENS", "200"))
STEP_SIZE_FALLBACK  = float(os.getenv("STEP_SIZE", "400.0"))   # mm, heuristic fallback
SUCCESS_SCORE_THRESHOLD = 0.70   # grade >= 0.70 means all clashes resolved
HEURISTIC_ONLY      = os.getenv("HEURISTIC_ONLY", "").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Logging helpers  (mandatory stdout format)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Heuristic fallback agent  (no LLM required)
# ---------------------------------------------------------------------------


def heuristic_action(obs: BimObservation, step_size: float = STEP_SIZE_FALLBACK) -> BimAction:
    """
    Greedy clash resolver used when the LLM output is unparseable.

    Picks the movable element with the highest total clash involvement and
    moves it along the weighted-average penetration vector scaled by the
    maximum penetration depth (× 1.5 margin) so the element fully escapes
    the clashing body in a single step rather than oscillating.
    """
    if not obs.clashes:
        return BimAction(element_index=0, translation=[0.0, 0.0, 0.0])

    guid_to_index: Dict[str, int] = {
        el.guid: el.movable_index
        for el in obs.elements
        if el.movable and el.movable_index is not None
    }

    involvement: Dict[str, float] = {}   # key: (idx, axis) but simplified to idx
    escape: Dict[int, List[float]] = {}
    max_depth: Dict[int, float] = {}

    for clash in obs.clashes:
        for guid, sign in [(clash.element_a_guid, 1.0), (clash.element_b_guid, -1.0)]:
            if guid not in guid_to_index:
                continue
            idx = guid_to_index[guid]
            vol = clash.overlap_volume
            involvement[idx] = involvement.get(idx, 0.0) + vol
            weighted = [v * vol * sign for v in clash.penetration_vector]
            if idx not in escape:
                escape[idx] = [0.0, 0.0, 0.0]
            for i in range(3):
                escape[idx][i] += weighted[i]
            # Track maximum penetration depth to size the step correctly
            max_depth[idx] = max(max_depth.get(idx, 0.0), clash.penetration_depth)

    if not involvement:
        return BimAction(element_index=0, translation=[0.0, 0.0, step_size])

    best = max(involvement, key=lambda k: involvement[k])
    raw = escape.get(best, [0.0, 0.0, 1.0])
    mag = math.sqrt(sum(v ** 2 for v in raw))
    if mag < 1e-9:
        raw, mag = [0.0, 0.0, 1.0], 1.0

    # Scale step to fully escape: penetration_depth * 2.5 gives a comfortable
    # clearance margin and avoids oscillation where 1.5x causes re-entry from
    # the opposite face. Use a small 100 mm floor only for near-zero depth.
    depth = max_depth.get(best, 0.0)
    actual_step = min(max(depth * 2.5, 100.0), 1000.0)

    return BimAction(element_index=best, translation=[v / mag * actual_step for v in raw])


# ---------------------------------------------------------------------------
# LLM prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert BIM (Building Information Modelling) coordination engineer.
    Your task is to resolve geometric clashes between MEP (ducts, pipes, cable trays)
    and structural elements (beams, columns) in a building model.

    At each step you receive the current list of clashes and must output EXACTLY one
    JSON object — nothing else — with this schema:
        {"element_index": <int>, "translation": [<dx>, <dy>, <dz>]}

    Rules:
    - element_index is the 0-based index of the MEP element to move (movable elements only).
    - translation is [dx, dy, dz] in millimetres; each component is clamped to [-1000.0, 1000.0] by the env.
    - Each clash provides penetration_depth (mm) and penetration_vector [dx, dy, dz].
    - Compute the escape translation as: penetration_vector × (penetration_depth × 1.5).
      This guarantees the element fully clears the obstruction in ONE step rather than many small nudges.
    - When multiple clashes involve the same element, use the largest penetration_depth to size the move.
    - Prioritise the movable element involved in the most or largest clashes.
    - Do NOT add any explanation — output only the JSON object.
""").strip()


def _clash_summary(obs: BimObservation) -> str:
    """Build a compact clash summary for the LLM prompt."""
    lines = [
        f"Task: {obs.task} | Step: {obs.step_number} | "
        f"Clashes: {obs.num_clashes} | Total overlap volume: {obs.total_clash_volume:.0f} mm\u00b3",
        f"Movable elements: {obs.num_movable_elements}",
        "",
        "Movable elements (index | ifc_class | current bbox_min | displacement):",
    ]
    for el in obs.elements:
        if el.movable:
            lines.append(
                f"  [{el.movable_index}] {el.ifc_class:30s} "
                f"min={el.bbox_min}  disp={el.displacement:.1f} mm"
            )

    lines += ["", "Active clashes (element_a -> element_b | overlap_vol | depth | pvec | recommended_step_mm):"]
    for c in obs.clashes[:10]:   # cap at 10 to stay within token budget
        recommended = [round(v * c.penetration_depth * 1.5, 1) for v in c.penetration_vector]
        lines.append(
            f"  {c.element_a_class} vs {c.element_b_class} | "
            f"vol={c.overlap_volume:.0f} mm\u00b3 | depth={c.penetration_depth:.1f} mm | "
            f"pvec={c.penetration_vector} | recommended_translation={recommended}"
        )
    if len(obs.clashes) > 10:
        lines.append(f"  ... and {len(obs.clashes) - 10} more clashes")

    return "\n".join(lines)


def get_llm_action(
    client: OpenAI,
    obs: BimObservation,
    history: List[str],
) -> BimAction:
    """
    Ask the LLM for the next action given the current observation.
    Falls back to the heuristic agent if the response cannot be parsed.
    """
    history_block = "\n".join(history[-4:]) if history else "None"
    user_content = (
        _clash_summary(obs)
        + f"\n\nRecent history:\n{history_block}"
        + "\n\nOutput your JSON action now:"
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Extract JSON even if the model wraps it in markdown fences
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)
        return BimAction(
            element_index=int(data["element_index"]),
            translation=[float(v) for v in data["translation"]],
        )
    except Exception as exc:
        print(f"[DEBUG] LLM parse failed ({exc}), using heuristic fallback.", flush=True)
        return heuristic_action(obs)


# ---------------------------------------------------------------------------
# Main async entry-point
# ---------------------------------------------------------------------------


async def main() -> None:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        print(f"[DEBUG] API_BASE_URL={API_BASE_URL} MODEL={MODEL_NAME} HEURISTIC_ONLY={HEURISTIC_ONLY}", flush=True)
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        # Connect to a running server, or spin up a Docker container as fallback
        base_url = os.getenv("BIM_SERVER_URL", "")
        if base_url:
            env = BimEnv(base_url=base_url)
            await env.__aenter__()
        else:
            env = await BimEnv.from_docker_image(IMAGE_NAME)

        # Task is set via the TASK env var on the server; just reset.
        result = await env.reset()

        obs = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = heuristic_action(obs) if HEURISTIC_ONLY else get_llm_action(client, obs, history)

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done   = result.done
            error  = None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            action_str = (
                f"move(idx={action.element_index},"
                f"t=[{action.translation[0]:.2f},"
                f"{action.translation[1]:.2f},"
                f"{action.translation[2]:.2f}])"
            )
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action_str} -> reward {reward:+.4f} | "
                f"clashes={obs.num_clashes} vol={obs.total_clash_volume:.4f}"
            )

            if done:
                break

        # Score = final grader value (already in [0, 1])
        score = obs.grade
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    import sys
    try:
        asyncio.run(main())
    except Exception as exc:
        # Last-resort guard: ensure a valid [END] line is always printed
        # so the evaluator never sees a non-zero exit without output.
        print(f"[DEBUG] Fatal error: {exc}", flush=True)
        print("[END] success=false steps=0 score=0.000 rewards=", flush=True)
    sys.exit(0)
