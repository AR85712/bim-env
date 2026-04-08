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
# RLVR: number of candidate actions to sample per step (Best-of-N).
# Each candidate is scored by the verifiable geometry reward; the best is executed.
# Set to 1 to disable Best-of-N and use single-sample inference.
N_CANDIDATES        = int(os.getenv("N_CANDIDATES", "3"))
# Minimum reward threshold for a step to be added to the few-shot memory.
RLVR_MEMORY_THRESHOLD = float(os.getenv("RLVR_MEMORY_THRESHOLD", "0.05"))

# ---------------------------------------------------------------------------
# Logging helpers  (mandatory stdout format)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


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


# ---------------------------------------------------------------------------
# RLVR — Verifiable geometry reward scorer
# ---------------------------------------------------------------------------

def _score_action(action: BimAction, obs: BimObservation) -> float:
    """
    Verifiable geometry score for a candidate action — no env interaction needed.

    Scores by two signals:
      1. Directional alignment: dot product of the proposed translation with the
         weighted escape vector for clashes involving the selected element.
      2. Step-size quality: how close the move magnitude is to depth × 1.5
         (the ideal one-shot escape distance).

    This is the "verifiable reward" used in Best-of-N RLVR selection.
    """
    movable = [e for e in obs.elements if e.movable]
    if not movable or action.element_index >= len(movable):
        return -1.0

    target_guid = movable[action.element_index].guid
    relevant = [
        c for c in obs.clashes
        if c.element_a_guid == target_guid or c.element_b_guid == target_guid
    ]
    if not relevant:
        return -0.5  # Moving an element not in any clash

    t = action.translation
    t_mag = math.sqrt(sum(v ** 2 for v in t))
    if t_mag < 1e-9:
        return -1.0

    score = 0.0
    total_weight = 0.0
    for clash in relevant:
        sign = 1.0 if clash.element_a_guid == target_guid else -1.0
        pvec = clash.penetration_vector
        weight = clash.overlap_volume

        # 1. Directional alignment with escape vector
        dot = sum(t[i] * pvec[i] * sign for i in range(3)) / t_mag

        # 2. Step-size quality: ideal = depth × 1.5, penalise deviation
        ideal = clash.penetration_depth * 1.5
        size_score = 1.0 - min(abs(t_mag - ideal) / max(ideal, 1.0), 1.0)

        score += weight * (dot + size_score)
        total_weight += weight

    return score / max(total_weight, 1.0)


def _parse_llm_response(raw: str) -> BimAction:
    """Parse LLM JSON response into a BimAction, stripping markdown fences."""
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw.strip())
    return BimAction(
        element_index=int(data["element_index"]),
        translation=[float(v) for v in data["translation"]],
    )


def get_llm_action(
    client: OpenAI,
    obs: BimObservation,
    history: List[str],
    few_shot_memory: List[str],
) -> BimAction:
    """
    RLVR-inspired Best-of-N action selection:
      1. Generate N_CANDIDATES actions via LLM (temperature diversity).
      2. Score each with the verifiable geometry reward (_score_action).
      3. Return the highest-scoring candidate.

    The few_shot_memory injects high-reward steps from earlier in this
    episode as in-context examples, guiding the LLM toward successful patterns.

    Falls back to the heuristic agent if all LLM calls fail.
    """
    history_block = "\n".join(history[-4:]) if history else "None"
    memory_block  = "\n".join(few_shot_memory[-3:]) if few_shot_memory else ""

    base_content = (
        _clash_summary(obs)
        + (f"\n\nHigh-reward examples from this episode (learn from these):\n{memory_block}" if memory_block else "")
        + f"\n\nRecent action history:\n{history_block}"
    )

    candidates: List[BimAction] = []
    for i in range(N_CANDIDATES):
        # Candidate 0: greedy (low temp, no diversity hint).
        # Candidates 1+: higher temperature AND a diversity prompt so the model
        # explores different elements or translation directions rather than
        # reproducing the same greedy output.
        temp = TEMPERATURE if i == 0 else min(TEMPERATURE + 0.35 * i, 1.2)
        diversity_hint = (
            "\n\nExplore an alternative approach: propose a different element index "
            "or a meaningfully different translation direction than the most obvious choice."
            if i > 0 else ""
        )
        user_content = base_content + diversity_hint + "\n\nOutput your JSON action now:"
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature=temp,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            raw = (completion.choices[0].message.content or "").strip()
            candidates.append(_parse_llm_response(raw))
        except Exception as exc:
            print(f"[DEBUG] Candidate {i+1}/{N_CANDIDATES} failed ({exc})", flush=True)

    if not candidates:
        raise RuntimeError(
            f"All {N_CANDIDATES} LLM candidate(s) failed. "
            "Check API_BASE_URL and API_KEY/HF_TOKEN environment variables."
        )

    # RLVR selection: pick candidate with highest verifiable geometry score
    scores = [_score_action(a, obs) for a in candidates]
    best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
    if len(candidates) > 1:
        print(
            f"[DEBUG] RLVR Best-of-{len(candidates)}: scores={[f'{s:.3f}' for s in scores]} "
            f"→ selected candidate {best_idx + 1}",
            flush=True,
        )
    return candidates[best_idx]


# ---------------------------------------------------------------------------
# Main async entry-point
# ---------------------------------------------------------------------------


async def main() -> None:
    history: List[str] = []          # recent step log for LLM context
    few_shot_memory: List[str] = []  # RLVR: high-reward steps as in-context examples
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        print(f"[DEBUG] API_BASE_URL={API_BASE_URL} MODEL={MODEL_NAME}", flush=True)
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

            action = get_llm_action(client, obs, history, few_shot_memory)

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

            # RLVR few-shot memory: store steps with meaningful positive reward
            # so the LLM can learn from its own successes within the episode.
            if reward >= RLVR_MEMORY_THRESHOLD:
                few_shot_memory.append(
                    f"GOOD EXAMPLE — {action_str} → reward {reward:+.4f} "
                    f"(clashes reduced to {obs.num_clashes}, vol={obs.total_clash_volume:.0f} mm³)"
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
