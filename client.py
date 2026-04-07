"""BIM Clash/Conflict Resolution Environment Client."""

from typing import Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import BimAction, BimObservation, ClashInfo, ElementInfo


class BimEnv(EnvClient[BimAction, BimObservation, State]):
    """
    Client for the BIM Clash/Conflict Resolution Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated episode on the server.

    Quick-start (async)::

        import asyncio
        from bim_env import BimAction, BimEnv

        async def main():
            async with BimEnv(base_url="http://localhost:8000") as client:
                result = await client.reset()
                print(f"Clashes: {result.observation.num_clashes}")

                # Move movable element 0 upward by 0.5 m
                result = await client.step(BimAction(
                    element_index=0,
                    translation=[0.0, 0.0, 0.5],
                ))
                print(f"Reward: {result.reward}")
                print(f"Clashes remaining: {result.observation.num_clashes}")

        asyncio.run(main())

    Quick-start (sync)::

        from bim_env import BimAction, BimEnv

        with BimEnv(base_url="http://localhost:8000").sync() as client:
            result = client.reset()
            result = client.step(BimAction(element_index=0, translation=[0.0, 0.5, 0.0]))
            print(result.observation.message)

    Docker::

        client = BimEnv.from_docker_image("bim-env:latest")
        try:
            result = client.reset()
        finally:
            client.close()
    """

    def _step_payload(self, action: BimAction) -> Dict:
        return {
            "element_index": action.element_index,
            "translation": list(action.translation),
        }

    def _parse_result(self, payload: Dict) -> StepResult[BimObservation]:
        obs_data = payload.get("observation", {})

        elements: List[ElementInfo] = [
            ElementInfo(**e) for e in obs_data.get("elements", [])
        ]
        clashes: List[ClashInfo] = [
            ClashInfo(**c) for c in obs_data.get("clashes", [])
        ]

        observation = BimObservation(
            elements=elements,
            clashes=clashes,
            total_clash_volume=obs_data.get("total_clash_volume", 0.0),
            num_clashes=obs_data.get("num_clashes", 0),
            num_movable_elements=obs_data.get("num_movable_elements", 0),
            step_number=obs_data.get("step_number", 0),
            all_clashes_resolved=obs_data.get("all_clashes_resolved", False),
            task=obs_data.get("task", "medium"),
            initial_clash_volume=obs_data.get("initial_clash_volume", 0.0),
            grade=obs_data.get("grade", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            message=obs_data.get("message", ""),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
