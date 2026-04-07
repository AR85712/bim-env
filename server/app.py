"""
FastAPI application for the BIM Clash/Conflict Resolution Environment.

Endpoints:
    POST /reset   — Start a new episode
    POST /step    — Apply a BimAction (translate a MEP element)
    GET  /state   — Current episode metadata
    GET  /schema  — Action / observation JSON schemas
    WS   /ws      — WebSocket for persistent RL training sessions

Usage:
    # From the HF-Hackathon directory:
    uvicorn bim_env.server.app:app --host 0.0.0.0 --port 8000

    # Or via the entry-point script:
    uv run --project bim_env server
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import BimAction, BimObservation
    from .bim_env_environment import BimEnvironment
except ModuleNotFoundError:
    from models import BimAction, BimObservation
    from server.bim_env_environment import BimEnvironment

app = create_app(
    BimEnvironment,
    BimAction,
    BimObservation,
    env_name="bim_env",
    max_concurrent_envs=4,
)


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for ``uv run server`` and ``python -m bim_env.server.app``."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BIM Clash Resolution server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
