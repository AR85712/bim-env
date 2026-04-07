"""Quick smoke test for BimClashEnv."""
import sys
sys.path.insert(0, "d:/HF-Hackathon")

from bim_env.gym_env import BimClashEnv, OBS_DIM

print(f"OBS_DIM = {OBS_DIM}")

for task in ["easy", "medium", "hard"]:
    env = BimClashEnv(task=task)
    obs, info = env.reset()
    print(f"[{task}] obs.shape={obs.shape}  range=[{float(obs.min()):.3f}, {float(obs.max()):.3f}]")
    assert obs.shape == (OBS_DIM,), f"Bad shape: {obs.shape}"

    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    n = info["num_clashes"]
    r = round(reward, 4)
    print(f"       reward={r}  terminated={terminated}  truncated={truncated}  clashes={n}")
    env.close()

print("\nSmoke test PASSED")
