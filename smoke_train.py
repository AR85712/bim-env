"""Short PPO training smoke test — 5000 steps, 2 envs."""
import sys, os
sys.path.insert(0, "d:/HF-Hackathon")
os.environ.setdefault("TASK", "easy")
os.environ["TOTAL_TIMESTEPS"] = "5000"
os.environ["N_ENVS"] = "2"
os.environ["EVAL_FREQ"] = "2500"
os.environ["SAVE_FREQ"] = "5000"

from bim_env.train import train
train()
