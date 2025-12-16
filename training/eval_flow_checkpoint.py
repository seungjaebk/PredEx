# scripts/eval_flow_checkpoint.py
import torch
import numpy as np
from pathlib import Path

from train_flow_matching import FlowMatchingModel, FlowSampleDataset
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

ckpt_path = Path("/home/seungjab/pipe-planner/checkpoints/epoch_020.pt")
sample_npz = Path("/home/seungjab/pipe-planner/experiments/20251118_test/20251118_100233_50015848_900_547_pipe/flow_samples/20251118_100233_50015848_900_547_pipe_flow_samples.npz")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Build dataset + grab a single sample
dataset = FlowSampleDataset(sample_npz.parent.parent.parent, modes=["pipe", "mapex"])
sample = dataset[0]  # choose whichever index you want
maps = sample["maps"].unsqueeze(0).to(device)  # add batch dim

# 2) Restore model weights
model = FlowMatchingModel().to(device)
state = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(state["model_state"])
model.eval()

# 3) Draw a noise sample and timestep, integrate one Euler step
flow_matcher = ConditionalFlowMatcher()
with torch.no_grad():
    z0 = torch.randn(1, 2, device=device)
    t = torch.rand(1, 1, device=device)
    zt = (1 - t) * z0  # start from noise (no target, since we’re sampling)
    pred_vel = model(maps, zt, t)
    delta = z0 + pred_vel  # first-order update toward expert delta

print("Predicted delta (row, col):", delta.squeeze().cpu().numpy())