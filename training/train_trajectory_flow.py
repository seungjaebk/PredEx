"""
Trajectory Flow Matching Training - Inspired by NoMAD/FlowNav

Outputs k=20 waypoint trajectories instead of single-step deltas.
Uses 1D temporal convolutions for sequence modeling.
"""
import argparse
import json
import random
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm
import wandb
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torch import amp

import os
os.environ['WANDB_API_KEY'] = '6b0a589988aa7615aeec04f1e215b1f9e1b84405'

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

DELTA_SCALE = 10.0  # Normalization scale for trajectory waypoints
TRAJECTORY_LEN = 20  # Number of future waypoints to predict


# --------------------------------------------------------------------------------------
# Dataset with Trajectory Output
# --------------------------------------------------------------------------------------

def center_crop(arr: np.ndarray, target_size: int = 128) -> np.ndarray:
    """Center crop a 2D array to target_size x target_size."""
    h, w = arr.shape
    if h <= target_size and w <= target_size:
        return arr
    start_h = (h - target_size) // 2
    start_w = (w - target_size) // 2
    return arr[start_h:start_h + target_size, start_w:start_w + target_size]


def process_trajectory_sample(sample: Dict, mode: str, map_id: str, 
                               trajectory_len: int = TRAJECTORY_LEN) -> Optional[Dict[str, torch.Tensor]]:
    """
    Process sample to output trajectory instead of single delta.
    
    Returns None if path is too short for trajectory.
    """
    obs = sample["obs_patch"].astype(np.float32)
    pred_mean = sample["pred_mean_patch"]
    pred_var = sample["pred_var_patch"]
    goal_vector = sample.get("goal_vector", np.zeros(2, dtype=np.float32))
    
    # --- Extract trajectory from path_prefix ---
    path_prefix = sample.get("path_prefix", None)
    
    if path_prefix is None:
        return None  # Skip samples without path
    
    path_prefix = np.asarray(path_prefix, dtype=np.float32)
    
    if path_prefix.ndim != 2 or path_prefix.shape[0] < 2:
        return None  # Invalid path
    
    # Create trajectory: relative positions from current pose
    # path_prefix[0] is current pose, path_prefix[1:] are future positions
    current_pose = path_prefix[0]
    
    # Get available future waypoints
    future_waypoints = path_prefix[1:] - current_pose  # Relative to current
    
    # Pad or truncate to trajectory_len
    if len(future_waypoints) >= trajectory_len:
        trajectory = future_waypoints[:trajectory_len]
    else:
        # Pad by repeating last waypoint (reached destination)
        last_wp = future_waypoints[-1] if len(future_waypoints) > 0 else np.zeros(2)
        padding = np.tile(last_wp, (trajectory_len - len(future_waypoints), 1))
        trajectory = np.vstack([future_waypoints, padding]) if len(future_waypoints) > 0 else padding
    
    trajectory = trajectory.astype(np.float32)  # (trajectory_len, 2)
    
    # Use first step direction as goal hint
    if len(future_waypoints) >= 1:
        step_vec = future_waypoints[0]
        norm = np.linalg.norm(step_vec)
        if norm > 1e-6:
            goal_vector = (step_vec / norm).astype(np.float32)
    
    # Center crop patches
    obs = center_crop(obs, 128)
    if pred_mean is not None:
        pred_mean = center_crop(pred_mean, 128)
    if pred_var is not None:
        pred_var = center_crop(pred_var, 128)
    
    # Handle None values
    if goal_vector is None:
        goal_vector = np.zeros(2, dtype=np.float32)
    else:
        goal_vector = goal_vector.astype(np.float32)

    if pred_mean is None:
        pred_mean = np.zeros_like(obs, dtype=np.float32)
    else:
        pred_mean = pred_mean.astype(np.float32)

    if pred_var is None:
        pred_var = np.zeros_like(obs, dtype=np.float32)
    else:
        pred_var = pred_var.astype(np.float32)

    mask_value = 1.0 if sample["pred_mean_patch"] is not None else 0.0
    mask = np.full_like(obs, mask_value, dtype=np.float32)

    # --- Data Augmentation: 90° Rotation ---
    k = np.random.randint(0, 4)
    
    obs = np.rot90(obs, k).copy()
    pred_mean = np.rot90(pred_mean, k).copy()
    pred_var = np.rot90(pred_var, k).copy()
    mask = np.rot90(mask, k).copy()
    
    def rot_vec_90k(v, k):
        """Rotate vector by k*90 degrees CCW."""
        for _ in range(k):
            v = np.array([-v[1], v[0]])
        return v
    
    # Rotate trajectory (each waypoint)
    trajectory = np.array([rot_vec_90k(wp, k) for wp in trajectory])
    goal_vector = rot_vec_90k(goal_vector, k)

    # Convert to tensors
    obs = torch.from_numpy(obs).unsqueeze(0) / 255.0
    mean = torch.from_numpy(pred_mean).unsqueeze(0) / 255.0
    var = torch.from_numpy(pred_var).unsqueeze(0)
    mask = torch.from_numpy(mask).unsqueeze(0)
    
    trajectory = torch.from_numpy(trajectory) / DELTA_SCALE  # (trajectory_len, 2)
    goal_vector = torch.from_numpy(goal_vector)

    return {
        "maps": torch.cat([obs, mean, var, mask], dim=0),
        "goal_vector": goal_vector,
        "trajectory": trajectory,  # NEW: (trajectory_len, 2) instead of "delta": (2,)
        "mode": mode,
        "map_id": map_id,
    }


class TrajectoryFlowDataset(IterableDataset):
    """Dataset that outputs trajectory sequences for training."""
    
    def __init__(
        self,
        file_paths: List[Path],
        modes: Optional[List[str]] = None,
        trajectory_len: int = TRAJECTORY_LEN,
        min_path_len: int = 5,  # Minimum path length to use sample
    ) -> None:
        self.file_paths = file_paths
        self.modes = modes
        self.trajectory_len = trajectory_len
        self.min_path_len = min_path_len

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        
        if worker_info is None:
            my_files = self.file_paths
        else:
            my_files = [
                f for i, f in enumerate(self.file_paths)
                if i % worker_info.num_workers == worker_info.id
            ]
        
        random.shuffle(my_files)

        for npz_path in my_files:
            try:
                with np.load(npz_path, allow_pickle=True) as data:
                    meta = data["meta"].item()
                    file_mode = meta.get("mode", "")
                    
                    if self.modes and file_mode not in self.modes:
                        continue

                    samples = data["samples"]
                    map_id = Path(meta.get("map_folder_path", meta.get("map", ""))).stem
                    
                    indices = list(range(len(samples)))
                    random.shuffle(indices)

                    for idx in indices:
                        raw_sample = samples[idx]
                        
                        # Check path length before processing
                        path = raw_sample.get("path_prefix", None)
                        if path is None or len(path) < self.min_path_len:
                            continue
                        
                        sample_mode = raw_sample.get("mode", file_mode)
                        if self.modes and sample_mode not in self.modes:
                            continue

                        result = process_trajectory_sample(
                            raw_sample, sample_mode, map_id, self.trajectory_len
                        )
                        if result is not None:
                            yield result

            except Exception as e:
                print(f"Skipping {npz_path}: {e}")


# --------------------------------------------------------------------------------------
# Model Architecture
# --------------------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x: torch.Tensor):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        emb = x * emb.unsqueeze(0) * self.scale
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb


class CoordinateEncoding(nn.Module):
    """Add 2D coordinate channels to feature maps."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device
        y_coords = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, y_coords, x_coords], dim=1)


class SpatialMapEncoder(nn.Module):
    """CNN encoder that preserves spatial information."""
    def __init__(self, in_channels: int = 4, feature_dim: int = 128) -> None:
        super().__init__()
        self.coord_enc = CoordinateEncoding()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels + 2, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.coord_enc(x)
        return self.backbone(x)  # (B, feature_dim, 8, 8)


class GoalConditionedAttention(nn.Module):
    """Goal-conditioned spatial attention using multi-head attention."""
    def __init__(self, spatial_dim: int = 128, goal_dim: int = 2, num_heads: int = 4):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.num_heads = num_heads
        self.head_dim = spatial_dim // num_heads
        
        self.goal_to_query = nn.Sequential(
            nn.Linear(goal_dim, spatial_dim),
            nn.LayerNorm(spatial_dim),
            nn.SiLU(),
            nn.Linear(spatial_dim, spatial_dim),
        )
        
        self.to_kv = nn.Conv2d(spatial_dim, spatial_dim * 2, kernel_size=1)
        self.out_proj = nn.Linear(spatial_dim, spatial_dim)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, spatial_features: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        B, C, H, W = spatial_features.shape
        
        query = self.goal_to_query(goal)
        query = query.view(B, self.num_heads, 1, self.head_dim)
        
        kv = self.to_kv(spatial_features)
        kv = kv.view(B, 2, self.num_heads, self.head_dim, H * W)
        key, value = kv[:, 0], kv[:, 1]
        
        key = key.permute(0, 1, 3, 2)
        value = value.permute(0, 1, 3, 2)

        attn_out = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        out = attn_out.squeeze(2).reshape(B, C)
        
        global_ctx = spatial_features.mean(dim=[2, 3])
        combined = out + global_ctx
        return self.out_proj(combined)


# --------------------------------------------------------------------------------------
# Temporal Convolution Block (Inspired by NoMAD's ConditionalUnet1D)
# --------------------------------------------------------------------------------------

class Conv1dBlock(nn.Module):
    """1D Convolution block with GroupNorm and activation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(8, out_channels),
            nn.Mish(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResBlock1d(nn.Module):
    """Residual block conditioned on global context."""
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size)
        
        # Condition projection (FiLM-style)
        self.cond_proj = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),  # scale and shift
        )
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        
        # Apply condition (FiLM)
        cond_out = self.cond_proj(cond)  # (B, out_channels * 2)
        scale, shift = cond_out.chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        
        h = self.conv2(h)
        return h + self.residual(x)


class TemporalUNet1D(nn.Module):
    """
    1D UNet for trajectory prediction.
    Simplified version of diffusion_policy's ConditionalUnet1D.
    
    Input: (B, 2, trajectory_len) noisy trajectory
    Condition: (B, cond_dim) global context
    Output: (B, 2, trajectory_len) velocity field
    """
    def __init__(
        self, 
        input_dim: int = 2, 
        cond_dim: int = 320,  # context + time embedding
        down_dims: List[int] = [128, 256, 512],
    ):
        super().__init__()
        self.input_dim = input_dim
        
        # Initial projection
        self.input_proj = nn.Conv1d(input_dim, down_dims[0], 1)
        
        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        in_ch = down_dims[0]
        for out_ch in down_dims:
            self.down_blocks.append(ConditionalResBlock1d(in_ch, out_ch, cond_dim))
            self.down_samples.append(nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1))
            in_ch = out_ch
        
        # Middle
        self.mid_block = ConditionalResBlock1d(down_dims[-1], down_dims[-1], cond_dim)
        
        # Decoder (upsampling)
        self.up_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        for i, out_ch in enumerate(reversed(down_dims[:-1])):
            in_ch = down_dims[-1 - i]
            self.up_samples.append(nn.ConvTranspose1d(in_ch, in_ch, 4, stride=2, padding=1))
            # Skip connection doubles channels
            self.up_blocks.append(ConditionalResBlock1d(in_ch + out_ch, out_ch, cond_dim))
        
        # Final upsampling back to input dims
        self.up_samples.append(nn.ConvTranspose1d(down_dims[0], down_dims[0], 4, stride=2, padding=1))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, down_dims[0]),
            nn.Mish(),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim, trajectory_len) noisy trajectory
            cond: (B, cond_dim) conditioning
        Returns:
            (B, input_dim, trajectory_len) predicted velocity
        """
        h = self.input_proj(x)
        
        # Encoder with skip connections
        skips = []
        for block, down in zip(self.down_blocks, self.down_samples):
            h = block(h, cond)
            skips.append(h)
            h = down(h)
        
        # Middle
        h = self.mid_block(h, cond)
        
        # Decoder with skip connections
        for i, (up, block) in enumerate(zip(self.up_samples[:-1], self.up_blocks)):
            h = up(h)
            skip = skips[-(i + 1)]
            # Handle size mismatch from downsampling
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode='linear', align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h, cond)
        
        # Final upsample
        h = self.up_samples[-1](h)
        if h.shape[-1] != x.shape[-1]:
            h = F.interpolate(h, size=x.shape[-1], mode='linear', align_corners=False)
        
        return self.output_proj(h)


class TrajectoryFlowMatchingModel(nn.Module):
    """
    Trajectory Flow Matching Model.
    
    Outputs k=20 waypoint trajectory instead of single delta.
    Uses spatial attention for map encoding + temporal UNet for trajectory.
    
    Architecture inspired by NoMAD (ICRA 2024) and FlowNav (IROS 2025).
    """
    def __init__(self, trajectory_len: int = TRAJECTORY_LEN) -> None:
        super().__init__()
        self.trajectory_len = trajectory_len
        
        spatial_dim = 128
        context_dim = 256
        time_dim = 64
        
        # Map Encoder (same as original)
        self.spatial_encoder = SpatialMapEncoder(in_channels=4, feature_dim=spatial_dim)
        self.goal_attention = GoalConditionedAttention(spatial_dim=spatial_dim, goal_dim=2, num_heads=4)
        
        self.context_proj = nn.Sequential(
            nn.Linear(spatial_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.SiLU(),
        )
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        # Trajectory decoder (instead of MLP)
        # cond_dim = context + time + goal
        cond_dim = context_dim + time_dim + 2
        self.trajectory_decoder = TemporalUNet1D(
            input_dim=2,
            cond_dim=cond_dim,
            down_dims=[128, 256, 512],
        )
    
    def forward(
        self, 
        maps: torch.Tensor,      # (B, 4, 128, 128)
        z_t: torch.Tensor,       # (B, trajectory_len, 2) noisy trajectory
        timestep: torch.Tensor,  # (B, 1)
        goal: torch.Tensor,      # (B, 2)
    ) -> torch.Tensor:
        """
        Predict velocity field for trajectory flow matching.
        
        Returns:
            (B, trajectory_len, 2) velocity field
        """
        B = maps.shape[0]
        
        # 1. Encode map with goal-conditioned attention
        spatial_features = self.spatial_encoder(maps)
        attended = self.goal_attention(spatial_features, goal)
        context = self.context_proj(attended)  # (B, context_dim)
        
        # 2. Time embedding
        t_emb = self.time_mlp(timestep.squeeze(-1))  # (B, time_dim)
        
        # 3. Combine conditioning
        global_cond = torch.cat([context, t_emb, goal], dim=-1)  # (B, cond_dim)
        
        # 4. Trajectory UNet expects (B, 2, trajectory_len)
        z_t_transposed = z_t.permute(0, 2, 1)  # (B, 2, trajectory_len)
        
        velocity = self.trajectory_decoder(z_t_transposed, global_cond)  # (B, 2, trajectory_len)
        
        return velocity.permute(0, 2, 1)  # (B, trajectory_len, 2)


# --------------------------------------------------------------------------------------
# Training Loop
# --------------------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    flow_matcher: ConditionalFlowMatcher,
    steps_per_epoch: int,
    scaler: amp.GradScaler,
    trajectory_len: int = TRAJECTORY_LEN,
) -> float:
    model.train()
    epoch_loss = 0.0
    count = 0

    pbar = tqdm(loader, total=steps_per_epoch, desc="train", leave=False)
    
    for batch in pbar:
        maps = batch["maps"].to(device)
        target_trajectory = batch["trajectory"].to(device)  # (B, trajectory_len, 2)
        goal_vector = batch["goal_vector"].to(device)
        
        B = maps.shape[0]

        # Flow matching on flattened trajectory
        z0 = torch.randn_like(target_trajectory)  # (B, trajectory_len, 2)
        
        # Flatten for flow matcher: (B, trajectory_len * 2)
        z0_flat = z0.view(B, -1)
        target_flat = target_trajectory.view(B, -1)
        
        t, zt_flat, ut_flat = flow_matcher.sample_location_and_conditional_flow(z0_flat, target_flat)
        
        # Reshape back
        zt = zt_flat.view(B, trajectory_len, 2).to(device)
        ut = ut_flat.view(B, trajectory_len, 2).to(device)
        t = t.to(device).unsqueeze(-1)

        optimizer.zero_grad(set_to_none=True)
        
        with amp.autocast(
            device_type=device.type if device.type in ("cuda", "cpu") else "cuda",
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        ):
            pred_velocity = model(maps, zt, t, goal_vector)
            loss = F.mse_loss(pred_velocity, ut)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        epoch_loss += loss_val
        count += 1
        
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return epoch_loss / max(count, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    flow_matcher: ConditionalFlowMatcher,
    steps_per_epoch: int,
    trajectory_len: int = TRAJECTORY_LEN,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, total=steps_per_epoch, desc="val", leave=False):
            maps = batch["maps"].to(device)
            target_trajectory = batch["trajectory"].to(device)
            goal_vector = batch["goal_vector"].to(device)
            
            B = maps.shape[0]

            z0 = torch.randn_like(target_trajectory)
            z0_flat = z0.view(B, -1)
            target_flat = target_trajectory.view(B, -1)
            
            t, zt_flat, ut_flat = flow_matcher.sample_location_and_conditional_flow(z0_flat, target_flat)
            
            zt = zt_flat.view(B, trajectory_len, 2).to(device)
            ut = ut_flat.view(B, trajectory_len, 2).to(device)
            t = t.to(device).unsqueeze(-1)

            with amp.autocast(
                device_type=device.type if device.type in ("cuda", "cpu") else "cuda",
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            ):
                pred_velocity = model(maps, zt, t, goal_vector)
                loss = F.mse_loss(pred_velocity, ut)
            
            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


def sample_trajectory(
    model: nn.Module,
    maps: torch.Tensor,
    goal: torch.Tensor,
    device: torch.device,
    num_steps: int = 10,
    trajectory_len: int = TRAJECTORY_LEN,
) -> np.ndarray:
    """
    Sample a trajectory using Euler integration.
    
    Returns:
        (trajectory_len, 2) numpy array of waypoints in pixel units
    """
    model.eval()
    
    with torch.no_grad():
        z = torch.randn((1, trajectory_len, 2), device=device)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((1, 1), i / num_steps, device=device)
            vel = model(maps, z, t, goal)
            z = z + vel * dt
        
        trajectory = z.squeeze(0).cpu().numpy() * DELTA_SCALE
    
    return trajectory


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trajectory Flow Matching Training")
    parser.add_argument("--experiments_root", type=Path, default=Path("experiments"))
    parser.add_argument("--modes", type=str, nargs="*", default=["pipe", "mapex", "nbv-2d"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)  # Lower LR for larger model
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--wandb_project", type=str, default="pipe-trajectory-flow")
    parser.add_argument("--trajectory_len", type=int, default=TRAJECTORY_LEN)
    parser.add_argument("--min_path_len", type=int, default=5)
    parser.add_argument("--start_date", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    wandb.init(project=args.wandb_project, config=vars(args))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Find files
    print("Scanning files...")
    all_files = sorted(args.experiments_root.rglob("*_flow_samples.npz"))
    
    if args.start_date:
        print(f"Filtering files older than {args.start_date}...")
        filtered = []
        for p in all_files:
            folder_name = p.parent.parent.name
            if folder_name[:8].isdigit() and folder_name[:8] >= args.start_date:
                filtered.append(p)
        all_files = filtered
    
    if args.max_files:
        all_files = all_files[:args.max_files]
    
    if not all_files:
        raise RuntimeError(f"No samples found under {args.experiments_root}")
    
    # Split files
    random.shuffle(all_files)
    val_count = int(len(all_files) * args.val_split)
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]
    
    print(f"Total files: {len(all_files)}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Create datasets
    train_dataset = TrajectoryFlowDataset(
        train_files, 
        modes=args.modes,
        trajectory_len=args.trajectory_len,
        min_path_len=args.min_path_len,
    )
    val_dataset = TrajectoryFlowDataset(
        val_files,
        modes=args.modes,
        trajectory_len=args.trajectory_len,
        min_path_len=args.min_path_len,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    
    # Estimate steps
    avg_samples_per_file = 500  # Conservative estimate with path length filter
    train_steps = (len(train_files) * avg_samples_per_file) // args.batch_size
    val_steps = (len(val_files) * avg_samples_per_file) // args.batch_size

    # Create model
    model = TrajectoryFlowMatchingModel(trajectory_len=args.trajectory_len).to(device)
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    flow_matcher = ConditionalFlowMatcher()
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, flow_matcher, 
            train_steps, scaler, args.trajectory_len
        )
        
        val_loss = 0.0
        if len(val_files) > 0:
            val_loss = evaluate(
                model, val_loader, device, flow_matcher, 
                val_steps, args.trajectory_len
            )
        
        scheduler.step()
        
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
        wandb.log({
            "train_loss": train_loss, 
            "val_loss": val_loss, 
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
        })

        # Save checkpoints
        state_dict = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": vars(args),
        }
        
        ckpt_path = args.checkpoint_dir / f"trajectory_epoch_{epoch:03d}.pt"
        torch.save(state_dict, ckpt_path)
        
        if val_loss < best_val_loss and len(val_files) > 0:
            best_val_loss = val_loss
            best_path = args.checkpoint_dir / "trajectory_best.pt"
            torch.save(state_dict, best_path)
            print(f"  -> New best model: {best_path}")

    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()
