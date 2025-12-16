import argparse
import json
import random
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
from torchcfm.models.models import MLP
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torch import amp

import os
os.environ['WANDB_API_KEY'] = '6b0a589988aa7615aeec04f1e215b1f9e1b84405'


# --------------------------------------------------------------------------------------
# Dataset (Iterable Optimized)
# --------------------------------------------------------------------------------------

# Normalization constant for delta (movement vector)
DELTA_SCALE = 10.0

def center_crop(arr: np.ndarray, target_size: int = 128) -> np.ndarray:
    """Center crop a 2D array to target_size x target_size."""
    h, w = arr.shape
    if h <= target_size and w <= target_size:
        return arr
    start_h = (h - target_size) // 2
    start_w = (w - target_size) // 2
    return arr[start_h:start_h + target_size, start_w:start_w + target_size]


def process_single_sample(sample: Dict, mode: str, map_id: str) -> Dict[str, torch.Tensor]:
    """Converts a raw dictionary sample to a torch tensor dict with AUGMENTATION."""
    obs = sample["obs_patch"].astype(np.float32)
    pred_mean = sample["pred_mean_patch"]
    pred_var = sample["pred_var_patch"]
    goal_vector = sample.get("goal_vector", np.zeros(2, dtype=np.float32))

    # --- Path-aware goal override ---
    # If path_prefix is available, use the first step direction as the goal hint.
    path_prefix = sample.get("path_prefix", None)
    if path_prefix is not None:
        try:
            # path_prefix is expected shape (k, 2) with (row, col) int positions
            path_prefix = np.asarray(path_prefix)
            if path_prefix.ndim == 2 and path_prefix.shape[0] >= 2:
                step_vec = path_prefix[1] - path_prefix[0]  # (row, col) delta
                norm = np.linalg.norm(step_vec)
                if norm > 1e-6:
                    goal_vector = (step_vec / norm).astype(np.float32)
        except Exception:
            # Fallback silently to provided goal_vector
            pass
    
    # Center crop patches from 257x257 to 128x128 (if needed)
    obs = center_crop(obs, 128)
    if pred_mean is not None:
        pred_mean = center_crop(pred_mean, 128)
    if pred_var is not None:
        pred_var = center_crop(pred_var, 128)
    
    # If goal_vector is None (e.g. no frontier), use zero
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

    # mask: 1 if prediction channels available, else 0
    mask_value = 1.0 if sample["pred_mean_patch"] is not None else 0.0
    mask = np.full_like(obs, mask_value, dtype=np.float32)

    delta = sample["delta"].astype(np.float32)

    # --- Data Augmentation: Rotation ---
    # Random rotation in [0, 2*pi]
    angle = np.random.uniform(0, 2 * np.pi)
    
    # 1. Rotate Maps (C, H, W) using kornia or simple scipy/numpy?
    # Since we are in a worker process, we can use scipy.ndimage.rotate
    # But it's slow. Let's use kornia on GPU? No, data loading is CPU.
    # Simple 90 degree rotations are fast, but continuous rotation is better.
    # For simplicity and speed, let's do 90 degree increments for now (0, 90, 180, 270)
    # This preserves pixel grid alignment which is nice for ConvNets.
    k = np.random.randint(0, 4)
    angle_discrete = k * (np.pi / 2)
    
    obs = np.rot90(obs, k)
    pred_mean = np.rot90(pred_mean, k)
    pred_var = np.rot90(pred_var, k)
    mask = np.rot90(mask, k)
    
    # 2. Rotate Vectors (delta, goal_vector)
    # If map rotates 90 deg CCW (k=1), the vector (relative to map) also rotates 90 deg CCW?
    # Example: Robot faces North. Wall is North. Map shows wall Up.
    # Rotate Map 90 deg CCW. Wall is now Left.
    # If robot was moving North (Up), relative to the NEW map, is it moving Left?
    # Yes. We rotate the World frame vector into the Augmented Frame.
    # So we apply the SAME rotation matrix to the vector.
    
    c, s = np.cos(angle_discrete), np.sin(angle_discrete)
    R = np.array([[c, -s], [s, c]]) # Standard rotation matrix (CCW)
    # Note: numpy array indexing is (row, col) -> (y, x).
    # Vector is usually (x, y) or (row, col)?
    # In our code: delta is (d_row, d_col).
    # Image rot90(k=1) moves Top (Row=0) to Left (Col=0). (r,c) -> (-c, r) roughly.
    # Let's verify k=1 (90 deg CCW):
    # (1, 0) [Down] -> (0, 1) [Right]? No.
    # (1, 0) [row=1] -> (0, 1) [col=1]?
    # Let's stick to standard 2D rotation.
    # If we treat (row, col) as (y, x) in image coordinates...
    # But standard matrix R rotates (x, y).
    # Let's just implement discrete logic to be safe.
    
    def rot_vec_90k(v, k):
        # v is (row, col)
        for _ in range(k):
            # 90 deg CCW: (r, c) -> (-c, r)
            v = np.array([-v[1], v[0]])
        return v

    delta = rot_vec_90k(delta, k)
    goal_vector = rot_vec_90k(goal_vector, k)

    # Normalize to [0,1]
    obs = torch.from_numpy(obs.copy()).unsqueeze(0) / 255.0
    mean = torch.from_numpy(pred_mean.copy()).unsqueeze(0) / 255.0
    var = torch.from_numpy(pred_var.copy()).unsqueeze(0)
    mask = torch.from_numpy(mask.copy()).unsqueeze(0)
    
    # Normalize delta
    delta = torch.from_numpy(delta) / DELTA_SCALE
    goal_vector = torch.from_numpy(goal_vector)

    return {
        "maps": torch.cat([obs, mean, var, mask], dim=0),
        "goal_vector": goal_vector,
        "delta": delta,
        "mode": mode,
        "map_id": map_id,
    }


class FlowIterableDataset(IterableDataset):
    def __init__(
        self,
        file_paths: List[Path],
        modes: Optional[List[str]] = None,
    ) -> None:
        self.file_paths = file_paths
        self.modes = modes

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        
        # Split files among workers
        if worker_info is None:  # Single-process
            my_files = self.file_paths
        else:
            # worker_0 gets 0, 4, 8... worker_1 gets 1, 5, 9...
            my_files = [
                f for i, f in enumerate(self.file_paths)
                if i % worker_info.num_workers == worker_info.id
            ]
        
        # Shuffle file order for this epoch
        random.shuffle(my_files)

        for npz_path in my_files:
            try:
                # Load the entire file into memory
                with np.load(npz_path, allow_pickle=True) as data:
                    meta = data["meta"].item()
                    file_mode = meta.get("mode", "")
                    
                    # Filter by mode at file level if possible
                    if self.modes and file_mode not in self.modes:
                        continue

                    samples = data["samples"]
                    map_id = Path(meta.get("map_folder_path", meta.get("map", ""))).stem
                    
                    # Shuffle samples within the file to break temporal correlation
                    # We use a list of indices to avoid copying the big array
                    indices = list(range(len(samples)))
                    random.shuffle(indices)

                    for idx in indices:
                        raw_sample = samples[idx]
                        
                        # Double check mode if mixed in file (rare but possible)
                        sample_mode = raw_sample.get("mode", file_mode)
                        if self.modes and sample_mode not in self.modes:
                            continue

                        yield process_single_sample(raw_sample, sample_mode, map_id)

            except Exception as e:
                print(f"Skipping {npz_path}: {e}")


# --------------------------------------------------------------------------------------
# Model skeleton - IMPROVED with Spatial Attention
# --------------------------------------------------------------------------------------

import math

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for time."""
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
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        device = x.device
        
        # Create coordinate grids normalized to [-1, 1]
        y_coords = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        
        # Concatenate coordinates as additional channels
        return torch.cat([x, y_coords, x_coords], dim=1)


class SpatialMapEncoder(nn.Module):
    """
    IMPROVED MapEncoder that preserves spatial information.
    - Uses coordinate encoding
    - Outputs spatial feature map (not just a vector)
    """
    def __init__(self, in_channels: int = 4, feature_dim: int = 64) -> None:
        super().__init__()
        self.coord_enc = CoordinateEncoding()
        
        # Input: (B, 4+2, 128, 128) with coordinate channels
        self.backbone = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(in_channels + 2, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            
            # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            
            # 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            
            # 16 -> 8
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # Output: (B, feature_dim, 8, 8) - SPATIAL features preserved!
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, 128, 128)
        x = self.coord_enc(x)  # (B, 6, 128, 128)
        return self.backbone(x)  # (B, feature_dim, 8, 8)


class GoalConditionedAttention(nn.Module):
    """
    Goal-conditioned spatial attention.
    Uses goal direction to query relevant spatial features.
    "Which parts of the map are relevant given where I want to go?"
    """
    def __init__(self, spatial_dim: int = 64, goal_dim: int = 2, num_heads: int = 4):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.num_heads = num_heads
        self.head_dim = spatial_dim // num_heads
        
        # Project goal to query
        self.goal_to_query = nn.Sequential(
            nn.Linear(goal_dim, spatial_dim),
            nn.LayerNorm(spatial_dim),
            nn.SiLU(),
            nn.Linear(spatial_dim, spatial_dim),
        )
        
        # Project spatial features to key/value
        self.to_kv = nn.Conv2d(spatial_dim, spatial_dim * 2, kernel_size=1)
        
        # Output projection
        self.out_proj = nn.Linear(spatial_dim, spatial_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, spatial_features: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # spatial_features: (B, C, H, W)
        # goal: (B, 2)
        B, C, H, W = spatial_features.shape
        
        # Goal -> Query: (B, C)
        query = self.goal_to_query(goal)  # (B, C)
        query = query.view(B, self.num_heads, 1, self.head_dim)  # (B, heads, 1, head_dim)
        
        # Spatial -> Key, Value: (B, C, H*W)
        kv = self.to_kv(spatial_features)  # (B, 2C, H, W)
        kv = kv.view(B, 2, self.num_heads, self.head_dim, H * W)  # (B, 2, heads, head_dim, HW)
        key, value = kv[:, 0], kv[:, 1]  # Each: (B, heads, head_dim, HW)

        # Reshape for scaled_dot_product_attention (uses FlashAttention when available)
        # query: (B, heads, 1, head_dim)
        # key/value: (B, heads, HW, head_dim)
        key = key.permute(0, 1, 3, 2)    # (B, heads, HW, head_dim)
        value = value.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)

        attn_out = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )  # (B, heads, 1, head_dim)
        out = attn_out.squeeze(2).reshape(B, C)  # (B, C)
        
        # Also get global average for context
        global_ctx = spatial_features.mean(dim=[2, 3])  # (B, C)
        
        # Combine attended features with global context
        combined = out + global_ctx
        return self.out_proj(combined)  # (B, C)


class ImprovedConditionalMLP(nn.Module):
    """
    IMPROVED Conditional MLP with larger capacity.
    """
    def __init__(self, dim: int, context_dim: int, hidden_dim: int = 512, time_emb_dim: int = 64) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        
        # input_dim = dim (z_t) + context_dim (attended map) + time_emb_dim + 2 (goal_vector)
        input_dim = dim + context_dim + time_emb_dim + 2
        
        # Deeper MLP for better reasoning
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, dim),
        )

    def forward(self, zt: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep)
        x = torch.cat([zt, context, t_emb, goal], dim=-1)
        return self.net(x)


class FlowMatchingModel(nn.Module):
    """
    IMPROVED Flow Matching Model with:
    1. Spatial feature preservation (coordinate encoding)
    2. Goal-conditioned attention (focus on relevant obstacles)
    3. Larger context dimension (256 instead of 128)
    4. Deeper MLP for better reasoning
    """
    def __init__(self) -> None:
        super().__init__()
        spatial_dim = 128  # Feature dimension in spatial maps
        context_dim = 256  # Increased from 128 for more capacity
        
        self.spatial_encoder = SpatialMapEncoder(in_channels=4, feature_dim=spatial_dim)
        self.goal_attention = GoalConditionedAttention(spatial_dim=spatial_dim, goal_dim=2, num_heads=4)
        
        # Project attended features to context_dim
        self.context_proj = nn.Sequential(
            nn.Linear(spatial_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.SiLU(),
        )
        
        self.flow = ImprovedConditionalMLP(
            dim=2,
            context_dim=context_dim,
            hidden_dim=512,  # Increased from 256
            time_emb_dim=64,  # Increased from 32
        )

    def forward(self, maps: torch.Tensor, zt: torch.Tensor, timestep: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # 1. Extract spatial features (preserves WHERE things are)
        spatial_features = self.spatial_encoder(maps)  # (B, 128, 8, 8)
        
        # 2. Goal-conditioned attention (focus on obstacles along goal direction)
        attended = self.goal_attention(spatial_features, goal)  # (B, 128)
        
        # 3. Project to context
        context = self.context_proj(attended)  # (B, 256)
        
        # 4. Predict velocity
        return self.flow(zt, timestep, context, goal)


# --------------------------------------------------------------------------------------
# LEGACY MODEL (for loading old checkpoints)
# --------------------------------------------------------------------------------------

class MapEncoder(nn.Module):
    """Legacy encoder for backward compatibility."""
    def __init__(self, in_channels: int = 4, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 17 * 17, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalMLP(nn.Module):
    """Legacy MLP for backward compatibility."""
    def __init__(self, dim: int, context_dim: int, hidden_dim: int = 128, time_emb_dim: int = 32) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        input_dim = dim + context_dim + time_emb_dim + 2
        self.net = MLP(dim=input_dim, out_dim=dim, w=hidden_dim, time_varying=False)

    def forward(self, zt: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep)
        x = torch.cat([zt, context, t_emb, goal], dim=-1)
        return self.net(x)


class FlowMatchingModelLegacy(nn.Module):
    """Legacy model for loading old checkpoints."""
    def __init__(self) -> None:
        super().__init__()
        context_dim = 128
        self.encoder = MapEncoder(in_channels=4, out_dim=context_dim)
        self.flow = ConditionalMLP(
            dim=2,
            context_dim=context_dim,
            hidden_dim=256,
            time_emb_dim=32, 
        )

    def forward(self, maps: torch.Tensor, zt: torch.Tensor, timestep: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        context = self.encoder(maps)
        return self.flow(zt, timestep, context, goal)


# --------------------------------------------------------------------------------------
# Training loop skeleton
# --------------------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    flow_matcher: ConditionalFlowMatcher,
    steps_per_epoch: int,
    scaler: amp.GradScaler,
) -> float:
    model.train()
    epoch_loss = 0.0
    count = 0

    # IterableDataset doesn't have len(), so we just iterate
    # Tqdm needs a total to show a bar, we use steps_per_epoch as an estimate
    pbar = tqdm(loader, total=steps_per_epoch, desc="train", leave=False)
    
    for i, batch in enumerate(pbar):
        maps = batch["maps"].to(device)
        target_delta = batch["delta"].to(device)  # z1
        goal_vector = batch["goal_vector"].to(device)

        z0 = torch.randn_like(target_delta)
        t, zt, ut = flow_matcher.sample_location_and_conditional_flow(z0, target_delta)
        t = t.to(device).unsqueeze(-1)
        zt = zt.to(device)
        ut = ut.to(device)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(
            device_type=device.type if device.type in ("cuda", "cpu") else "cuda",
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        ):
            pred_velocity = model(maps, zt, t, goal_vector)
            loss = nn.functional.mse_loss(pred_velocity, ut)

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
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, total=steps_per_epoch, desc="val", leave=False):
            maps = batch["maps"].to(device)
            target_delta = batch["delta"].to(device)
            goal_vector = batch["goal_vector"].to(device)

            z0 = torch.randn_like(target_delta)
            t, zt, ut = flow_matcher.sample_location_and_conditional_flow(z0, target_delta)
            t = t.to(device).unsqueeze(-1)
            zt = zt.to(device)
            ut = ut.to(device)

            with amp.autocast(
                device_type=device.type if device.type in ("cuda", "cpu") else "cuda",
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            ):
                pred_velocity = model(maps, zt, t, goal_vector)
                loss = nn.functional.mse_loss(pred_velocity, ut)
            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline flow-matching training skeleton")
    parser.add_argument(
        "--experiments_root",
        type=Path,
        default=Path("experiments"),
        help="Root directory containing *_test experiment folders",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="*",
        default=["pipe", "mapex", "nbv-2d"],
        help="Filter training data by mode",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of npz files for quick tests")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of FILES to use for validation")
    parser.add_argument("--wandb_project", type=str, default="pipe-flow-matching", help="WandB project name")
    parser.add_argument(
        "--start_date", 
        type=str, 
        default=None, 
        help="Filter files by start date (YYYYMMDD). e.g. '20251128' to skip older runs."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Init WandB
    wandb.init(project=args.wandb_project, config=vars(args))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, name: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")
    
    # 1. Find all files
    print("Scanning files...")
    all_files = sorted(args.experiments_root.rglob("*_flow_samples.npz"))
    
    # Optional: Filter by Date
    if args.start_date:
        print(f"Filtering files older than {args.start_date}...")
        filtered_files = []
        for p in all_files:
            # Path structure: experiments/YYYYMMDD_HHMMSS_Map_.../...npz
            # We look for the date string in the path parts
            # Usually the parent folder starts with YYYYMMDD
            # Let's check specific parent folder name
            # p.parent.parent.name should be "YYYYMMDD_HHMMSS_..."
            folder_name = p.parent.parent.name
            # Check if folder starts with a date that is >= start_date
            # Simple string comparison works for YYYYMMDD format
            if folder_name[:8].isdigit() and folder_name[:8] >= args.start_date:
                filtered_files.append(p)
        
        print(f"Filtered out {len(all_files) - len(filtered_files)} old files.")
        all_files = filtered_files

    if args.max_files is not None:
        all_files = all_files[:args.max_files]
    
    if not all_files:
        raise RuntimeError(f"No samples found under {args.experiments_root}")
        
    # 2. Shuffle and Split files (Train/Val)
    # We split by FILE (Episode), not sample. This prevents data leakage.
    random.shuffle(all_files)
    val_count = int(len(all_files) * args.val_split)
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]
    
    print(f"Found {len(all_files)} files.")
    print(f"Train files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")
    
    # 3. Create datasets
    train_dataset = FlowIterableDataset(train_files, modes=args.modes)
    val_dataset = FlowIterableDataset(val_files, modes=args.modes)
    
    # 4. DataLoaders
    # num_workers > 0 ensures we process multiple files in parallel (better mixing)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=2, 
        pin_memory=True,
        drop_last=False
    )

    # Estimate steps per epoch for tqdm (approximate)
    # Assume avg 1000 samples per file
    avg_samples_per_file = 1000
    train_steps = (len(train_files) * avg_samples_per_file) // args.batch_size
    val_steps = (len(val_files) * avg_samples_per_file) // args.batch_size

    model = FlowMatchingModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    flow_matcher = ConditionalFlowMatcher()
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, flow_matcher, train_steps, scaler
        )
        
        if len(val_files) > 0:
            val_loss = evaluate(model, val_loader, device, flow_matcher, val_steps)
        else:
            val_loss = 0.0
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

        # Save regular checkpoint
        ckpt_path = args.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        state_dict = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": vars(args),
        }
        torch.save(state_dict, ckpt_path)
        
        # Save best checkpoint
        if val_loss < best_val_loss and len(val_files) > 0:
            best_val_loss = val_loss
            best_path = args.checkpoint_dir / "best.pt"
            torch.save(state_dict, best_path)
            print(f"New best model saved to {best_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
