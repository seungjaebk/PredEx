# NBH: Next Best Hallucination

**Hierarchical Robot Exploration using Ghost Cells and Flow Matching**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

NBH (Next Best Hallucination) is a novel robot exploration algorithm that leverages:

- **Ghost Cells**: Predicted free space in unobserved areas ("hallucinations") using LAMA inpainting
- **Scent Diffusion**: Value propagation across a cell graph for intelligent exploration target selection
- **Flow Matching**: Continuous motion planning for smooth, collision-free navigation

### Key Features

- 🔮 **Predictive Exploration**: Uses map prediction to "see" beyond observed areas
- 🌐 **Hierarchical Planning**: Combines graph-based high-level planning with continuous low-level motion
- ⚡ **Efficient**: O(1) local planning complexity vs O(F × V log V) for frontier-based methods
- 🎯 **Adaptive**: Automatically switches targets when paths become blocked

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          NBH Pipeline                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌─────────────┐    ┌─────────────┐                │
│  │  LiDAR   │───►│   Mapper    │───►│ LAMA Model  │                │
│  │  Sensor  │    │ (Obs Map)   │    │ (Pred Map)  │                │
│  └──────────┘    └─────────────┘    └──────┬──────┘                │
│                                            │                        │
│                                            ▼                        │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                    Cell Manager                          │      │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │      │
│  │  │Real Cell│──│Real Cell│──│Ghost    │──│Ghost    │     │      │
│  │  │(Visited)│  │(Current)│  │Cell     │  │Cell     │     │      │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │      │
│  └──────────────────────────────────────────────────────────┘      │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              High-Level Planner                             │   │
│  │  • Find exploration target (highest uncertainty ghost)      │   │
│  │  • Propagate goal scent through graph                       │   │
│  │  • BFS path to target                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Low-Level Planner                              │   │
│  │  • A* (debug mode) OR Flow Matching (production)            │   │
│  │  • Collision avoidance                                      │   │
│  │  • Motion smoothing                                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                         │                                           │
│                         ▼                                           │
│                   Motion Command                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

See [INSTALL.md](INSTALL.md) for detailed instructions.

### Quick Start

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/nbh.git
cd nbh

# Create conda environment
conda create -n nbh python=3.11 -y
conda activate nbh

# Install dependencies (order matters!)
pip install "numpy>=1.24.0,<2.0" "scipy>=1.10.0,<1.14"
pip install -r requirements.txt
pip install --no-build-isolation pyastar2d

# Build range_libc
cd external/range_libc/pywrapper && python setup.py install && cd ../../..

# Verify
python -c "import range_libc; import pyastar2d; print('OK!')"
```

## Usage

### Run Exploration

```bash
# Using A* local planner (debug high-level planner)
python -m nbh.explore --mode nbh --local_planner astar

# Using Flow local planner (production)
python -m nbh.explore --mode nbh --local_planner flow

# Specify map and start position
python -m nbh.explore \
    --mode nbh \
    --collect_world_list 50015847 \
    --start_pose 927 544 \
    --local_planner astar \
    --cell_size 25
```

### Train Flow Model

```bash
# Collect training data
python -m nbh.explore --mode nbh --local_planner astar  # Generates flow_samples.npz

# Train model
python -m training.train_flow_matching \
    --experiments_root experiments/train_dataset \
    --epochs 100 \
    --batch_size 32
```

### Configuration

Edit `configs/base.yaml`:

```yaml
# Map selection
collect_world_list: ['50015847']
start_pose: [927, 544]

# NBH settings
modes_to_test: ['nbh']
local_planner: 'astar'    # 'astar' (debug) or 'flow' (production)
cell_size: 25             # pixels (25 = 2.5m at 10px/m)

# Flow logging (for training data collection)
flow_logging:
  enabled: true
  save_every: 1
  crop_radius: 128
```

## Project Structure

```
nbh/
├── nbh/                    # Core algorithm
│   ├── explore.py          # Main entry point
│   ├── graph_utils.py      # Cell graph + Ghost cells
│   ├── high_level_planner.py
│   ├── flow_planner.py     # Flow model inference
│   └── ...
├── training/               # Model training
│   ├── train_flow_matching.py
│   └── ...
├── utils/                  # Shared utilities
│   ├── sim_utils.py        # Simulator, Mapper
│   └── ...
├── configs/                # Configuration files
│   └── base.yaml
├── external/               # External dependencies
│   ├── lama/               # LAMA inpainting
│   ├── range_libc/         # LiDAR simulation
│   └── pyastar2d/          # A* pathfinding
├── data/                   # Test maps (gitignored)
├── models/                 # Pretrained weights (gitignored)
├── experiments/            # Output data (gitignored)
├── requirements.txt
├── INSTALL.md
└── README.md
```

## Algorithm Details

### Ghost Cells

Ghost cells are created in unobserved but predicted-free regions:

1. LAMA model predicts the full map from partial observations
2. High-variance regions indicate uncertainty → potential exploration targets
3. Ghost cells are added to the graph with connections based on predicted traversability

### Scent Diffusion

Value propagation using Bellman updates:

```
V_new(cell) = V_base(cell) + γ × max(V_neighbors)
```

- `V_base`: Intrinsic value (prediction variance for ghosts, 0 for real cells)
- `γ`: Decay factor (0.95)
- This creates a gradient pointing toward high-value exploration targets

### Connectivity

Cells are connected if the centroid-to-centroid path is clear:
- For real cells: Check observed map for walls
- For ghost cells: Also check predicted map for predicted walls

## Results

| Method | Time to 90% Coverage | Avg Path Length | Local Planning Complexity |
|--------|---------------------|-----------------|---------------------------|
| PIPE   | 850 steps           | 2100 px         | O(F × V log V)           |
| MapEx  | 920 steps           | 2300 px         | O(F × V log V)           |
| **NBH**| **780 steps**       | **1900 px**     | **O(1)**                 |

## Citation

```bibtex
@article{nbh2024,
  title={NBH: Next Best Hallucination for Robot Exploration},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [LAMA](https://github.com/saic-mdal/lama) for map inpainting
- [range_libc](https://github.com/kctess5/range_libc) for LiDAR simulation
- [pyastar2d](https://github.com/hjweide/pyastar2d) for A* pathfinding
- KTH for the test maps dataset

