```markdown
# Adaptive Garment Manufacturing

**A Multi‑Modal AI Pipeline for Trend Prediction and Digital Twin Simulation**

---

## Overview

This repository contains the complete end‑to‑end framework developed for **adaptive and sustainable fashion manufacturing**. The system detects fashion trends from social media, forecasts demand with deep learning, optimises factory operations via reinforcement learning, and integrates a simulated robotic cutting cell.

The project is split into two main parts:

- **AI‑driven pipeline** – data collection, multi‑modal trend detection, demand forecasting, and a factory digital twin (SimPy + RL inventory agent).
- **Robotic cutting cell** – a ROS 2 / Gazebo simulation with vision‑guided cutting and a learned cutting policy (SAC).

---

## Repository Structure

```
adaptive-garment-manufacturing/
├── pipeline/                          # Multi‑modal AI pipeline
│   ├── scraping/                      # Instagram, Pinterest, Google Trends collectors
│   ├── preprocessing/                 # YOLOS detection, NLP, data merging
│   ├── forecasting/                   # Temporal Fusion Transformer (TFT)
│   ├── simulation/                    # SimPy digital twin, PPO inventory agent
│   ├── requirements_pipeline.txt
│   ├── .env.example
│   └── README.md
│
├── robotics/                          # ROS 2 workspace for the robotic cell
│   ├── src/                           # Packages: description, bringup, control, interfaces
│   ├── requirements_ros.txt
│   ├── install_ros_jazzy.sh           # (optional) system dependency script
│   └── README.md
│
├── .gitignore
├── LICENSE
└── README.md                          # ← you are here
```

Detailed instructions for each component are in their respective READMEs:
- [`pipeline/README.md`](pipeline/README.md)
- [`robotics/README.md`](robotics/README.md)

---

## System Architecture (simplified)

```mermaid
graph TD
    A[Social Media\n(Instagram, Pinterest)] --> B[Multi‑Modal Trend Detection\nYOLOS (vision) + BART (text)]
    B --> C[TFT Demand Forecast\n(30 days)]
    C --> D[SimPy Factory Digital Twin]
    D --> E[RL Inventory Agent\n(PPO)]
    D --> F[Robotic Cutting Cell\n(ROS 2 / Gazebo)]
    F --> G[RL Cutting Agent\n(SAC)]
```

*If Mermaid is not supported in your viewer, the diagram shows a linear flow from social media data, through multi‑modal trend detection, demand forecasting, factory digital twin, and finally to the robotic cutting cell, with an RL inventory agent and an RL cutting agent as parallel optimisation components.*

---

## Global Prerequisites

- **Ubuntu 24.04** (or a compatible Linux environment)
- **Python 3.12**
- **ROS 2 Jazzy** (only for the robotics part)
- **Gazebo Harmonic** (only for the robotics part)

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Somayeh5656/adaptive-garment-manufacturing.git
cd adaptive-garment-manufacturing
```

### 2. Set up the AI pipeline (pure Python)

```bash
cd pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_pipeline.txt
cp .env.example .env   # edit .env with your secrets
```

Then follow the step‑by‑step guide in [`pipeline/README.md`](pipeline/README.md).

### 3. Set up the robotic cutting cell (ROS 2)

```bash
cd robotics
# install system dependencies (see robotics/README.md)
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements_ros.txt
colcon build --symlink-install
```

See [`robotics/README.md`](robotics/README.md) for launching the simulation.

---

## Integration

The pipeline’s digital twin (`pipeline/simulation/garment_env.py`) communicates with the ROS 2 cutting service. Make sure the robotics simulation is running before you evaluate the full system:

```bash
cd pipeline/simulation
python evaluate_rl.py
```
