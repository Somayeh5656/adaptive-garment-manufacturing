# Adaptive Garment Manufacturing

**A Multi‑Modal AI Pipeline for Trend Prediction and Digital Twin Simulation**

M.Sc. Thesis – Tampere University, May 2026  
*Somayeh Mozaffari*

---

## Overview

This repository contains the complete end‑to‑end framework developed for **adaptive and sustainable fashion manufacturing**. The system detects fashion trends from social media, forecasts demand with deep learning, optimises factory operations via reinforcement learning, and integrates a simulated robotic cutting cell.

The project is split into two main parts:

- **AI‑driven pipeline** – data collection, multi‑modal trend detection, demand forecasting, and a factory digital twin (SimPy + RL inventory agent).
- **Robotic cutting cell** – a ROS 2 / Gazebo simulation with vision‑guided cutting and a learned cutting policy (SAC).

---

## Repository Structure
adaptive-garment-manufacturing/
├── pipeline/ # Multi‑modal AI pipeline
│ ├── scraping/ # Instagram, Pinterest, Google Trends collectors
│ ├── preprocessing/ # YOLOS detection, NLP, data merging
│ ├── forecasting/ # Temporal Fusion Transformer (TFT)
│ ├── simulation/ # SimPy digital twin, PPO inventory agent
│ ├── requirements_pipeline.txt
│ ├── .env.example
│ └── README.md
│
├── robotics/ # ROS 2 workspace for the robotic cell
│ ├── src/ # Packages: description, bringup, control, interfaces
│ ├── requirements_ros.txt
│ ├── install_ros_jazzy.sh # (optional) system dependency script
│ └── README.md
│
├── .gitignore
├── LICENSE
└── README.md # ← you are here

text

Detailed instructions for each component are in their respective READMEs:
- [`pipeline/README.md`](pipeline/README.md)
- [`robotics/README.md`](robotics/README.md)

---

## System Architecture (simplified)
Social Media (Instagram, Pinterest)
│
▼
Multi‑Modal Trend Detection ← YOLOS (vision) + BART (text)
│
▼
TFT Demand Forecast (30 days)
│
▼
SimPy Factory Digital Twin ──► RL Inventory Agent (PPO)
│
▼
Robotic Cutting Cell (ROS 2 / Gazebo) ← RL Cutting Agent (SAC)

text

---

## Global Prerequisites

- **Ubuntu 24.04** (or a compatible Linux environment)
- **Python 3.12**
- **ROS 2 Jazzy** (only for the robotics part)
- **Gazebo Harmonic** (only for the robotics part)

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/adaptive-garment-manufacturing.git
cd adaptive-garment-manufacturing
2. Set up the AI pipeline (pure Python)
bash
cd pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_pipeline.txt
cp .env.example .env   # edit .env with your secrets
Then follow the step‑by‑step guide in pipeline/README.md.

3. Set up the robotic cutting cell (ROS 2)
bash
cd robotics
# install system dependencies (see robotics/README.md)
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements_ros.txt
colcon build --symlink-install
See robotics/README.md for launching the simulation.

Integration
The pipeline’s digital twin (pipeline/simulation/garment_env.py) communicates with the ROS 2 cutting service. Make sure the robotics simulation is running before you evaluate the full system with:

bash
cd pipeline/simulation
python evaluate_rl.py
