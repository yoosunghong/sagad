# Technology Stack Specification

This document establishes the software, libraries, and runtime frameworks designated for the project implementation.

## 1. Deep Learning & Core AI Frameworks
* **Language:** Python 3.10+
* **Primary Engine:** PyTorch (v2.0+) - Core optimization tensors and autograd loops.
* **Geometric Processing:** PyTorch Geometric (PyG) - Powering the graph neural networks (GATv2) for structural mesh parsing.
* **Reinforcement Learning Core:** Stable-Baselines3 or custom Gymnasium environment wrappers for execution of the continuous PPO algorithm.
* **Experiment Tracking:** Weights & Biases (WandB) - Model logging, convergence analysis, and reward function verification.

## 2. 3D Geometry Processing & Environment Emulation
* **Data Pipelines:** Trimesh / Open3D - Mesh loading, vertex color injection, normals extraction, Laplacian matrix evaluations, and analytical geometry tracking.
* **Vision Models:** HuggingFace Transformers (CLIP `ViT-B/32`) - For extracting embeddings from rendered states to calculate semantic diversity metrics.
* **Human Annotation Interface:** Gradio / Streamlit - A lightweight web engine designed to serve pairwise comparative evaluations for technical artists.

## 3. Real-Time Engine Deployment
* **Target Engine:** Unreal Engine 5.4+
* **Asset Formats:** FBX / OBJ for baseline parsing; glTF / USD for flexible dynamic pipelines.
* **Vertex Scripting:** Custom Material Graphs featuring extensive World Position Offset (WPO) expressions.
* **Instance Placement Engines:** Hierarchical Instanced Static Meshes (HISM) combined with Unreal PCG Graphs to deploy structural variations in real-time.
