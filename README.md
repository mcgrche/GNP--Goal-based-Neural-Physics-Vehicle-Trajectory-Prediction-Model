# GNP--Goal-based-Neural-Physics-Vehicle-Trajectory-Prediction-Model
This repository contains the manuscript and materials for the paper "Goal-based Neural Physics Vehicle Trajectory Prediction Model"

# Goal-based Neural Physics Vehicle Trajectory Prediction Model (GNP)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2024.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2024.xxxxx)

## Abstract

Vehicle trajectory prediction plays a vital role in intelligent transportation systems and autonomous driving, as it significantly affects vehicle behavior planning and control, thereby influencing traffic safety and efficiency. Numerous studies have been conducted to predict short-term vehicle trajectories in the immediate future. However, long-term trajectory prediction remains a major challenge due to accumulated errors and uncertainties. Additionally, balancing accuracy with interpretability in the prediction is another challenging issue in predicting vehicle trajectory. 

To address these challenges, this paper proposes a **Goal-based Neural Physics Vehicle Trajectory Prediction Model (GNP)**. The GNP model simplifies vehicle trajectory prediction into a two-stage process: determining the vehicle's goal and then choosing the appropriate trajectory to reach this goal. The GNP model contains two sub-modules to achieve this process. The first sub-module employs a multi-head attention mechanism to accurately predict goals. The second sub-module integrates a deep learning model with a physics-based social force model to progressively predict the complete trajectory using the generated goals. The GNP demonstrates state-of-the-art long-term prediction accuracy compared to four baseline models. We provide interpretable visualization results to highlight the multi-modality and inherent nature of our neural physics framework.

## Key Features

- **🎯 Goal-based Prediction**: Explicitly models vehicle intentions and predicts multiple potential goals
- **🔬 Physics-Informed**: Integrates neural networks with social force models for interpretable predictions
- **⚡ Real-time Performance**: Efficient implementation suitable for real-world deployment (36.7ms per sample)
- **📊 State-of-the-art Accuracy**: Superior long-term prediction performance on highway datasets
- **🔍 Interpretable Results**: Visualizable force analysis and intention modeling
- **🌐 Multi-modal Output**: Generates multiple trajectory hypotheses with confidence measures

## Model Architecture

The GNP model consists of two main sub-modules:

### 1. Goal Prediction Sub-module
- **Intention Mode Extraction**: Clusters trajectory patterns to identify driving intentions
- **Transformer Encoder-Decoder**: Enhanced architecture for goal prediction
- **Multi-head Attention**: Captures temporal and spatial dependencies
- **Top-K Goal Sampling**: Generates multiple potential destinations

### 2. Trajectory Prediction Sub-module (Neural Social Force)
- **Goal Attraction Force**: Neural network-learned attraction towards predicted goals
- **Repulsion Force**: Physics-based collision avoidance and lane constraints
- **Vehicle Dynamics**: Numerical integration for trajectory completion
- **Interpretable Parameters**: Learnable force coefficients with physical meaning

![Model Architecture](docs/figures/model_architecture.png)

*Figure 1: GNP model architecture with dual sub-module framework*

## Repository Structure
