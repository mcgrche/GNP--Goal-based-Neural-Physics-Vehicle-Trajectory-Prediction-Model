# Goal-based Neural Physics Vehicle Trajectory Prediction Model (GNP)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2409.15182-b31b1b.svg)](https://arxiv.org/abs/2409.15182)

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
- **Top-K Goal Sampling**: Generates multiple potential destinations and their probabilities

### 2. Trajectory Prediction Sub-module (Neural Social Force)
- **Goal Attraction Force**: Neural network-learned attraction towards predicted goals
- **Repulsion Force**: Physics-based collision avoidance and lane constraints
- **Vehicle Dynamics**: Numerical integration for trajectory completion
- **Interpretable Parameters**: Learnable force coefficients with physical meaning

![Model Architecture](assets/framework_v2.png)

*Figure 1: GNP model architecture with dual sub-module framework*

## Quick Start

### 1. Data Preparation

```bash
# Download and preprocess HighD dataset
python scripts/download_data.sh --dataset highd
python scripts/preprocess_data.py --dataset highd --output data/processed/

# Extract intention modes
python src/data/intention_clustering.py --data data/processed/highd/ --n_clusters 200
```

### 2. Training

```bash
# Train goal prediction sub-module
python src/training/train_goal_prediction.py \
    --data_path data/processed/highd/ \
    --intention_modes data/intention_modes/highd_200.pkl \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001

# Train trajectory prediction sub-module
python src/training/train_trajectory.py \
    --data_path data/processed/highd/ \
    --goal_model models/checkpoints/goal_prediction_best.pth \
    --epochs 150 \
    --batch_size 32 \
    --lr 0.0005

# End-to-end fine-tuning
python src/training/train_gnp.py \
    --data_path data/processed/highd/ \
    --pretrained_goal models/checkpoints/goal_prediction_best.pth \
    --pretrained_traj models/checkpoints/trajectory_prediction_best.pth \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.0001
```

### 3. Evaluation

```bash
# Evaluate on test set
python scripts/evaluate_model.py \
    --model_path models/pretrained/gnp_highd.pth \
    --data_path data/processed/highd/test/ \
    --output results/highd_evaluation.json

# Generate visualizations
python src/utils/visualization.py \
    --model_path models/pretrained/gnp_highd.pth \
    --data_path data/processed/highd/test/ \
    --output_dir results/visualizations/
```

## Repository Structure

```
GNP--Goal-based-Neural-Physics-Vehicle-Trajectory-/
├── README.md
├── LICENSE
├── .gitignore
│
├── assets/                              # Project assets and visualizations
│
├── goal-prediction/                     # Goal Prediction Sub-module
│   ├── config/
│   │   └── __pycache__/                # Python cache files
│   ├── HighD.py                        # HighD dataset processing for goal prediction
│   ├── Ngsim.py                        # NGSIM dataset processing for goal prediction
│   ├── loaddata.py                     # Data loading utilities for goal prediction
│   ├── model.py                        # Goal prediction model implementation
│   ├── train.py                        # Training script for goal prediction
│   ├── transformer_decoder.py          # Transformer decoder implementation
│   ├── transformer_encoder.py          # Transformer encoder implementation
│   └── utils.py                        # Utility functions for goal prediction
│
└── neural social force trajectory prediction/    # Neural Social Force Sub-module
    ├── config/
    │   ├── righd_goals.yaml            # HighD dataset goals configuration
    │   └── righd_rep.yaml              # HighD dataset repulsion configuration
    ├── loaddata_goals.py               # Data loading for goal-based prediction
    ├── loaddata_repulsion.py           # Data loading for repulsion modeling
    ├── model_goals.py                  # Goal attraction force model
    ├── model_repulsion.py              # Repulsion force model
    ├── train_goals.py                  # Training script for goal attraction
    ├── train_repulsion_fulltest.py     # Full testing for repulsion training
    └── utils.py                        # Utility functions for trajectory prediction
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/vehicle-trajectory-prediction.git
cd vehicle-trajectory-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Dependencies

```bash
# Core dependencies
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Specific for trajectory prediction
scipy>=1.7.0
tqdm>=4.62.0
tensorboard>=2.7.0
h5py>=3.1.0

# Optional for visualization
plotly>=5.0.0
opencv-python>=4.5.0
```


## Datasets

### Supported Datasets
- **HighD**: German highway drone recordings (25 Hz, 420m sections)
- **NGSIM**: US highway camera data (10 Hz, I-80 & US-101)


## To-Do List

### High Priority
- ✅ **Goal Prediction sub-module**: Codes for transformer-based goal-prediction sub-module
- ✅ **Neural Social Force Trajectory Prediction sub-module**: Codes for neural social force including attraction and repulsive force
- ✅ **HighD dataset Support**: Support HighD dataset and data processing
- ✅ **Repository Structure**: Improve confidence estimation for safety-critical applications
- 🚧 **NGSIM dataset Support**: Support NGSIM dataset and data processing
- 🚧 **Data preprocessing**: Preprocessing raw data files
- 🚧 **Installation**: Installation and implementation detail for data preparation, training, and evaluation

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{gan2024goal,
  title={Goal-based Neural Physics Vehicle Trajectory Prediction Model},
  author={Gan, Rui and Shi, Haotian and Li, Pei and Wu, Keshu and An, Bocheng and Li, Linheng and Ma, Junyi and Ma, Chengyuan and Ran, Bin},
  journal={arXiv preprint arXiv:2409.15182},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please open an issue.
- **Rui Gan**: [rgan6@wisc.edu](mailto:rgan6@wisc.edu)

## Updates

- **2024-06-01**: Initial repository setup and model implementation
- **2024-06-01**: Added comprehensive documentation and examples

---

**Project Status**: ✅ Active Development | 📊 Research Paper Submitted | 🚀 Ready for Deployment
