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

```
vehicle-trajectory-prediction/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnp_model.py              # Main GNP model
│   │   ├── goal_prediction.py        # Goal prediction sub-module
│   │   ├── trajectory_prediction.py  # Neural social force sub-module
│   │   └── baseline_models.py        # Baseline implementations
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py           # Dataset loading utilities
│   │   ├── preprocessing.py         # Data preprocessing functions
│   │   └── intention_clustering.py  # Intention mode extraction
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py               # Evaluation metrics (ADE, FDE, RMSE)
│   │   ├── visualization.py         # Force visualization and plotting
│   │   ├── physics.py               # Social force physics utilities
│   │   └── config.py                # Configuration management
│   │
│   └── training/
│       ├── __init__.py
│       ├── train_goal_prediction.py # Goal prediction training
│       ├── train_trajectory.py      # Trajectory prediction training
│       └── train_gnp.py             # End-to-end training
│
├── data/
│   ├── raw/
│   │   ├── highd/                   # HighD dataset
│   │   └── ngsim/                   # NGSIM dataset
│   ├── processed/                   # Preprocessed data
│   └── intention_modes/             # Clustered intention patterns
│
├── models/
│   ├── pretrained/                  # Pre-trained model weights
│   ├── checkpoints/                 # Training checkpoints
│   └── exported/                    # Exported models for deployment
│
├── notebooks/
│   ├── data_exploration.ipynb       # Dataset analysis
│   ├── intention_analysis.ipynb     # Intention mode visualization
│   ├── model_evaluation.ipynb       # Performance evaluation
│   └── force_visualization.ipynb    # Physics interpretation
│
├── experiments/
│   ├── baseline_comparison/         # Baseline model experiments
│   ├── ablation_studies/           # Component analysis
│   └── generalization_tests/       # Cross-dataset evaluation
│
├── docs/
│   ├── figures/                    # Paper figures and visualizations
│   ├── api_reference.md            # API documentation
│   ├── dataset_guide.md            # Dataset preparation guide
│   └── model_details.md            # Detailed model explanation
│
├── tests/
│   ├── test_data_processing.py     # Data pipeline tests
│   ├── test_models.py              # Model functionality tests
│   └── test_training.py            # Training pipeline tests
│
└── scripts/
    ├── download_data.sh            # Dataset download script
    ├── preprocess_data.py          # Data preprocessing script
    ├── evaluate_model.py           # Model evaluation script
    └── run_experiments.py          # Experiment runner
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

### 4. Inference

```python
import torch
from src.models.gnp_model import GNPModel
from src.data.data_loader import HighDDataLoader

# Load pre-trained model
model = GNPModel.load_from_checkpoint('models/pretrained/gnp_highd.pth')
model.eval()

# Load test data
data_loader = HighDDataLoader('data/processed/highd/test/', batch_size=1)

# Make predictions
with torch.no_grad():
    for batch in data_loader:
        observed_trajectories = batch['observed']
        neighboring_vehicles = batch['neighbors']
        
        # Predict goals
        predicted_goals, goal_probs = model.predict_goals(
            observed_trajectories, neighboring_vehicles
        )
        
        # Generate trajectories
        predicted_trajectories = model.predict_trajectories(
            observed_trajectories, neighboring_vehicles, predicted_goals
        )
        
        # Visualize results
        model.visualize_forces(
            observed_trajectories, predicted_trajectories, 
            save_path='results/force_analysis.png'
        )
```

## Key Results

### Performance Comparison (RMSE in meters)

| Model | HighD Dataset |||||  NGSIM Dataset ||||| 
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|       | 1s | 2s | 3s | 4s | 5s | 1s | 2s | 3s | 4s | 5s |
| S-LSTM | 0.22 | 0.62 | 1.27 | 2.15 | 3.41 | 0.65 | 1.31 | 2.16 | 3.25 | 4.55 |
| CS-LSTM | 0.22 | 0.61 | 1.24 | 2.10 | 3.27 | 0.61 | 1.27 | 2.08 | 3.10 | 4.37 |
| STDAN | 0.29 | 0.68 | 1.17 | 1.88 | 2.76 | 0.42 | 1.01 | 1.69 | 2.56 | 3.67 |
| CDSTraj | 0.13 | 0.21 | 0.32 | 0.38 | 1.05 | 0.36 | 0.86 | 1.36 | 2.02 | 2.85 |
| **GNP (Ours)** | **0.09** | **0.17** | **0.26** | **0.37** | **0.50** | **0.27** | **0.55** | **0.86** | **1.21** | **1.59** |

### Computational Performance

| Model | Batch Time (ms) | Per-Sample (ms) |
|-------|-----------------|-----------------|
| DenseTNT | 13,298.90 | 208.70 |
| MultiPath++ | 175.36 | 2.74 |
| **GNP (Ours)** | **2,348.68** | **36.71** |

![Visualization Results](docs/figures/force_visualization.png)

*Figure 2: Interpretable force analysis showing goal attraction (yellow) and repulsion forces (blue/black)*

## Key Figures

### Model Architecture
![Architecture](docs/figures/model_architecture.png)

### Intention Mode Clustering
![Intention Modes](docs/figures/intention_modes.png)

### Force Visualization Examples
![Force Analysis](docs/figures/force_examples.png)

### Performance Comparison
![Results](docs/figures/performance_comparison.png)

## Configuration

Key configuration parameters in `src/utils/config.py`:

```python
# Model hyperparameters
MODEL_CONFIG = {
    'goal_prediction': {
        'transformer_layers': 2,
        'attention_heads': 4,
        'embedding_dim': 128,
        'n_goals': 20,
        'n_intention_modes': 200
    },
    'trajectory_prediction': {
        'lstm_hidden_dim': 64,
        'mlp_hidden_dims': [128, 64, 32],
        'time_steps': 50,
        'integration_method': 'euler'
    },
    'social_force': {
        'goal_force_scale': 1.0,
        'repulsion_force_scale': 2.0,
        'collision_radius': 2.0,
        'lane_width': 3.5
    }
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'scheduler': 'cosine',
    'early_stopping_patience': 10
}
```

## Datasets

### Supported Datasets
- **HighD**: German highway drone recordings (25 Hz, 420m sections)
- **NGSIM**: US highway camera data (10 Hz, I-80 & US-101)

### Data Format
```python
# Expected input format
{
    'observed_trajectory': torch.Tensor,  # Shape: [batch, time_obs, 2]
    'future_trajectory': torch.Tensor,    # Shape: [batch, time_pred, 2] 
    'neighboring_vehicles': torch.Tensor, # Shape: [batch, n_neighbors, time_obs, 2]
    'lane_markings': torch.Tensor,        # Shape: [batch, n_lanes, 2]
    'vehicle_id': torch.LongTensor        # Shape: [batch]
}
```

## Ablation Studies

Component contribution analysis:

| Variant | Intention Modes | Goal Force | Repulsion Force | ADE/FDE/RMSE |
|---------|----------------|------------|-----------------|--------------|
| w/o IM | ✗ | ✓ | ✓ | 2.21/4.02/3.53 |
| w/o Rep | ✓ | ✓ | ✗ | 0.87/1.46/0.77 |
| **Full GNP** | **✓** | **✓** | **✓** | **0.59/1.07/0.50** |

## To-Do List

### High Priority
- [ ] **Multi-lane Highway Extension**: Expand to complex highway interchanges and ramps
- [ ] **Urban Scenario Support**: Adapt model for city driving with traffic lights and intersections  
- [ ] **Real-time Optimization**: Further reduce inference latency for deployment
- [ ] **Uncertainty Quantification**: Improve confidence estimation for safety-critical applications

### Medium Priority
- [ ] **Advanced Intention Modeling**: Explore hierarchical intention representations
- [ ] **Enhanced Physics Model**: Develop more sophisticated potential field formulations
- [ ] **Cross-dataset Generalization**: Improve transfer learning between different traffic scenarios
- [ ] **Multi-agent Interaction**: Model complex multi-vehicle coordination scenarios

### Low Priority
- [ ] **Mobile Deployment**: Optimize for edge computing devices
- [ ] **Synthetic Data Generation**: Create physics-based data augmentation
- [ ] **Interactive Visualization**: Web-based force analysis tool
- [ ] **API Development**: REST API for real-time predictions

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
black src/
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{gan2024goal,
  title={Goal-based Neural Physics Vehicle Trajectory Prediction Model},
  author={Gan, Rui and Shi, Haotian and Li, Pei and Wu, Keshu and An, Bocheng and You, Junwei and Li, Linheng and Ma, Junyi and Ma, Chengyuan and Ran, Bin},
  journal={Transportation Research Part C: Emerging Technologies},
  year={2024},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Traffic Operations and Safety (TOPS) Laboratory at University of Wisconsin-Madison
- HighD and NGSIM dataset providers
- Open-source trajectory prediction community

## Contact

- **Rui Gan**: [email@wisc.edu](mailto:email@wisc.edu)
- **Haotian Shi**: [shihaotian95@tongji.edu.cn](mailto:shihaotian95@tongji.edu.cn) 
- **Pei Li**: [pei.li@wisc.edu](mailto:pei.li@wisc.edu)

## Updates

- **2024-06-01**: Initial repository setup and model implementation
- **2024-06-01**: Added comprehensive documentation and examples
- **2024-06-01**: Released pre-trained models for HighD and NGSIM datasets

---

**Project Status**: ✅ Active Development | 📊 Research Paper Submitted | 🚀 Ready for Deployment
