# Neural Architecture Search Implementation

This project focuses on classifying EEG signals into different driving behavior categories using deep learning and Neural Architecture Search (NAS). The goal was to automatically discover optimal neural network architectures for EEG data and compare a generic NAS-based CNN with an EEG-specific architecture inspired by EEGNet.

The pipeline handles raw EEG `.set` files, performs preprocessing, balances class distributions, and trains models using PyTorch with Optuna-based hyperparameter optimization.

## Dataset

- Input: EEG `.set` files loaded using MNE
- Each file contains epochs (trials × channels × time)
- Total channels: 59
- Labels mapped into 5 behavioral classes: (Smooth, Acceleration, Deceleration, Lane Change and Turning)


| Component        | Description                              |
|------------------|------------------------------------------|
| Data Format      | EEG epochs (NumPy arrays)                |
| Channels         | 59 EEG channels                         |
| Preprocessing    | Standardization (mean=0, std=1)          |
| Labels           | 5 driving behavior classes              |

## What This Project Does

- Loads EEG data using MNE and extracts trials and labels :contentReference[oaicite:0]{index=0}  
- Maps raw event IDs into meaningful behavior classes  
- Normalizes EEG signals across time dimension  
- Handles class imbalance using weighted loss  
- Splits dataset into train and test sets  
- Implements PyTorch Dataset and DataLoader pipelines  
- Uses mixed precision training for performance optimization  

# Model Architectures

1. **NAS-based CNN (Optuna-driven)**

- Dynamically builds 1D CNN architectures
- Searches over:
  - Number of layers (2–5)
  - Channels (16, 32, 64)
  - Kernel sizes (3, 5, 7)
  - Pooling strategies
  - Hidden layer size and dropout
- Uses AdaptiveAvgPooling + Fully Connected classifier

2. **EEG-specific NAS (EEGNet-style)**

- Designed specifically for EEG signals
- Includes:
  - Temporal convolution (captures frequency patterns)
  - Depthwise spatial convolution (captures channel relationships)
  - Separable convolution (efficient feature extraction)
- Uses 2D convolutions tailored for EEG structure
- Hyperparameters optimized via Optuna:
  - Temporal kernel size
  - Number of filters
  - Depth multiplier
  - Dropout rate

### Training and Optimization

- Optimizer: Adam
- Loss: CrossEntropyLoss with class weights
- Mixed precision training using CUDA AMP
- NAS search:
  - Basic CNN: 50 trials
  - EEG-specific model: 100 trials
- Final models trained for longer epochs after search

### Evaluation

- Metric: Classification Accuracy
- Evaluated on held-out test set
- Overfitting monitored via training vs testing performance
- Best architecture selected using Optuna study results

### Dependencies

   ```bash
   pip install mne optuna torch torchvision torchaudio scikit-learn tqdm
   ```