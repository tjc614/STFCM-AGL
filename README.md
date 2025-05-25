# STFCM-AGL

A deep learning framework for time series forecasting using STFCM-AGL.

## Project Structure

```
├── run.py              # Main entry point
├── models.py           # Model definitions (AGL-STFCM, GraphSAGE)
├── trainer.py          # Training logic
├── data_loader.py      # Data loading utilities
├── metrics.py          # Evaluation metrics
├── config.py           # Configuration classes
├── utils.py            # Utility functions
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Supported Datasets

1. **TAIEX** - Taiwan Stock Exchange Capitalization Weighted Stock Index
2. **SSE** - Shanghai Stock Exchange Composite Index  
3. **Traffic** - Traffic flow data
4. **Temperature** - Environmental temperature sensor data
5. **EPC** - Electric Power Consumption data
6. **EEG** - Electroencephalography brain signal data

## Quick Start

### Single Run
```bash
python run.py --dataset taiex --mode single --epochs 100
```

### Grid Search
```bash
python run.py --dataset taiex --mode grid_search
```

### Dataset-specific Examples

```bash
# TAIEX dataset
python run.py --dataset taiex --year 2004 --order 4 --d_hidden 5

# SSE dataset  
python run.py --dataset sse --year 2022 --order 3 --d_hidden 4

# Traffic dataset
python run.py --dataset traffic --order 5 --filter_nums 12

# Temperature dataset
python run.py --dataset temp --order 4 --d_hidden 6

# EPC dataset
python run.py --dataset epc --order 3 --kernel_size 5

# EEG dataset
python run.py --dataset eeg --eeg_subject 1 --order 4
```

## Command Line Arguments

### Dataset Parameters
- `--dataset`: Dataset to use (taiex, sse, traffic, temp, epc, eeg)
- `--year`: Year for TAIEX/SSE dataset
- `--eeg_subject`: Subject number for EEG dataset (1-6)

### Model Parameters  
- `--order`: Time window size (default: 4)
- `--d_hidden`: Hidden dimension size (default: 4)
- `--filter_nums`: Number of TCN filters (default: 10)
- `--kernel_size`: Convolution kernel size (default: 3)
- `--aggregator_type`: GraphSAGE aggregator (default: adaptive_pool)
- `--lambda1`: Lambda1 for granularity allocation (default: 1.0)
- `--lambda2`: Lambda2 for granularity allocation (default: 1.0)

### Training Parameters
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--mode`: Running mode (single, grid_search)

## Model Architecture

The AGL-STFCM model consists of:

1. **GraphSAGE Layers**: Learn node embeddings with adaptive granularity allocation
2. **Temporal Convolutional Networks (TCN)**: Capture temporal dependencies
3. **Granularity Allocation Strategy**: Dynamic feature scaling using sigmoid functions

## Key Features

- **Multi-scale Temporal Modeling**: Handles different time scales in data
- **Graph-based Feature Learning**: Captures relationships between variables
- **Adaptive Granularity**: Dynamic feature importance weighting
- **Comprehensive Evaluation**: RMSE, MAE, and MAPE metrics

## Requirements

```
tensorflow>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
tcn>=3.0.0
```

## Installation

```bash
pip install -r requirements.txt
```

## Output

Results are saved to `./save/` directory with timestamp, containing:
- Model weights
- Training history
- Evaluation metrics
- Configuration used
- Source code backup

