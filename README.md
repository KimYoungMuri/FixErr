# FixErr: Transformer-Based Code Repair Model

## Project Evolution
This project began as an exploration of code repair using LSTM-based architectures. After initial experiments with LSTM, we transitioned to a transformer-based approach to leverage the power of self-attention mechanisms and parallel processing capabilities. This evolution was motivated by the transformer architecture's superior ability to capture long-range dependencies and its proven success in various sequence-to-sequence tasks.

## Overview
FixErr is a neural network model designed to automatically fix bugs in code. It uses a transformer-based architecture to learn the mapping between buggy code and its corrected version, effectively acting as an automated code repair system.

## Architecture

### Model Components
1. **Encoder-Decoder Architecture**
   - Custom implementation of transformer architecture
   - Multi-head self-attention in encoder
   - Cross-attention in decoder
   - Position embeddings for sequential code representation

2. **Attention Mechanism**
   - Custom implementation of multi-head attention
   - Explicit shape checking and error handling
   - Dedicated output projection layer
   - Support for both self-attention and cross-attention

3. **Training Infrastructure**
   - GPU support with CUDA compatibility
   - Gradient accumulation for stable training
   - Learning rate scheduling
   - Checkpointing system
   - Experiment tracking via Weights & Biases

## Technical Details

### Model Configuration
- Hidden size: 768
- Number of layers: 6
- Attention heads: 12
- Intermediate size: 3072
- Maximum position embeddings: 512
- Dropout rate: 0.1
- Layer normalization epsilon: 1e-12

### Training Setup
- Batch size: 8
- Number of epochs: 3
- Evaluation steps: 1000
- Checkpoint saving: Every 5000 steps
- Gradient accumulation steps: 1
- Learning rate scheduling with warmup

### Data Processing
- Tokenization using BERT tokenizer
- Custom dataset class for code samples
- Attention mask generation
- Proper tensor shape handling
- GPU-optimized data loading with pin_memory

## Project Structure
```
.
├── models/
│   └── transformer_model.py    # Core model architecture
├── training/
│   └── trainer.py             # Training infrastructure
├── data/
│   └── dataset.py             # Dataset handling
├── preprocessing/
│   └── code_processor.py      # Code preprocessing
├── config/
│   └── model_config.py        # Model configuration
├── checkpoints/               # Saved model checkpoints
├── train.py                   # Training script
└── requirements.txt           # Project dependencies
```

## Setup and Installation

1. **Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Data Preparation**
   - Place your code repair dataset in the appropriate format
   - Run preprocessing scripts if necessary

3. **Training**
   ```bash
   # Basic training
   python train.py

   # Training with sleep prevention (Mac)
   caffeinate python train.py
   ```

## Key Features
- Custom transformer implementation for fine-grained control
- Robust error handling and shape checking
- GPU acceleration support
- Comprehensive training monitoring
- Checkpoint system for training recovery
- Experiment tracking and visualization

## Future Work
- [ ] Add results and performance metrics
- [ ] Implement model inference pipeline
- [ ] Add support for more programming languages
- [ ] Optimize training speed
- [ ] Add model quantization for deployment
