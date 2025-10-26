# PRISM API Reference

This document provides detailed API documentation for the PRISM framework.

## Data Pipeline (`data_pipeline.py`)

### Core Functions

#### `extract_and_normalize_pose(video_path: str) -> Tuple[np.ndarray, np.ndarray]`
Extract pose landmarks from video and normalize coordinates.

**Parameters:**
- `video_path`: Path to input video file

**Returns:**
- `pose_data`: Normalized pose coordinates (T, 33, 4)
- `confidence`: Detection confidence scores (T, 33)

#### `extract_and_normalize_pose_with_kinematics(video_path: str) -> Tuple[np.ndarray, np.ndarray]`
Extract pose landmarks and compute kinematic features.

**Parameters:**
- `video_path`: Path to input video file

**Returns:**
- `pose_data`: Normalized pose coordinates (T, 33, 4)
- `kinematic_features`: Computed kinematic features (T, 100)

#### `kinematic_features(normalized_poses: np.ndarray) -> np.ndarray`
Transform normalized pose coordinates into kinematic features.

**Parameters:**
- `normalized_poses`: Normalized pose data (T, 33, 4)

**Returns:**
- `features`: Kinematic features (T, 100)

### Dataset Class

#### `PRISMDataset`
PyTorch Dataset for loading pose sequences and kinematic features.

**Parameters:**
- `data_dir`: Directory containing .npy files
- `labels_file`: Path to labels pickle file
- `sequence_length`: Fixed sequence length (default: 30)
- `transform`: Optional data transformation
- `use_kinematics`: Whether to use kinematic features (default: True)

**Methods:**
- `__len__()`: Return dataset size
- `__getitem__(idx)`: Return sample at index

## Models (`models.py`)

### Baseline Models

#### `BaselineLSTM`
2-layer LSTM for pose sequence classification.

**Parameters:**
- `input_size`: Input feature dimension
- `hidden_size`: LSTM hidden size
- `num_layers`: Number of LSTM layers
- `num_classes`: Number of output classes
- `dropout`: Dropout rate
- `bidirectional`: Whether to use bidirectional LSTM

#### `BaselineGRU`
2-layer GRU for pose sequence classification.

**Parameters:**
- `input_size`: Input feature dimension
- `hidden_size`: GRU hidden size
- `num_layers`: Number of GRU layers
- `num_classes`: Number of output classes
- `dropout`: Dropout rate
- `bidirectional`: Whether to use bidirectional GRU

### ST-GCN Model

#### `STGCN_PRISM`
Spatial-Temporal Graph Convolutional Network for action recognition.

**Parameters:**
- `num_joints`: Number of pose joints (default: 33)
- `in_channels`: Input feature channels (default: 4)
- `num_classes`: Number of output classes
- `hidden_channels`: List of hidden channel sizes
- `temporal_kernel_sizes`: List of temporal kernel sizes
- `dropout`: Dropout rate
- `use_attention`: Whether to use attention mechanism

### Utility Functions

#### `get_human_skeleton_adjacency_matrix() -> torch.Tensor`
Create adjacency matrix for human skeleton based on MediaPipe landmarks.

**Returns:**
- `adjacency_matrix`: Binary adjacency matrix (33, 33)

#### `train_baseline(model_type: str, **kwargs) -> Dict`
Train baseline models with standard training loop.

**Parameters:**
- `model_type`: Type of model ('lstm', 'gru', 'stgcn')
- `train_loader`: Training data loader
- `val_loader`: Validation data loader
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate
- `weight_decay`: Weight decay
- `num_classes`: Number of output classes
- `device`: Device to use ('auto', 'cpu', 'cuda')
- `save_model`: Whether to save model
- `model_save_path`: Path to save model

**Returns:**
- `results`: Dictionary containing training results

## Privacy Module (`privacy_module.py`)

### Configuration

#### `PrivacyConfig`
Configuration class for differential privacy parameters.

**Attributes:**
- `epsilon`: Privacy budget (ε)
- `delta`: Failure probability (δ)
- `max_grad_norm`: Gradient clipping threshold
- `noise_multiplier`: Noise multiplier for DP-SGD
- `epochs`: Number of training epochs
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `use_opacus`: Whether to use Opacus library

### Privacy Trainers

#### `PrivacyPreservingTrainer`
Main trainer class for privacy-preserving model training.

**Parameters:**
- `model`: PyTorch model to train
- `privacy_config`: Privacy configuration

**Methods:**
- `train(train_loader, val_loader, num_epochs, learning_rate)`: Train model with privacy

#### `ManualDPTrainer`
Manual implementation of DP-SGD training.

**Parameters:**
- `model`: PyTorch model to train
- `privacy_config`: Privacy configuration

**Methods:**
- `train(train_loader, val_loader, num_epochs, learning_rate)`: Train model with manual DP-SGD

### Utility Functions

#### `create_privacy_config(epsilon: float, **kwargs) -> PrivacyConfig`
Create privacy configuration with automatic parameter calculation.

**Parameters:**
- `epsilon`: Privacy budget
- `delta`: Failure probability (default: 1e-5)
- `max_grad_norm`: Gradient clipping threshold (default: 1.0)
- `epochs`: Number of epochs (default: 1)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 0.001)
- `use_opacus`: Whether to use Opacus (default: True)

**Returns:**
- `config`: PrivacyConfig object

## Benchmarks (`benchmarks.py`)

### Evaluation Functions

#### `evaluate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict`
Evaluate classification performance metrics.

**Parameters:**
- `y_true`: True labels
- `y_pred`: Predicted labels
- `num_classes`: Number of classes

**Returns:**
- `metrics`: Dictionary of evaluation metrics

#### `measure_inference_latency(model: nn.Module, data_loader: DataLoader, **kwargs) -> Dict`
Measure model inference latency.

**Parameters:**
- `model`: PyTorch model
- `data_loader`: Data loader for inference
- `device`: Device to use ('auto', 'cpu', 'cuda')
- `num_warmup`: Number of warmup iterations
- `num_measurements`: Number of measurement iterations

**Returns:**
- `latency_results`: Dictionary of latency metrics

### Model-Specific Evaluators

#### `evaluate_baseline_lstm(model: nn.Module, test_loader: DataLoader, **kwargs) -> Dict`
Evaluate LSTM baseline model.

#### `evaluate_stgcn_no_dp(model: nn.Module, test_loader: DataLoader, **kwargs) -> Dict`
Evaluate STGCN model without differential privacy.

#### `evaluate_stgcn_with_dp(model: nn.Module, test_loader: DataLoader, **kwargs) -> Dict`
Evaluate STGCN model with differential privacy.

### Comparison and Reporting

#### `generate_comparison_table(lstm_results: Dict, stgcn_no_dp_results: Dict, stgcn_with_dp_results: Dict, **kwargs) -> Dict`
Generate comparison table for all model variants.

**Parameters:**
- `lstm_results`: LSTM evaluation results
- `stgcn_no_dp_results`: STGCN (no DP) evaluation results
- `stgcn_with_dp_results`: STGCN (with DP) evaluation results
- `save_path`: Path to save results (optional)
- `format`: Output format ('csv', 'latex', 'json')

**Returns:**
- `comparison_data`: Formatted comparison results

#### `run_comprehensive_benchmark(lstm_model: nn.Module, stgcn_model: nn.Module, dp_model: nn.Module, test_loader: DataLoader, **kwargs) -> Dict`
Run comprehensive benchmark across all model variants.

**Parameters:**
- `lstm_model`: Trained LSTM model
- `stgcn_model`: Trained STGCN model
- `dp_model`: Trained DP-STGCN model
- `test_loader`: Test data loader
- `privacy_epsilon`: Privacy budget used
- `privacy_delta`: Privacy failure probability
- `output_dir`: Output directory for results
- `device`: Device to use

**Returns:**
- `benchmark_results`: Complete benchmark results

## Optimization (`optimization.py`)

### Quantization Functions

#### `quantize_model(stgcn_model: nn.Module, calibration_data: DataLoader, **kwargs) -> nn.Module`
Apply post-training dynamic quantization to STGCN model.

**Parameters:**
- `stgcn_model`: Trained STGCN model
- `calibration_data`: Data loader for calibration
- `device`: Device to use

**Returns:**
- `quantized_model`: Quantized model

#### `quantize_lstm_model(lstm_model: nn.Module, calibration_data: DataLoader, **kwargs) -> nn.Module`
Apply post-training dynamic quantization to LSTM model.

**Parameters:**
- `lstm_model`: Trained LSTM model
- `calibration_data`: Data loader for calibration
- `device`: Device to use

**Returns:**
- `quantized_model`: Quantized model

### Model Export

#### `save_optimized_model(model: nn.Module, model_name: str, output_dir: str = "optimized_models", **kwargs) -> Dict[str, str]`
Save model in multiple formats for deployment.

**Parameters:**
- `model`: PyTorch model to save
- `model_name`: Name for saved model
- `output_dir`: Output directory
- `device`: Device to use

**Returns:**
- `saved_paths`: Dictionary of saved file paths

### Performance Comparison

#### `compare_model_performance(original_model: nn.Module, quantized_model: nn.Module, test_loader: DataLoader, **kwargs) -> Dict`
Compare original and quantized model performance.

**Parameters:**
- `original_model`: Original Float32 model
- `quantized_model`: Quantized Int8 model
- `test_loader`: Test data loader
- `device`: Device to use

**Returns:**
- `comparison_results`: Performance comparison metrics

#### `optimize_stgcn_model(stgcn_model: nn.Module, calibration_data: DataLoader, test_loader: DataLoader, **kwargs) -> Dict`
Complete optimization pipeline for STGCN model.

**Parameters:**
- `stgcn_model`: Trained STGCN model
- `calibration_data`: Data loader for calibration
- `test_loader`: Test data loader
- `model_name`: Name for optimized model
- `output_dir`: Output directory
- `device`: Device to use

**Returns:**
- `optimization_results`: Complete optimization results

## Error Handling

All functions include comprehensive error handling and validation:

- **Input Validation**: Check parameter types and ranges
- **Device Compatibility**: Automatic device detection and fallback
- **Memory Management**: Efficient memory usage for large models
- **Error Messages**: Clear, actionable error messages
- **Logging**: Detailed logging for debugging and monitoring

## Performance Considerations

- **Batch Processing**: Optimized for batch inference
- **Memory Efficiency**: Minimal memory footprint
- **GPU Acceleration**: Automatic GPU utilization when available
- **Quantization**: Significant model size and speed improvements
- **Privacy Overhead**: Minimal computational overhead for DP training

## Examples

See the `examples/` directory for complete usage examples:

- `train_example.py`: Basic training examples
- `privacy_training_example.py`: Privacy-preserving training
- `kinematic_features_example.py`: Kinematic feature extraction
- `benchmark_example.py`: Model evaluation and benchmarking
- `optimization_example.py`: Model optimization and deployment
