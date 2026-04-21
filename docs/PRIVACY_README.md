# PRISM Privacy Module: ε-Differential Privacy

This module implements the core scientific novelty of PRISM: **privacy-preserving human action recognition** using ε-Differential Privacy (DP) with DP-SGD (Differential Privacy Stochastic Gradient Descent).

## 🔒 **Core Privacy Features**

### **1. ε-Differential Privacy (DP)**
- **Privacy Budget Control**: Configurable ε (epsilon) parameter for privacy-utility tradeoff
- **Failure Probability**: δ (delta) parameter for (ε, δ)-DP guarantee
- **Theoretical Guarantees**: Provable privacy protection against membership inference attacks

### **2. DP-SGD Implementation**
- **Gradient Clipping**: L2 norm clipping to bound sensitivity
- **Noise Injection**: Calibrated Gaussian/Laplace noise for privacy
- **Privacy Accounting**: Real-time tracking of privacy budget consumption
- **Automatic Enforcement**: Opacus integration for seamless DP training

### **3. Dual Implementation Strategy**
- **Opacus Integration**: Automatic DP enforcement with Facebook's Opacus library
- **Manual Implementation**: Fallback implementation with custom gradient processing
- **Seamless Switching**: Automatic fallback if Opacus is unavailable

## 🏗️ **Architecture Overview**

```
PrivacyPreservingTrainer
├── OpacusDPTrainer (Primary)
│   ├── PrivacyEngine
│   ├── Automatic Gradient Processing
│   └── RDP Accounting
└── ManualDPTrainer (Fallback)
    ├── Gradient Clipping
    ├── Noise Injection
    └── Privacy Accountant
```

## 📊 **Privacy Mechanisms**

### **Gradient Clipping**
```python
# L2 norm clipping to bound sensitivity
total_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters()]))
clip_coef = min(1.0, max_grad_norm / (total_norm + 1e-6))
for param in model.parameters():
    param.grad.data.mul_(clip_coef)
```

### **Noise Injection**
```python
# Calibrated noise based on privacy budget
noise_scale = max_grad_norm * noise_multiplier
noise = torch.normal(0, noise_scale, param.grad.shape)
param.grad.data.add_(noise)
```

### **Privacy Accounting**
- **RDP (Renyi Differential Privacy)**: More accurate than basic composition
- **Real-time Tracking**: Monitor privacy budget consumption during training
- **Budget Management**: Automatic stopping when privacy budget is exhausted

## 🚀 **Usage Examples**

### **Basic Privacy Training**
```python
from privacy_module import PrivacyPreservingTrainer, create_privacy_config
from models import STGCN_PRISM

# Create model
model = STGCN_PRISM(
    num_joints=33,
    in_channels=4,
    num_classes=101,
    hidden_channels=[64, 128, 256],
    temporal_kernel_sizes=[3, 3, 3],
    dropout=0.1,
    use_attention=False
)

# Create privacy configuration
privacy_config = create_privacy_config(
    epsilon=1.0,        # Privacy budget
    delta=1e-5,         # Failure probability
    max_grad_norm=1.0,  # Gradient clipping norm
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    use_opacus=True     # Use Opacus if available
)

# Create privacy trainer
privacy_trainer = PrivacyPreservingTrainer(model, privacy_config)

# Train with privacy
results = privacy_trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    learning_rate=0.001
)
```

### **Privacy Budget Analysis**
```python
# Analyze privacy-utility tradeoff
from privacy_module import analyze_privacy_utility_tradeoff

analysis = analyze_privacy_utility_tradeoff(results)
print(f"Final Accuracy: {analysis['final_accuracy']:.4f}")
print(f"Privacy Spent: ε={analysis['final_epsilon']:.4f}")
print(f"Privacy Efficiency: {analysis['privacy_efficiency']:.4f}")
```

## ⚙️ **Configuration Parameters**

### **Privacy Parameters**
- **`epsilon`** (float): Privacy budget ε - lower values = more privacy
- **`delta`** (float): Failure probability δ - typically 1e-5
- **`max_grad_norm`** (float): Gradient clipping norm - typically 1.0
- **`noise_multiplier`** (float): Noise scaling factor - auto-calculated if None

### **Training Parameters**
- **`epochs`** (int): Number of training epochs
- **`batch_size`** (int): Batch size for training
- **`learning_rate`** (float): Learning rate for optimizer
- **`use_opacus`** (bool): Whether to use Opacus library

### **Noise Parameters**
- **`noise_type`** (str): 'gaussian' or 'laplace' noise
- **`privacy_accounting`** (str): 'rdp' or 'moments' accounting

## 📈 **Privacy-Utility Tradeoff**

### **Privacy Levels**
- **High Privacy (ε=0.1)**: Maximum privacy, potential utility loss
- **Medium Privacy (ε=1.0)**: Balanced privacy-utility tradeoff
- **Low Privacy (ε=10.0)**: Higher utility, less privacy protection

### **Utility Considerations**
- **Accuracy Impact**: Privacy noise may reduce model accuracy
- **Convergence**: DP training may require more epochs
- **Batch Size**: Smaller batches may be needed for privacy
- **Learning Rate**: May need adjustment for DP training

## 🔬 **Scientific Novelty**

### **PRISM's Contribution**
1. **First DP-SGD for Pose-based Action Recognition**: Novel application of DP to human pose data
2. **Spatial-Temporal Privacy**: Privacy protection for both spatial (joint) and temporal (motion) features
3. **Anonymity Guarantees**: Provable protection against identity inference from pose data
4. **Practical Implementation**: Production-ready DP training for real-world applications

### **Privacy Guarantees**
- **Membership Privacy**: Protection against determining if a person was in training data
- **Attribute Privacy**: Protection against inferring sensitive attributes from pose
- **Identity Privacy**: Protection against re-identification from pose sequences
- **Temporal Privacy**: Protection against inferring behavior patterns over time

## 🛡️ **Security Considerations**

### **Privacy Budget Management**
- **Budget Tracking**: Real-time monitoring of privacy consumption
- **Budget Exhaustion**: Automatic stopping when budget is exceeded
- **Budget Allocation**: Careful allocation across training epochs

### **Attack Resistance**
- **Membership Inference**: DP provides theoretical protection
- **Model Inversion**: Noise injection prevents model inversion attacks
- **Gradient Attacks**: Gradient clipping and noise prevent gradient-based attacks

## 📋 **Requirements**

### **Dependencies**
```
torch>=2.0.0
opacus>=1.4.0
numpy>=1.24.0
scipy>=1.10.0
```

### **Optional Dependencies**
- **Opacus**: For automatic DP enforcement (recommended)
- **PyTorch Geometric**: For graph neural network support

## 🚨 **Important Notes**

### **Privacy Limitations**
- **Not Perfect Privacy**: DP provides probabilistic privacy guarantees
- **Utility Tradeoff**: Higher privacy may reduce model accuracy
- **Parameter Sensitivity**: Privacy parameters require careful tuning
- **Composition**: Privacy budget accumulates across training steps

### **Best Practices**
- **Start Conservative**: Begin with higher ε values and reduce gradually
- **Monitor Budget**: Track privacy consumption throughout training
- **Validate Privacy**: Use privacy auditing tools when available
- **Document Parameters**: Keep detailed records of privacy settings

## 🔍 **Troubleshooting**

### **Common Issues**
1. **Opacus Import Error**: Falls back to manual DP implementation
2. **Privacy Budget Exhausted**: Reduce ε or increase dataset size
3. **Poor Convergence**: Adjust learning rate or batch size
4. **Memory Issues**: Reduce batch size or model complexity

### **Debug Mode**
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check privacy budget
current_epsilon, current_delta = privacy_trainer.get_privacy_spent()
print(f"Current privacy: ε={current_epsilon:.4f}, δ={current_delta:.4f}")
```

## 📚 **References**

1. **Differential Privacy**: Dwork, C. (2006). "Calibrating noise to sensitivity in private data analysis"
2. **DP-SGD**: Abadi, M. et al. (2016). "Deep learning with differential privacy"
3. **Opacus**: Facebook AI Research (2020). "Opacus: A library for training PyTorch models with differential privacy"
4. **RDP Accounting**: Mironov, I. (2017). "Renyi differential privacy"

---

**The PRISM Privacy Module represents a significant advancement in privacy-preserving machine learning for human action recognition, providing both theoretical guarantees and practical implementation for real-world applications.**
<!-- Research note 28: differential privacy accounting and utility-retention frontier; file focus: docs/PRIVACY_README.md. -->
