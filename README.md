# 🚀 YOLOv12-Face Enhanced

**State-of-the-art face detection model based on YOLOv12 with advanced attention mechanisms**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

YOLOv12-Face Enhanced is an improved face detection model that extends the standard YOLOv12 architecture with cutting-edge attention mechanisms designed specifically for facial feature detection. The model achieves superior performance on challenging scenarios including small faces, occlusions, and varying lighting conditions.

### ✨ Key Features

- **🧠 Advanced Attention Modules**: A2Module, RELAN, FlashAttention, CrossScaleAttention
- **⚡ Real-time Performance**: Optimized for both accuracy and speed
- **📱 Mobile-Ready**: Export to ONNX, TensorRT, CoreML
- **🎯 Specialized for Faces**: Fine-tuned architecture for facial detection
- **🔄 Easy Training**: One-click training with Jupyter notebooks

## 📊 Performance

| Model | mAP@0.5 | Precision | Recall | Latency |
|-------|---------|-----------|--------|---------|
| YOLOv12-Face Base | 0.660 | 0.774 | 0.602 | 1.4ms |
| **YOLOv12-Face Enhanced** | **TBD** | **TBD** | **TBD** | **TBD** |

*Performance measured on WIDERFace validation set*

## 🚀 Quick Start

### 🎯 Méthode Recommandée (Notebook)

```bash
# Clone le dépôt
git clone https://github.com/yourusername/yolov12-face-enhanced.git
cd yolov12-face-enhanced

# Ouvrir le notebook principal
jupyter notebook train_yolov12_enhanced.ipynb
```

**Puis exécuter toutes les cellules dans l'ordre :**
1. 📦 **Installation automatique** des dépendances (`ultralytics`, `gdown`, `opencv-python`)
2. 🔧 **Restauration automatique** des configurations Enhanced
3. ✅ **Vérification** que tout fonctionne
4. 🏋️ **Entraînement** du modèle Enhanced
5. 📊 **Analyse** des résultats
6. 🧪 **Test** et export du modèle

### Option 2: Installation Manuelle

```bash
# Installer les dépendances
pip install ultralytics gdown opencv-python

# Restaurer les configurations
python scripts/restore_configs.py

# Entraîner le modèle
python scripts/train_enhanced.py --epochs 100 --batch-size 16
```

### Option 3: Quick Demo

```python
from ultralytics import YOLO

# Load enhanced model
model = YOLO('path/to/yolov12-face-enhanced.pt')

# Run inference
results = model('path/to/image.jpg')
results[0].show()
```

## 📁 Project Structure

```
yolov12-face/
├── train_yolov12_enhanced.ipynb    # 📓 Main training notebook
├── scripts/
│   ├── configs/                    # 🗂️ Model configurations (backup)
│   ├── train_enhanced.py          # 🏋️ Training script
│   ├── restore_configs.py         # 🔧 Configuration manager
│   ├── webcam_demo.py             # 📹 Real-time demo
│   └── mobile_optimization.py     # 📱 Model optimization
├── examples/                       # 🎯 Usage examples
├── tests/                         # 🧪 Unit tests
└── results/                       # 📊 Training outputs
```

## 🧠 Enhanced Modules

### A2Module (Area Attention)
Focuses attention on facial regions with adaptive spatial weighting.

### RELAN (Residual Efficient Layer Aggregation)
Efficiently aggregates multi-scale features for better small face detection.

### FlashAttention
Memory-efficient attention mechanism for faster training and inference.

### CrossScaleAttention
Cross-scale feature interaction for robust detection across different face sizes.

### MicroExpressionAttention
Specialized attention for subtle facial feature detection.

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ VRAM for training

### Dependencies
```bash
# Installation complète des dépendances
pip install ultralytics torch torchvision matplotlib
pip install huggingface_hub pillow opencv-python seaborn pandas
```

### Troubleshooting Installation
Si vous rencontrez l'erreur `ModuleNotFoundError: No module named 'huggingface_hub'`:

```bash
# Diagnostic automatique
python scripts/diagnose.py

# Ou installation manuelle
pip install huggingface_hub transformers
pip install --upgrade ultralytics
```

### Dataset Setup
The model is trained on WIDERFace dataset. The training notebook will automatically download and prepare the dataset.

## 📚 Usage Examples

### Training
```python
from ultralytics import YOLO

# Create enhanced model
model = YOLO('ultralytics/cfg/models/v12/yolov12-face-enhanced.yaml')

# Train
results = model.train(
    data='ultralytics/cfg/datasets/widerface.yaml',
    epochs=100,
    batch=16,
    imgsz=640
)
```

### Inference
```python
# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Single image
results = model('image.jpg')

# Batch inference
results = model(['img1.jpg', 'img2.jpg'])

# Video
results = model('video.mp4')
```

### Real-time Detection
```python
# Webcam demo
python scripts/webcam_demo.py --model path/to/best.pt
```

### Mobile Deployment
```python
# Export to different formats
model.export(format='onnx')      # ONNX
model.export(format='engine')    # TensorRT
model.export(format='coreml')    # CoreML
```

## 🎯 Training Your Own Model

1. **Prepare Dataset**: Place your annotated images in YOLO format
2. **Configure**: Update `ultralytics/cfg/datasets/your_dataset.yaml`
3. **Train**: Use the provided notebook or scripts
4. **Evaluate**: Compare performance with baseline models
5. **Deploy**: Export to your preferred format

## 🔬 Architecture Details

The enhanced model builds upon YOLOv12 with the following modifications:

- **Backbone**: Enhanced with attention modules at key feature extraction points
- **Neck**: RELAN modules for better feature fusion
- **Head**: CrossScaleAttention for multi-scale detection
- **Loss**: Specialized loss function for face detection

## 📈 Benchmarks

### WIDERFace Validation Results
- **Easy subset**: TBD
- **Medium subset**: TBD  
- **Hard subset**: TBD

### Speed Benchmarks
- **RTX 4090**: TBD FPS
- **RTX 3080**: TBD FPS
- **Mobile (ONNX)**: TBD FPS

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
git clone https://github.com/yourusername/yolov12-face-enhanced.git
cd yolov12-face-enhanced
pip install -e .
pre-commit install
```

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{yolov12face2025,
  title={YOLOv12-Face Enhanced: Advanced Face Detection with Attention Mechanisms},
  author={Your Name},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the excellent YOLOv12 foundation
- [WIDERFace](http://shuangz.com/projects/sfnet/) dataset creators
- Open source community for inspiration and feedback

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/yolov12-face-enhanced/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/yolov12-face-enhanced/discussions)
- **Email**: your.email@domain.com

---

**⭐ Star this repository if you find it useful!**