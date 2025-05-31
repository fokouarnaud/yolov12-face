# 🚀 YOLOv12-Face Enhanced

Fork d'Ultralytics pour la détection de visages avec modules d'attention Enhanced.

## ⚡ Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Entraînement
Utiliser le notebook : **`train_yolov12_enhanced_fixed.ipynb`**

## 🧠 Modules Enhanced

- **A2Module** : Area Attention Module
- **RELAN** : Residual Efficient Layer Aggregation Network

## 📁 Fichiers importants

- `ultralytics/nn/modules/enhanced.py` - Modules Enhanced
- `ultralytics/cfg/models/v12/yolov12-face-enhanced.yaml` - Config modèle
- `requirements.txt` - Dépendances (sans ultralytics)
- `train_yolov12_enhanced_fixed.ipynb` - **Notebook principal**

## ⚠️ Important

**Ne PAS installer `ultralytics` via pip** - utiliser le fork local.

Le notebook configure automatiquement le path Python pour utiliser le fork local.

## 🎯 Prêt pour l'entraînement

Tout est configuré ! Lance directement le notebook `train_yolov12_enhanced_fixed.ipynb` et exécute les cellules dans l'ordre pour détecter et corriger d'éventuelles erreurs.
