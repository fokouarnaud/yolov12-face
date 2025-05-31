# 🚀 YOLOv12-Face Enhanced

Fork d'Ultralytics pour la détection de visages avec modules d'attention Enhanced.

## 📋 Résumé du Projet

Ce projet est un **fork modifié d'Ultralytics** qui ajoute des modules d'attention avancés pour améliorer la détection de visages :

- **A2Module** : Area Attention Module pour l'attention spatiale et canal
- **RELAN** : Residual Efficient Layer Aggregation Network pour l'agrégation multi-échelle

## 🔧 Configuration Corrigée

### ✅ Problèmes Résolus
1. **Import des modules Enhanced** : Maintenant fonctionnel
2. **Requirements.txt** : Corrigé pour utiliser le fork local
3. **Notebook** : Version corrigée sans installation d'Ultralytics via pip

## 🚀 Installation et Usage

### 1. Validation du Setup
```bash
python validate_setup.py
```

### 2. Installation des Dépendances
```bash
pip install -r requirements.txt
```
> ⚠️ **Important** : Ne PAS installer `ultralytics` via pip !

### 3. Test Rapide
```bash
python quick_test.py
```

### 4. Entraînement
Utiliser le notebook corrigé : `train_yolov12_enhanced_fixed.ipynb`

## 📁 Structure du Projet

```
yolov12-face/
├── ultralytics/                    # Fork modifié d'Ultralytics
│   ├── nn/modules/
│   │   ├── enhanced.py            # ✅ Modules A2Module et RELAN
│   │   └── __init__.py            # ✅ Import des modules Enhanced
│   └── cfg/
│       ├── datasets/widerface.yaml
│       └── models/v12/
│           ├── yolov12-face.yaml
│           └── yolov12-face-enhanced.yaml
├── scripts/
├── requirements.txt               # ✅ Corrigé (sans ultralytics)
├── train_yolov12_enhanced_fixed.ipynb  # ✅ Notebook corrigé
├── validate_setup.py             # ✅ Script de validation
└── quick_test.py                  # ✅ Test rapide
```

## 🧠 Modules Enhanced

### A2Module (Area Attention Module)
- **Fonction** : Attention spatiale et canal
- **Usage** : `A2Module(in_channels, out_channels)`
- **Avantage** : Améliore la localisation des visages

### RELAN (Residual Efficient Layer Aggregation Network)
- **Fonction** : Agrégation multi-échelle avec connexions résiduelles
- **Usage** : `RELAN(in_channels, out_channels)`
- **Avantage** : Capture des caractéristiques à différentes échelles

## 📊 Entraînement

```python
from ultralytics import YOLO

# Charger le modèle Enhanced
model = YOLO('ultralytics/cfg/models/v12/yolov12-face-enhanced.yaml')

# Entraîner
results = model.train(
    data='ultralytics/cfg/datasets/widerface.yaml',
    epochs=100,
    batch=16,
    imgsz=640
)
```

## 🎯 Points Clés

### ✅ À Faire
- Utiliser le fork local dans `./ultralytics/`
- Ajouter `sys.path.insert(0, '.')` au début des scripts
- Utiliser `train_yolov12_enhanced_fixed.ipynb`

### ❌ À Éviter
- `pip install ultralytics` (écrase le fork)
- Utiliser l'ancien notebook avec les erreurs d'import
- Modifier les fichiers de base d'Ultralytics

## 🔬 Tests et Validation

### Test Rapide des Imports
```python
import sys
sys.path.insert(0, '.')

from ultralytics.nn.modules.enhanced import A2Module, RELAN
print("✅ Modules Enhanced importés avec succès!")
```

### Validation Complète
```bash
python validate_setup.py
```

## 📈 Performances Attendues

Avec les modules Enhanced, attendez-vous à :
- **Meilleure précision** sur les visages de petite taille
- **Amélioration du mAP** par rapport au modèle de base
- **Robustesse** accrue dans des conditions difficiles

## 🐛 Résolution des Problèmes

### Import Error
```
ModuleNotFoundError: No module named 'ultralytics.nn.modules.enhanced'
```
**Solution** : Exécuter `python validate_setup.py` et vérifier le path

### Syntax Error dans __init__.py
**Solution** : Vérifier que `from .enhanced import *` est correctement ajouté

### Conflit avec Ultralytics installé
**Solution** : `pip uninstall ultralytics` puis utiliser le fork local

## 🎉 Statut du Projet

- ✅ **Fork Ultralytics** : Opérationnel
- ✅ **Modules Enhanced** : Fonctionnels (A2Module, RELAN)
- ✅ **Configuration** : Corrigée
- ✅ **Notebook** : Version corrigée disponible
- ✅ **Validation** : Scripts de test inclus

**Le projet est prêt pour l'entraînement !** 🚀

---

*Pour plus de détails, voir `CORRECTION_GUIDE.md`*
