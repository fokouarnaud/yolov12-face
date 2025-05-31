# YOLOv12-Face Enhanced avec Modules d'Attention

Fork d'Ultralytics YOLOv12 avec modules d'attention Enhanced (A2Module et RELAN) pour la détection de visages.

## 🚀 État du Projet

### ✅ Modules Implémentés

- **A2Module** : Area Attention Module pour l'attention spatiale et par canal
- **RELAN** : Residual Efficient Layer Aggregation Network pour l'agrégation multi-échelle

### ✅ Corrections Appliquées

1. **Import des modules Enhanced dans `tasks.py`** :
   - Ajout de `from ultralytics.nn.modules.enhanced import A2Module, RELAN`
   - Ajout aux `globals()` pour le parsing YAML
   - Ajout dans les listes de modules reconnus

2. **Configuration YAML** :
   - `yolov12-face-enhanced.yaml` avec structure cohérente
   - Intégration de A2Module et RELAN dans le backbone

3. **Dependencies** :
   - Ajout de `timm>=0.9.0` dans requirements.txt

## 📁 Structure du Projet

```
yolov12-face/
├── ultralytics/                    # Fork modifié d'Ultralytics
│   ├── nn/
│   │   ├── tasks.py               # Modifié pour reconnaître A2Module et RELAN
│   │   └── modules/
│   │       ├── __init__.py        # Import des modules Enhanced
│   │       └── enhanced.py        # Définition de A2Module et RELAN
│   └── cfg/
│       ├── datasets/
│       │   └── widerface.yaml
│       └── models/v12/
│           ├── yolov12-face.yaml
│           └── yolov12-face-enhanced.yaml  # Configuration Enhanced
├── requirements.txt                # Dependencies (avec timm)
├── test_enhanced_modules.py        # Script de test
├── test_enhanced_notebook.ipynb    # Notebook de test
└── README.md                       # Ce fichier
```

## 🔧 Installation

1. **Cloner le repository** :
```bash
git clone https://github.com/your-username/yolov12-face.git
cd yolov12-face
```

2. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

3. **NE PAS installer ultralytics** - Nous utilisons le fork local !

## 🚀 Utilisation

### Test des Modules

```bash
python test_enhanced_modules.py
```

### Entraînement du Modèle Enhanced

```python
import sys
sys.path.insert(0, '.')  # Ajouter le répertoire courant au PYTHONPATH

from ultralytics import YOLO

# Charger le modèle Enhanced
model = YOLO('ultralytics/cfg/models/v12/yolov12-face-enhanced.yaml')

# Entraîner
model.train(
    data='ultralytics/cfg/datasets/widerface.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda'
)
```

### Inference

```python
# Charger le modèle entraîné
model = YOLO('path/to/best.pt')

# Prédiction
results = model('path/to/image.jpg')
```

## 🔍 Modules Enhanced

### A2Module (Area Attention Module)

- **Attention par canal** : Utilise average pooling et max pooling
- **Attention spatiale** : Convolution 7x7 pour capturer les relations spatiales
- **Fusion** : Combine les deux types d'attention pour améliorer la détection

### RELAN (Residual Efficient Layer Aggregation Network)

- **Multi-échelle** : Convolutions 1x1, 3x3, 5x5, et 7x7
- **Agrégation** : Fusion des caractéristiques multi-échelle
- **Connexion résiduelle** : Préserve l'information originale

## 📊 Performance Attendue

Les modules Enhanced devraient améliorer :
- La détection des petits visages
- La robustesse aux occlusions
- La précision dans les scènes complexes

## 🐛 Résolution de Problèmes

### KeyError: 'A2Module'

Si vous rencontrez cette erreur, vérifiez que :
1. Le fichier `enhanced.py` existe dans `ultralytics/nn/modules/`
2. Les imports sont corrects dans `tasks.py`
3. Le PYTHONPATH inclut le répertoire du projet

### Import Error

Assurez-vous d'avoir ajouté le répertoire au PYTHONPATH :
```python
import sys
sys.path.insert(0, '.')
```

## 📝 Notes Importantes

1. **Fork Local** : Ce projet utilise un fork modifié d'Ultralytics. Ne pas installer ultralytics via pip !
2. **Modules Enhanced** : A2Module et RELAN sont des modules personnalisés pour améliorer la détection de visages
3. **Compatibilité** : Compatible avec PyTorch >= 2.0.0

## 🤝 Contribution

Les contributions sont bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 📄 Licence

Ce projet est basé sur Ultralytics YOLO (AGPL-3.0 License).
