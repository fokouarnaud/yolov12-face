# 📋 Guide de Restauration Manuelle des Configurations YOLOv12-Face Enhanced

## 🎯 Objectif

Ce guide explique comment restaurer manuellement les fichiers de configuration personnalisés après une réinstallation d'Ultralytics.

## 📁 Structure des Sauvegardes

Tous nos fichiers personnalisés sont sauvegardés dans `scripts/configs/` :

```
scripts/configs/
├── datasets/
│   └── widerface.yaml                    # Configuration dataset WIDERFace
├── models/v12/
│   ├── yolov12-face.yaml                # Modèle de base
│   └── yolov12-face-enhanced.yaml       # Modèle avec modules d'attention
└── modules/
    └── enhanced.py                       # Modules d'attention personnalisés
```

## 🔄 Procédure de Restauration Manuelle

### Étape 1 : Vérifier l'Installation d'Ultralytics

```bash
pip install ultralytics
```

### Étape 2 : Créer la Structure de Destination

```bash
# Créer les dossiers nécessaires dans ultralytics
mkdir -p ultralytics/cfg/datasets
mkdir -p ultralytics/cfg/models/v12
mkdir -p ultralytics/nn/modules
```

### Étape 3 : Copier les Fichiers de Configuration

#### 📊 Dataset Configuration
```bash
cp scripts/configs/datasets/widerface.yaml ultralytics/cfg/datasets/widerface.yaml
```

#### 🤖 Model Configurations
```bash
cp scripts/configs/models/v12/yolov12-face.yaml ultralytics/cfg/models/v12/yolov12-face.yaml
cp scripts/configs/models/v12/yolov12-face-enhanced.yaml ultralytics/cfg/models/v12/yolov12-face-enhanced.yaml
```

#### 🧠 Enhanced Modules
```bash
cp scripts/configs/modules/enhanced.py ultralytics/nn/modules/enhanced.py
```

### Étape 4 : Mettre à Jour le Fichier __init__.py

Ajouter l'import des modules enhanced dans `ultralytics/nn/modules/__init__.py` :

1. Ouvrir le fichier : `ultralytics/nn/modules/__init__.py`
2. Localiser la ligne qui commence par `__all__`
3. Ajouter cette ligne **AVANT** la ligne `__all__` :

```python
from .enhanced import *
```

**Exemple de contenu final :**
```python
# ... autres imports ...
from .enhanced import *
__all__ = (
    "Conv",
    "DWConv",
    # ... autres exports ...
)
```

## 🪟 Commandes Windows (PowerShell/CMD)

```cmd
REM Copier les configurations
copy "scripts\configs\datasets\widerface.yaml" "ultralytics\cfg\datasets\widerface.yaml"
copy "scripts\configs\models\v12\yolov12-face.yaml" "ultralytics\cfg\models\v12\yolov12-face.yaml"
copy "scripts\configs\models\v12\yolov12-face-enhanced.yaml" "ultralytics\cfg\models\v12\yolov12-face-enhanced.yaml"
copy "scripts\configs\modules\enhanced.py" "ultralytics\nn\modules\enhanced.py"
```

## 🐧 Commandes Linux/macOS

```bash
# Copier les configurations
cp scripts/configs/datasets/widerface.yaml ultralytics/cfg/datasets/widerface.yaml
cp scripts/configs/models/v12/yolov12-face.yaml ultralytics/cfg/models/v12/yolov12-face.yaml
cp scripts/configs/models/v12/yolov12-face-enhanced.yaml ultralytics/cfg/models/v12/yolov12-face-enhanced.yaml
cp scripts/configs/modules/enhanced.py ultralytics/nn/modules/enhanced.py
```

## ✅ Vérification de la Restauration

Après la restauration, testez que tout fonctionne :

```python
# Test 1: Import des modules
from ultralytics.nn.modules.enhanced import A2Module, RELAN
print("✅ Modules enhanced importés")

# Test 2: Création d'un modèle
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v12/yolov12-face-enhanced.yaml')
print("✅ Modèle enhanced créé")
```

## 🤖 Script de Restauration Automatique

Pour automatiser le processus, vous pouvez utiliser ce script Python :

```python
import shutil
from pathlib import Path

def restore_configs():
    """Restaure automatiquement les configurations"""
    
    files_map = [
        ('scripts/configs/datasets/widerface.yaml', 'ultralytics/cfg/datasets/widerface.yaml'),
        ('scripts/configs/models/v12/yolov12-face.yaml', 'ultralytics/cfg/models/v12/yolov12-face.yaml'),
        ('scripts/configs/models/v12/yolov12-face-enhanced.yaml', 'ultralytics/cfg/models/v12/yolov12-face-enhanced.yaml'),
        ('scripts/configs/modules/enhanced.py', 'ultralytics/nn/modules/enhanced.py')
    ]
    
    for src, dst in files_map:
        src_path = Path(src)
        dst_path = Path(dst)
        
        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            print(f"🔄 Restauré: {dst}")
    
    # Mettre à jour __init__.py
    init_file = Path('ultralytics/nn/modules/__init__.py')
    if init_file.exists():
        with open(init_file, 'r') as f:
            content = f.read()
        
        if 'from .enhanced import *' not in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('__all__'):
                    lines.insert(i, 'from .enhanced import *')
                    break
            
            with open(init_file, 'w') as f:
                f.write('\n'.join(lines))
            print("✅ __init__.py mis à jour")

# Exécuter la restauration
restore_configs()
```

## 🚨 Points d'Attention

1. **Ordre d'installation** : Toujours installer Ultralytics AVANT de restaurer les fichiers
2. **Permissions** : Assurez-vous d'avoir les droits d'écriture dans le dossier ultralytics
3. **Versions** : Ces configurations sont compatibles avec Ultralytics 8.0+
4. **Sauvegarde** : Toujours garder une copie de sauvegarde des fichiers dans `scripts/configs/`

## 🔧 Troubleshooting

### Erreur "Module enhanced not found"
- Vérifiez que `enhanced.py` est bien dans `ultralytics/nn/modules/`
- Vérifiez que l'import est ajouté dans `__init__.py`

### Erreur "Model config not found"
- Vérifiez que les fichiers YAML sont dans `ultralytics/cfg/models/v12/`
- Vérifiez les chemins dans votre code

### Erreur "Dataset not found"
- Vérifiez que `widerface.yaml` est dans `ultralytics/cfg/datasets/`
- Vérifiez que le dataset WIDERFace est téléchargé

## 🎯 Utilisation Après Restauration

Une fois la restauration terminée, vous pouvez utiliser :

```python
from ultralytics import YOLO

# Modèle Enhanced
model = YOLO('ultralytics/cfg/models/v12/yolov12-face-enhanced.yaml')

# Entraînement
model.train(
    data='ultralytics/cfg/datasets/widerface.yaml',
    epochs=100,
    batch=16
)
```

**✅ La restauration est terminée ! Le projet YOLOv12-Face Enhanced est prêt à l'emploi.**