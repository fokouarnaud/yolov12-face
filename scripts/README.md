# Scripts YOLOv12-Face

Ce dossier contient les scripts pour préparer le dataset WIDERFace et entraîner YOLOv12 pour la détection de visages.

## 📋 Scripts disponibles

### 1. `prepare_widerface.py`
Script Python principal pour télécharger et préparer le dataset WIDERFace.

**Utilisation :**
```bash
python scripts/prepare_widerface.py --output datasets/widerface
```

**Options :**
- `--output` : Répertoire de sortie (défaut: `datasets/widerface`)
- `--gdrive` : Utiliser Google Drive pour le téléchargement

**Fonctionnalités :**
- ✅ Téléchargement automatique depuis HuggingFace ou Google Drive
- ✅ Extraction des archives ZIP
- ✅ Conversion des annotations au format YOLO
- ✅ Création du fichier `data.yaml` pour Ultralytics
- ✅ Vérification de l'intégrité du dataset

### 2. `get_widerface.sh` (Linux/Mac)
Script bash pour télécharger le dataset sur Linux/Mac.

```bash
chmod +x scripts/get_widerface.sh
./scripts/get_widerface.sh [output_dir] [use_gdrive]
```

### 3. `get_widerface.bat` (Windows)
Script batch pour télécharger le dataset sur Windows.

```cmd
scripts\get_widerface.bat [output_dir]
```

### 4. `train_yolov12_face.py`
Script d'entraînement utilisant l'API Ultralytics.

**Utilisation basique :**
```bash
# Entraîner YOLOv12n
python scripts/train_yolov12_face.py --model yolov12n.yaml --epochs 100

# Entraîner YOLOv12s avec batch size personnalisé
python scripts/train_yolov12_face.py --model yolov12s.yaml --batch-size 32

# Reprendre l'entraînement
python scripts/train_yolov12_face.py --resume --name mon_experience
```

**Options principales :**
- `--model` : Configuration du modèle (yolov12n/s/m/l/x.yaml)
- `--data` : Fichier de données (défaut: datasets/widerface/data.yaml)
- `--epochs` : Nombre d'epochs (défaut: 300)
- `--batch-size` : Taille du batch (défaut: 16)
- `--img-size` : Taille des images (défaut: 640)
- `--device` : Device cuda/cpu (défaut: auto)
- `--resume` : Reprendre l'entraînement

**Modes :**
```bash
# Entraînement
python scripts/train_yolov12_face.py --mode train

# Validation
python scripts/train_yolov12_face.py --mode val --weights runs/train/exp/weights/best.pt

# Export
python scripts/train_yolov12_face.py --mode export --weights runs/train/exp/weights/best.pt --formats onnx torchscript
```

## 🚀 Workflow complet

### 1. Préparer le dataset
```bash
# Télécharger et préparer WIDERFace
python scripts/prepare_widerface.py
```

### 2. Entraîner le modèle
```bash
# Entraîner YOLOv12n pour 100 epochs
python scripts/train_yolov12_face.py --model yolov12n.yaml --epochs 100
```

### 3. Valider le modèle
```bash
# Valider sur le dataset de validation
python scripts/train_yolov12_face.py --mode val --weights runs/train/exp/weights/best.pt
```

### 4. Exporter le modèle
```bash
# Exporter en ONNX et TorchScript
python scripts/train_yolov12_face.py --mode export --weights runs/train/exp/weights/best.pt
```

## 📊 Structure du dataset

Après préparation, le dataset aura cette structure :
```
datasets/widerface/
├── images/
│   ├── train/      # Images d'entraînement
│   ├── val/        # Images de validation
│   └── test/       # Images de test
├── labels/
│   ├── train/      # Labels YOLO (format txt)
│   ├── val/        # Labels YOLO
│   └── test/       # Labels YOLO
└── data.yaml       # Configuration du dataset
```

## 🔧 Dépannage

### Erreur de téléchargement
- Essayez l'option `--gdrive` pour utiliser Google Drive
- Vérifiez votre connexion Internet
- Installez `gdown` : `pip install gdown`

### Erreur de mémoire GPU
- Réduisez le batch size : `--batch-size 8`
- Réduisez la taille des images : `--img-size 320`
- Utilisez le CPU : `--device cpu`

### Dataset non trouvé
- Vérifiez que le dataset est dans `datasets/widerface/`
- Vérifiez le chemin dans `data.yaml`
- Relancez `prepare_widerface.py`

## 📝 Notes

- Le dataset WIDERFace contient environ 32k images
- L'entraînement complet peut prendre plusieurs heures/jours selon le GPU
- Les meilleurs résultats sont obtenus avec 300+ epochs
- Utilisez TensorBoard pour suivre l'entraînement : `tensorboard --logdir runs/train`
