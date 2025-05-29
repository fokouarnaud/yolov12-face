# 🚀 Guide Rapide YOLOv12-Face

## Installation et Préparation

### 1. Cloner et installer
```bash
# Vous êtes déjà dans le bon répertoire
cd C:\Users\cedric\Desktop\box\01-Projects\Face-Recognition\yolov12-face

# Installer les dépendances
pip install -r requirements.txt
pip install gdown opencv-python
```

### 2. Télécharger et préparer le dataset WIDERFace
```bash
# Windows
python scripts\prepare_widerface.py

# Linux/Mac
python scripts/prepare_widerface.py

# Si problème de téléchargement, utiliser Google Drive
python scripts/prepare_widerface.py --gdrive
```

## Entraînement

### Option 1 : Commande YOLO standard
```bash
# Entraîner YOLOv12n-face
yolo detect train data=ultralytics/cfg/datasets/widerface.yaml model=ultralytics/cfg/models/v12/yolov12-face.yaml epochs=100 imgsz=640

# Entraîner YOLOv12s-face avec plus de batch
yolo detect train data=ultralytics/cfg/datasets/widerface.yaml model=ultralytics/cfg/models/v12/yolov12-face.yaml epochs=100 imgsz=640 batch=32 name=yolov12s-face
```

### Option 2 : Script Python personnalisé
```bash
# Entraînement basique
python scripts/train_yolov12_face.py --model yolov12-face.yaml --epochs 100

# Entraînement avec paramètres personnalisés
python scripts/train_yolov12_face.py --model yolov12-face.yaml --epochs 300 --batch-size 16 --img-size 640 --patience 50

# Reprendre l'entraînement
python scripts/train_yolov12_face.py --resume --name yolov12s-face
```

## Validation

```bash
# Valider le modèle
yolo detect val model=runs/detect/train/weights/best.pt data=ultralytics/cfg/datasets/widerface.yaml

# Ou avec le script
python scripts/train_yolov12_face.py --mode val --weights runs/detect/train/weights/best.pt
```

## Export

```bash
# Export ONNX
yolo export model=runs/detect/train/weights/best.pt format=onnx

# Export multiple formats
python scripts/train_yolov12_face.py --mode export --weights runs/detect/train/weights/best.pt --formats onnx torchscript tflite
```

## Inférence

```bash
# Test sur une image
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/image.jpg

# Test sur une vidéo
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/video.mp4

# Test avec webcam
yolo detect predict model=runs/detect/train/weights/best.pt source=0
```

## Structure des Fichiers

```
yolov12-face/
├── scripts/
│   ├── prepare_widerface.py      # Script de préparation du dataset
│   ├── train_yolov12_face.py     # Script d'entraînement
│   ├── get_widerface.sh          # Script bash (Linux/Mac)
│   └── get_widerface.bat         # Script batch (Windows)
├── ultralytics/
│   └── cfg/
│       ├── datasets/
│       │   └── widerface.yaml    # Configuration du dataset
│       └── models/
│           └── v12/
│               └── yolov12-face.yaml  # Configuration du modèle
├── datasets/
│   └── widerface/                # Dataset (créé après préparation)
│       ├── images/
│       ├── labels/
│       └── data.yaml
└── runs/
    └── detect/                   # Résultats d'entraînement
        └── train/
            └── weights/
                ├── best.pt       # Meilleurs poids
                └── last.pt       # Derniers poids
```

## Résultats Attendus

- **mAP@0.5** : ~91-93% sur WIDERFace validation
- **Vitesse** : ~150 FPS sur GPU RTX 3080 (YOLOv12n)
- **Taille** : ~6MB (YOLOv12n)

## Troubleshooting

### Erreur de mémoire GPU
```bash
# Réduire le batch size
python scripts/train_yolov12_face.py --batch-size 8

# Utiliser l'accumulation de gradient
yolo detect train ... batch=4 accumulate=4
```

### Dataset non trouvé
```bash
# Vérifier que le dataset est bien préparé
python scripts/prepare_widerface.py --output datasets/widerface

# Vérifier le chemin dans le YAML
cat ultralytics/cfg/datasets/widerface.yaml
```

### Erreur de téléchargement
```bash
# Utiliser Google Drive
python scripts/prepare_widerface.py --gdrive

# Ou télécharger manuellement depuis
# http://shuoyang1213.me/WIDERFACE/
```

## Tips d'Optimisation

1. **Augmentations** : Activer mosaic et mixup améliore la robustesse
2. **Learning Rate** : Commencer avec lr0=0.01 et utiliser cosine annealing
3. **Early Stopping** : Utiliser patience=50 pour éviter l'overfitting
4. **Multi-Scale** : Entraîner avec différentes tailles (320, 416, 640)

## Commandes Utiles

```bash
# Visualiser les résultats avec TensorBoard
tensorboard --logdir runs/detect

# Benchmark de vitesse
yolo benchmark model=runs/detect/train/weights/best.pt imgsz=640

# Créer une vidéo de démonstration
yolo detect predict model=runs/detect/train/weights/best.pt source=video.mp4 save=True
```

## Contact et Support

- **Issues** : Créer une issue sur GitHub
- **Documentation** : https://docs.ultralytics.com/
- **WIDERFace** : http://shuoyang1213.me/WIDERFACE/

---

🎯 **Commande rapide pour démarrer** :
```bash
# Tout-en-un : préparer dataset et entraîner
python scripts/prepare_widerface.py && python scripts/train_yolov12_face.py --epochs 100
```
