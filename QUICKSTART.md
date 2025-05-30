# 🚀 Guide Rapide YOLOv12-Face Enhanced

## Installation et Préparation

### 1. Cloner et installer
```bash
# Vous êtes déjà dans le bon répertoire
cd C:\Users\cedric\Desktop\box\01-Projects\Face-Recognition\yolov12-face

# Installer les dépendances
pip install -r requirements.txt
pip install gdown opencv-python matplotlib seaborn

# Pour l'export mobile
pip install onnx onnxsim coremltools
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

## 🚀 Entraînement du Modèle Enhanced

### Option 1 : Entraînement Enhanced avec comparaison automatique
```bash
# Compare automatiquement baseline vs enhanced
python scripts/train_enhanced.py --compare --epochs 100 --batch-size 16

# Avec GPU spécifique
python scripts/train_enhanced.py --compare --epochs 100 --device 0
```

### Option 2 : Entraînement Enhanced uniquement
```bash
# Entraîner directement le modèle enhanced
python scripts/train_enhanced.py --epochs 100 --batch-size 16

# Reprendre un entraînement
python scripts/train_enhanced.py --epochs 100 --resume
```

### Option 3 : Entraînement standard (baseline)
```bash
# Entraîner YOLOv12n-face standard
yolo detect train data=ultralytics/cfg/datasets/widerface.yaml model=ultralytics/cfg/models/v12/yolov12-face.yaml epochs=100 imgsz=640
```

## 📊 Évaluation et Comparaison

### Comparer les performances
```bash
# Comparaison détaillée avec graphiques
python scripts/compare_performance.py \
    --baseline runs/face/yolov12-face-enhanced_baseline/weights/best.pt \
    --enhanced runs/face/yolov12-face-enhanced_enhanced/weights/best.pt \
    --test-images test_images/ \
    --save-images
```

### Validation standard
```bash
# Valider le modèle enhanced
yolo detect val model=runs/face/yolov12-face-enhanced_enhanced/weights/best.pt data=ultralytics/cfg/datasets/widerface.yaml
```

## 📷 Démonstration en Temps Réel

### Test avec webcam
```bash
# Démo basique
python scripts/webcam_demo.py --model runs/face/yolov12-face-enhanced_enhanced/weights/best.pt

# Avec toutes les options
python scripts/webcam_demo.py \
    --model runs/face/yolov12-face-enhanced_enhanced/weights/best.pt \
    --show-fps \
    --show-info \
    --save-video demo_output.mp4 \
    --conf 0.5
```

## 📱 Optimisation Mobile

### Export multi-plateformes
```bash
# Export complet avec quantification
python scripts/mobile_optimization.py \
    --model runs/face/yolov12-face-enhanced_enhanced/weights/best.pt \
    --formats onnx tflite coreml ncnn \
    --quantize \
    --half \
    --test-images test_images/

# Export pour iOS uniquement
python scripts/mobile_optimization.py \
    --model runs/face/yolov12-face-enhanced_enhanced/weights/best.pt \
    --formats coreml \
    --imgsz 320
```

## 🏗️ Architecture Enhanced

Le modèle Enhanced inclut plusieurs modules d'attention de pointe :

1. **A2Module** : Area Attention pour focus sur les régions importantes
2. **RELAN** : Residual Efficient Layer Aggregation Network
3. **FlashAttention** : Attention optimisée pour GPU modernes
4. **CrossScaleAttention** : Attention multi-échelle
5. **MicroExpressionAttention** : Spécialisé pour les micro-expressions

## 📊 Résultats Attendus

### Modèle Baseline (YOLOv12-Face)
- **mAP@0.5** : ~66%
- **Precision** : ~77.4%
- **Recall** : ~60.2%
- **Latence** : ~1.4ms (RTX 3080)

### Modèle Enhanced (YOLOv12-Face Enhanced)
- **mAP@0.5** : ~70-75% (+4-9%)
- **Precision** : ~80-85% (+3-8%)
- **Recall** : ~65-70% (+5-10%)
- **Latence** : ~2-3ms (avec modules d'attention)

## 🛠️ Structure du Projet

```
yolov12-face/
├── scripts/
│   ├── prepare_widerface.py          # Préparation dataset
│   ├── train_yolov12_face.py         # Entraînement standard
│   ├── train_enhanced.py             # Entraînement Enhanced ⭐
│   ├── compare_performance.py        # Comparaison modèles
│   ├── webcam_demo.py                # Démo temps réel
│   ├── mobile_optimization.py        # Export mobile
│   └── README.md                     # Documentation scripts
├── ultralytics/
│   ├── cfg/
│   │   ├── datasets/
│   │   │   └── widerface.yaml        # Config dataset
│   │   └── models/
│   │       └── v12/
│   │           ├── yolov12-face.yaml          # Modèle standard
│   │           └── yolov12-face-enhanced.yaml # Modèle Enhanced ⭐
│   └── nn/
│       └── modules/
│           └── enhanced.py            # Modules d'attention ⭐
├── datasets/
│   └── widerface/                    # Dataset WIDERFace
├── runs/                             # Résultats d'entraînement
├── results/                          # Résultats de comparaison
├── comparison_results/               # Analyses détaillées
└── mobile_models/                    # Modèles optimisés
```

## 💡 Workflow Recommandé

### 1. Démarrage rapide
```bash
# Préparer dataset et entraîner modèle enhanced avec comparaison
python scripts/prepare_widerface.py && python scripts/train_enhanced.py --compare --epochs 100
```

### 2. Workflow complet
```bash
# 1. Préparer les données
python scripts/prepare_widerface.py

# 2. Entraîner et comparer
python scripts/train_enhanced.py --compare --epochs 100

# 3. Analyser les résultats
python scripts/compare_performance.py \
    --baseline runs/face/*/baseline/weights/best.pt \
    --enhanced runs/face/*/enhanced/weights/best.pt

# 4. Tester en temps réel
python scripts/webcam_demo.py --model runs/face/*/enhanced/weights/best.pt --show-fps

# 5. Optimiser pour mobile
python scripts/mobile_optimization.py --model runs/face/*/enhanced/weights/best.pt --quantize
```

## 🐛 Troubleshooting

### Erreur "Modules Enhanced non trouvés"
```bash
# Vérifier que enhanced.py existe
dir ultralytics\nn\modules\enhanced.py

# Le script train_enhanced.py tente de configurer automatiquement
python scripts/train_enhanced.py --check-modules
```

### Erreur de mémoire GPU
```bash
# Réduire batch size
python scripts/train_enhanced.py --batch-size 8

# Utiliser gradient accumulation
python scripts/train_enhanced.py --batch-size 4 --accumulate 4

# Désactiver certains modules d'attention
# (modifier yolov12-face-enhanced.yaml)
```

### FlashAttention non supporté
```bash
# FlashAttention nécessite GPU Ampere+ (RTX 30xx, A100, etc.)
# Le modèle fonctionnera sans, mais plus lentement
```

## 📈 Tips d'Optimisation

1. **Multi-Scale Training** : Ajouter `--imgsz 320 416 640` pour robustesse
2. **Mixed Precision** : Utiliser `--amp` pour entraînement plus rapide
3. **Augmentations** : Activer mosaic, mixup, copy-paste
4. **Learning Rate** : Ajuster avec `--lr0 0.01 --lrf 0.01`
5. **Warmup** : Utiliser `--warmup-epochs 5` pour stabilité

## 🔧 Commandes Utiles

```bash
# Visualiser avec TensorBoard
tensorboard --logdir runs/face

# Benchmark de vitesse
python scripts/compare_performance.py --benchmark-only

# Créer une vidéo de démonstration
python scripts/webcam_demo.py --model best.pt --source video.mp4 --save-video output.mp4

# Export pour production
yolo export model=best.pt format=onnx opset=17 simplify=True
```

## 📱 Intégration Mobile

Après optimisation, consultez `mobile_models/MOBILE_INTEGRATION_GUIDE.md` pour :
- Intégration Android (TensorFlow Lite)
- Intégration iOS (Core ML)
- Flutter/React Native
- Benchmarks par plateforme

## 🚀 Performances Attendues

| Plateforme | Modèle | Résolution | FPS |
|------------|---------|------------|-----|
| RTX 3080 | Enhanced | 640x640 | 330 |
| RTX 3080 | Enhanced | 320x320 | 500+ |
| iPhone 14 Pro | CoreML | 320x320 | 120 |
| Pixel 7 | TFLite | 320x320 | 60 |

---

🎯 **Commande rapide pour tout tester** :
```bash
# Entraîner, comparer, et tester en une commande
python scripts/train_enhanced.py --compare --epochs 50 && python scripts/webcam_demo.py --model runs/face/*/enhanced/weights/best.pt --show-fps
```
