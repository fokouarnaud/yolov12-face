# 📁 Scripts YOLOv12-Face

Ce dossier contient tous les scripts nécessaires pour entraîner, évaluer et déployer le modèle YOLOv12-Face.

## 📋 Vue d'ensemble des scripts

### 1. **prepare_widerface.py** 
Prépare le dataset WIDERFace pour l'entraînement avec YOLO.
```bash
python prepare_widerface.py --source path/to/widerface --output datasets/widerface
```

### 2. **train_yolov12_face.py**
Script d'entraînement de base pour YOLOv12-Face.
```bash
python train_yolov12_face.py --data ../datasets/widerface/data.yaml --epochs 100
```

### 3. **train_enhanced.py** ⭐
Script d'entraînement pour le modèle Enhanced avec modules d'attention.
```bash
# Entraînement simple du modèle enhanced
python train_enhanced.py --data ../datasets/widerface/data.yaml --epochs 100

# Comparaison avec le modèle de base
python train_enhanced.py --compare --epochs 50
```

### 4. **compare_performance.py** 📊
Compare les performances entre deux modèles (baseline vs enhanced).
```bash
python compare_performance.py \
    --baseline runs/face/baseline/weights/best.pt \
    --enhanced runs/face/enhanced/weights/best.pt \
    --test-images test_images/ \
    --save-images
```

### 5. **webcam_demo.py** 📷
Démonstration en temps réel avec une webcam.
```bash
# Utilisation basique
python webcam_demo.py --model runs/face/enhanced/weights/best.pt

# Avec toutes les options
python webcam_demo.py \
    --model runs/face/enhanced/weights/best.pt \
    --show-fps \
    --show-info \
    --save-video output.mp4
```

### 6. **mobile_optimization.py** 📱
Optimise le modèle pour le déploiement mobile.
```bash
# Export dans tous les formats
python mobile_optimization.py \
    --model runs/face/enhanced/weights/best.pt \
    --formats onnx tflite coreml ncnn \
    --quantize \
    --test-images test_images/
```

## 🚀 Workflow recommandé

### 1. **Préparation des données**
```bash
# Télécharger WIDERFace
./get_widerface.sh

# Préparer le dataset
python prepare_widerface.py --source ../datasets/widerface --output ../datasets/widerface
```

### 2. **Entraînement du modèle Enhanced**
```bash
# Entraînement complet avec comparaison
python train_enhanced.py --compare --epochs 100 --batch-size 16

# Ou entraînement direct du modèle enhanced
python train_enhanced.py --epochs 100 --batch-size 16
```

### 3. **Évaluation et comparaison**
```bash
# Comparer les performances
python compare_performance.py \
    --baseline ../runs/face/yolov12-face-enhanced_baseline/weights/best.pt \
    --enhanced ../runs/face/yolov12-face-enhanced_enhanced/weights/best.pt \
    --test-images ../test_images/ \
    --save-images
```

### 4. **Test en temps réel**
```bash
# Tester avec une webcam
python webcam_demo.py --model ../runs/face/yolov12-face-enhanced_enhanced/weights/best.pt --show-fps
```

### 5. **Optimisation mobile**
```bash
# Exporter pour mobile
python mobile_optimization.py \
    --model ../runs/face/yolov12-face-enhanced_enhanced/weights/best.pt \
    --formats tflite coreml \
    --quantize
```

## 📊 Résultats attendus

### Modèle de base
- mAP@0.5: ~66%
- Precision: ~77%
- Recall: ~60%
- Latence: ~1.4ms

### Modèle Enhanced
- mAP@0.5: ~70-75% (attendu)
- Precision: ~80-85% (attendu)
- Recall: ~65-70% (attendu)
- Latence: ~2-3ms (avec modules d'attention)

## 🛠️ Dépendances

```bash
# Installation des dépendances
pip install ultralytics opencv-python matplotlib seaborn tqdm

# Pour l'export mobile
pip install onnx onnxsim coremltools
```

## 📝 Notes importantes

1. **GPU recommandé** : Les entraînements sont optimisés pour GPU NVIDIA avec CUDA
2. **Mémoire** : Le modèle Enhanced nécessite plus de VRAM (~8GB minimum)
3. **Dataset** : WIDERFace doit être téléchargé et préparé avant l'entraînement
4. **Modules Enhanced** : Assurez-vous que `enhanced.py` est dans `ultralytics/nn/modules/`

## 🐛 Dépannage

### Erreur "Module Enhanced non trouvé"
- Vérifiez que `enhanced.py` existe dans `../ultralytics/nn/modules/`
- Le script `train_enhanced.py` tentera d'ajouter automatiquement les imports

### Erreur CUDA/GPU
- Utilisez `--device cpu` pour entraîner sur CPU (plus lent)
- Vérifiez votre installation CUDA avec `nvidia-smi`

### Erreur de mémoire
- Réduisez le batch size: `--batch-size 8` ou `--batch-size 4`
- Réduisez la taille d'image: `--imgsz 320`

## 📚 Documentation détaillée

Pour plus d'informations sur chaque script, utilisez l'option `--help`:
```bash
python script_name.py --help
```

## 🤝 Contribution

Pour ajouter de nouveaux scripts ou améliorer les existants:
1. Suivez la structure existante
2. Ajoutez une documentation claire
3. Incluez des exemples d'utilisation
4. Testez sur différentes configurations

## 📧 Support

Pour toute question ou problème:
- Consultez d'abord ce README et les messages d'erreur
- Vérifiez que toutes les dépendances sont installées
- Assurez-vous que les chemins de fichiers sont corrects
