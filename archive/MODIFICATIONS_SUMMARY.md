# 📋 Résumé des Modifications - YOLOv12-Face

## ✅ Modifications Apportées au Repository

### 1. **Scripts de Préparation du Dataset** (`scripts/`)
- ✅ `prepare_widerface.py` : Script Python robuste pour télécharger et convertir WIDERFace
  - Téléchargement depuis HuggingFace ou Google Drive
  - Conversion automatique au format YOLO
  - Vérification de l'intégrité
  - Création du fichier `data.yaml`
- ✅ `get_widerface.sh` : Script bash pour Linux/Mac
- ✅ `get_widerface.bat` : Script batch pour Windows
- ✅ `train_yolov12_face.py` : Script d'entraînement utilisant l'API Ultralytics
- ✅ `README.md` : Documentation des scripts

### 2. **Configurations Ultralytics** (`ultralytics/cfg/`)
- ✅ `models/v12/yolov12-face.yaml` : Configuration du modèle optimisée pour les visages
- ✅ `datasets/widerface.yaml` : Configuration du dataset WIDERFace

### 3. **Documentation**
- ✅ `QUICKSTART.md` : Guide de démarrage rapide
- ✅ `.gitignore` : Mis à jour pour ignorer datasets et fichiers volumineux

## 🚀 Workflow d'Utilisation

### Étape 1 : Préparer le Dataset
```bash
# Depuis le répertoire yolov12-face
python scripts/prepare_widerface.py
```

### Étape 2 : Entraîner
```bash
# Option 1 : Commande YOLO standard
yolo detect train data=ultralytics/cfg/datasets/widerface.yaml model=ultralytics/cfg/models/v12/yolov12-face.yaml epochs=100

# Option 2 : Script Python
python scripts/train_yolov12_face.py --model yolov12-face.yaml --epochs 100
```

### Étape 3 : Valider et Exporter
```bash
# Validation
yolo detect val model=runs/detect/train/weights/best.pt data=ultralytics/cfg/datasets/widerface.yaml

# Export ONNX
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

## 📁 Structure Finale
```
yolov12-face/
├── scripts/                      # 🆕 Scripts de préparation et entraînement
│   ├── prepare_widerface.py
│   ├── train_yolov12_face.py
│   ├── get_widerface.sh
│   ├── get_widerface.bat
│   └── README.md
├── ultralytics/
│   └── cfg/
│       ├── datasets/
│       │   └── widerface.yaml    # 🆕 Configuration WIDERFace
│       └── models/
│           └── v12/
│               └── yolov12-face.yaml  # 🆕 Modèle optimisé visages
├── datasets/                     # 📁 Créé après prepare_widerface.py
│   └── widerface/
│       ├── images/
│       ├── labels/
│       └── data.yaml
├── QUICKSTART.md                 # 🆕 Guide rapide
└── .gitignore                    # 🔄 Mis à jour
```

## 🎯 Points Clés

1. **Compatibilité Totale** : Utilise l'infrastructure Ultralytics existante
2. **Robustesse** : Gestion des erreurs de téléchargement avec alternatives
3. **Simplicité** : Un seul script pour tout préparer
4. **Flexibilité** : Fonctionne sur Windows, Linux et Mac
5. **Documentation** : Guides clairs et exemples

## 💡 Prochaines Étapes

1. **Tester le téléchargement** :
   ```bash
   python scripts/prepare_widerface.py
   ```

2. **Lancer un entraînement test** :
   ```bash
   python scripts/train_yolov12_face.py --epochs 10 --batch-size 4
   ```

3. **Commit et Push** :
   ```bash
   git add .
   git commit -m "Add WIDERFace dataset preparation and training scripts"
   git push origin main
   ```

## ⚠️ Notes Importantes

- Le dataset WIDERFace fait ~1.5GB, assurez-vous d'avoir assez d'espace
- L'entraînement complet peut prendre plusieurs heures/jours
- Utilisez `--gdrive` si le téléchargement direct échoue
- Les fichiers du dataset sont dans `.gitignore` pour éviter de les pusher

Cette approche garantit que vous pouvez travailler entièrement depuis votre repo local sans dépendre d'autres sources, tout en restant compatible avec l'écosystème Ultralytics.
