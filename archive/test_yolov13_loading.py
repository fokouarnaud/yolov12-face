"""
Test script pour vérifier le chargement du modèle YOLOv13-Face
"""

import sys
import os

# Ajouter le répertoire du projet au path
project_dir = r"C:\Users\cedric\Desktop\box\01-Projects\Face-Recognition\yolov12-face"
sys.path.insert(0, project_dir)

# Importer YOLO
from ultralytics import YOLO

# Test de chargement du modèle
print("Test de chargement YOLOv13n-Face...")
try:
    model_v13 = YOLO('ultralytics/cfg/models/v13/yolov13n-face.yaml', task='detect')
    print("✅ Modèle YOLOv13n-Face chargé avec succès!")
    
    # Afficher les informations du modèle
    print("\nInformations du modèle:")
    print(f"- Nombre de paramètres: {sum(p.numel() for p in model_v13.model.parameters()):,}")
    print(f"- Nombre de couches: {len(list(model_v13.model.modules()))}")
    
    # Test de prédiction sur une image factice
    import torch
    dummy_input = torch.randn(1, 3, 640, 640)
    print("\nTest de forward pass...")
    
    model_v13.model.eval()
    with torch.no_grad():
        output = model_v13.model(dummy_input)
    
    print(f"✅ Forward pass réussi!")
    if isinstance(output, list):
        print(f"- Nombre de sorties: {len(output)}")
        for i, out in enumerate(output):
            print(f"  - Sortie {i}: {out.shape}")
    else:
        print(f"- Forme de sortie: {output.shape}")
        
except Exception as e:
    print(f"❌ Erreur lors du chargement: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
