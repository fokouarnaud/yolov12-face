"""Test script pour vérifier que les modules Enhanced sont bien reconnus - Version 2"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

# Test 1: Import direct des modules
print("Test 1: Import direct des modules Enhanced...")
try:
    from ultralytics.nn.modules.enhanced import A2Module, RELAN
    print("✅ Import direct réussi: A2Module et RELAN")
except Exception as e:
    print(f"❌ Erreur import direct: {e}")

# Test 2: Test des signatures des constructeurs
print("\nTest 2: Test des signatures des constructeurs...")
try:
    import torch
    
    # Test A2Module avec différents arguments
    print("Test A2Module avec arguments YOLO:")
    a2_module = A2Module(512, 512, 1)  # c1, c2, n
    x = torch.randn(1, 512, 32, 32)
    out = a2_module(x)
    print(f"✅ A2Module avec 3 args: {x.shape} -> {out.shape}")
    
    # Test RELAN avec différents arguments
    print("\nTest RELAN avec arguments YOLO:")
    relan = RELAN(512, 512, 1)  # c1, c2, n
    x = torch.randn(1, 512, 32, 32)
    out = relan(x)
    print(f"✅ RELAN avec 3 args: {x.shape} -> {out.shape}")
    
except Exception as e:
    print(f"❌ Erreur test constructeurs: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Charger le modèle Enhanced
print("\nTest 3: Chargement du modèle Enhanced...")
try:
    from ultralytics import YOLO
    
    # Spécifier le chemin complet du fichier de configuration
    config_path = Path(__file__).parent / "ultralytics" / "cfg" / "models" / "v12" / "yolov12-face-enhanced.yaml"
    print(f"Chemin de config: {config_path}")
    
    # Créer le modèle
    model = YOLO(str(config_path))
    print("✅ Modèle Enhanced chargé avec succès!")
    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # Afficher les modules Enhanced dans le modèle
    print("\nModules Enhanced dans le modèle:")
    for name, module in model.model.named_modules():
        if isinstance(module, (A2Module, RELAN)):
            print(f"  - {name}: {type(module).__name__}")
    
except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test forward pass
print("\nTest 4: Test forward pass...")
try:
    if 'model' in locals():
        x = torch.randn(1, 3, 640, 640)
        print(f"Input shape: {x.shape}")
        
        # Test predict (inference mode)
        with torch.no_grad():
            output = model.model(x)
        
        print(f"✅ Forward pass réussi!")
        print(f"Output shape: {output.shape if hasattr(output, 'shape') else 'Multiple outputs'}")
        
except Exception as e:
    print(f"❌ Erreur forward pass: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Test terminé!")
print("\nRésumé:")
print("- Les modules Enhanced acceptent maintenant les arguments YOLO")
print("- Le paramètre 'n' est ignoré mais accepté pour compatibilité")
print("- Le modèle devrait maintenant se charger correctement")
