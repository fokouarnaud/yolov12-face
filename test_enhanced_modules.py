"""Test script pour vérifier que les modules Enhanced sont bien reconnus"""

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

# Test 2: Import via modules
print("\nTest 2: Import via modules...")
try:
    from ultralytics.nn.modules import A2Module, RELAN
    print("✅ Import via modules réussi: A2Module et RELAN")
except Exception as e:
    print(f"❌ Erreur import via modules: {e}")

# Test 3: Vérifier dans globals() de tasks.py
print("\nTest 3: Vérifier globals() dans tasks.py...")
try:
    import ultralytics.nn.tasks as tasks
    if 'A2Module' in dir(tasks):
        print("✅ A2Module trouvé dans tasks.py")
    else:
        print("❌ A2Module non trouvé dans tasks.py")
    
    if 'RELAN' in dir(tasks):
        print("✅ RELAN trouvé dans tasks.py")
    else:
        print("❌ RELAN non trouvé dans tasks.py")
except Exception as e:
    print(f"❌ Erreur accès tasks.py: {e}")

# Test 4: Charger le modèle Enhanced
print("\nTest 4: Chargement du modèle Enhanced...")
try:
    from ultralytics import YOLO
    
    # Spécifier le chemin complet du fichier de configuration
    config_path = Path(__file__).parent / "ultralytics" / "cfg" / "models" / "v12" / "yolov12-face-enhanced.yaml"
    print(f"Chemin de config: {config_path}")
    
    # Créer le modèle
    model = YOLO(str(config_path))
    print("✅ Modèle Enhanced chargé avec succès!")
    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Test terminé!")
