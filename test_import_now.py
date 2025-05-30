#!/usr/bin/env python3
"""Test immédiat de l'import Enhanced"""

import sys
import os
from pathlib import Path

# Naviguer vers le répertoire du projet
project_dir = Path(__file__).parent
os.chdir(project_dir)

print(f"📁 Répertoire: {project_dir}")
print(f"📁 Working dir: {os.getcwd()}")

# Vérifier que le fichier enhanced.py existe
enhanced_file = Path("ultralytics/nn/modules/enhanced.py")
print(f"📄 Fichier enhanced.py existe: {enhanced_file.exists()}")

if enhanced_file.exists():
    print(f"📊 Taille: {enhanced_file.stat().st_size} bytes")

# Test d'import
print("\n🧪 TEST D'IMPORT:")

try:
    # Ajouter le path pour être sûr
    sys.path.insert(0, str(project_dir))
    
    print("1. Import direct du module enhanced...")
    from ultralytics.nn.modules.enhanced import A2Module, RELAN
    print("✅ Import direct réussi")
    
    print("2. Test d'instanciation...")
    a2 = A2Module(64, 64)
    relan = RELAN(128, 128)
    print("✅ Instanciation réussie")
    
    print("3. Test via ultralytics.nn.modules...")
    from ultralytics.nn.modules import A2Module as A2_alt
    print("✅ Import via modules réussi")
    
    print("\n🎉 TOUS LES TESTS PASSENT !")
    
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    
    # Diagnostic
    print("\n🔍 DIAGNOSTIC:")
    
    # Vérifier __init__.py
    init_file = Path("ultralytics/nn/modules/__init__.py")
    if init_file.exists():
        with open(init_file, 'r') as f:
            content = f.read()
        
        if 'from .enhanced import *' in content:
            print("✅ Import enhanced dans __init__.py")
        else:
            print("❌ Import enhanced MANQUANT dans __init__.py")
    
    # Essayer d'importer ultralytics directement
    try:
        import ultralytics
        print(f"✅ Ultralytics installé: {ultralytics.__file__}")
    except:
        print("❌ Ultralytics non installé")
    
except Exception as e:
    print(f"❌ Erreur générale: {e}")
    import traceback
    traceback.print_exc()
