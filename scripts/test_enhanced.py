#!/usr/bin/env python3
"""
🧪 Test de vérification des modules Enhanced
===========================================
"""

import sys
import torch
from pathlib import Path

def test_enhanced_import():
    """Test d'import des modules enhanced"""
    
    print("🧪 TEST D'IMPORT DES MODULES ENHANCED")
    print("=" * 45)
    
    try:
        # Test import direct
        from ultralytics.nn.modules.enhanced import A2Module, RELAN
        print("✅ Import direct réussi: A2Module, RELAN")
        
        # Test import via ultralytics.nn.modules
        from ultralytics.nn.modules import A2Module as A2_alt, RELAN as RELAN_alt
        print("✅ Import via modules réussi")
        
        # Test instanciation
        a2 = A2Module(64, 64)
        relan = RELAN(128, 128)
        print("✅ Instanciation réussie")
        
        # Test forward pass
        x1 = torch.randn(1, 64, 32, 32)
        x2 = torch.randn(1, 128, 16, 16)
        
        out1 = a2(x1)
        out2 = relan(x2)
        
        print(f"✅ A2Module: {x1.shape} -> {out1.shape}")
        print(f"✅ RELAN: {x2.shape} -> {out2.shape}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur de test: {e}")
        return False

def test_model_creation():
    """Test de création du modèle Enhanced"""
    
    print(f"\n🏗️ TEST DE CRÉATION DU MODÈLE")
    print("=" * 35)
    
    try:
        from ultralytics import YOLO
        
        # Vérifier que le fichier config existe
        config_path = Path('ultralytics/cfg/models/v12/yolov12-face-enhanced.yaml')
        if not config_path.exists():
            print(f"❌ Fichier config non trouvé: {config_path}")
            return False
        
        print(f"✅ Fichier config trouvé: {config_path}")
        
        # Créer le modèle
        model = YOLO(str(config_path))
        print("✅ Modèle Enhanced créé avec succès")
        
        # Tester une prédiction factice
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model.model(dummy_input)
        print(f"✅ Test forward pass: input {dummy_input.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur de création modèle: {e}")
        return False

def test_diagnostics():
    """Diagnostics détaillés"""
    
    print(f"\n🔍 DIAGNOSTICS DÉTAILLÉS")
    print("=" * 30)
    
    # Vérifier les fichiers
    files_to_check = [
        'ultralytics/nn/modules/enhanced.py',
        'ultralytics/nn/modules/__init__.py',
        'ultralytics/cfg/models/v12/yolov12-face-enhanced.yaml'
    ]
    
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} MANQUANT")
    
    # Vérifier le contenu de __init__.py
    init_file = Path('ultralytics/nn/modules/__init__.py')
    if init_file.exists():
        with open(init_file, 'r') as f:
            content = f.read()
        
        if 'from .enhanced import *' in content:
            print("✅ Import enhanced dans __init__.py")
        else:
            print("❌ Import enhanced MANQUANT dans __init__.py")
        
        enhanced_modules = ['A2Module', 'RELAN']
        for module in enhanced_modules:
            if module in content:
                print(f"✅ {module} dans __all__")
            else:
                print(f"❌ {module} MANQUANT dans __all__")

def main():
    """Fonction principale de test"""
    
    print("🚀 YOLOv12-Face Enhanced - Tests de Vérification\n")
    
    # Tests
    import_ok = test_enhanced_import()
    model_ok = test_model_creation()
    test_diagnostics()
    
    # Résumé
    print(f"\n📊 RÉSUMÉ DES TESTS")
    print("=" * 25)
    print(f"Import modules: {'✅' if import_ok else '❌'}")
    print(f"Création modèle: {'✅' if model_ok else '❌'}")
    
    if import_ok and model_ok:
        print("\n🎉 TOUS LES TESTS PASSENT !")
        print("Le système Enhanced est prêt à l'emploi.")
        return True
    else:
        print("\n❌ CERTAINS TESTS ÉCHOUENT")
        print("Vérifiez les erreurs ci-dessus.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
