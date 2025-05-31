#!/usr/bin/env python3
"""
Script de validation complète pour YOLOv12-Face Enhanced
Ce script teste tous les composants nécessaires
"""

import sys
import os
from pathlib import Path
import traceback

def main():
    print("🔧 VALIDATION YOLOV12-FACE ENHANCED")
    print("=" * 50)
    
    # 1. Configuration du path
    current_dir = Path.cwd()
    print(f"📁 Répertoire courant: {current_dir}")
    
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        print(f"✅ Path ajouté: {current_dir}")
    
    # 2. Vérification structure
    print(f"\n📂 Vérification de la structure:")
    
    required_files = [
        "ultralytics/__init__.py",
        "ultralytics/nn/modules/__init__.py", 
        "ultralytics/nn/modules/enhanced.py",
        "ultralytics/cfg/models/v12/yolov12-face.yaml",
        "ultralytics/cfg/datasets/widerface.yaml"
    ]
    
    all_files_exist = True
    for file_path in required_files:
        full_path = current_dir / file_path
        status = "✅" if full_path.exists() else "❌"
        print(f"   {status} {file_path}")
        if not full_path.exists():
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ ERREUR: Fichiers manquants!")
        print("💡 Assurez-vous d'être dans le bon répertoire avec le fork complet")
        return False
    
    # 3. Test des imports
    print(f"\n🧪 Test des imports:")
    
    try:
        print("   • Import torch...")
        import torch
        print(f"     ✅ PyTorch {torch.__version__}")
        
        print("   • Import ultralytics...")
        from ultralytics import YOLO
        print("     ✅ YOLO importé")
        
        print("   • Import modules enhanced...")
        from ultralytics.nn.modules.enhanced import A2Module, RELAN
        print("     ✅ A2Module et RELAN importés")
        
        print("   • Import depuis modules...")
        from ultralytics.nn.modules import A2Module as A2_alt, RELAN as RELAN_alt
        print("     ✅ Imports alternatifs OK")
        
    except Exception as e:
        print(f"     ❌ ERREUR: {e}")
        traceback.print_exc()
        return False
    
    # 4. Test fonctionnel des modules
    print(f"\n⚙️ Test fonctionnel:")
    
    try:
        # Créer des données de test
        x = torch.randn(2, 64, 32, 32)
        print(f"   • Données test: {x.shape}")
        
        # Test A2Module
        a2 = A2Module(64, 128)
        out_a2 = a2(x)
        print(f"   • A2Module: {x.shape} -> {out_a2.shape} ✅")
        
        # Test RELAN
        relan = RELAN(64, 128)
        out_relan = relan(x)
        print(f"   • RELAN: {x.shape} -> {out_relan.shape} ✅")
        
        # Test avec même taille
        a2_same = A2Module(64, 64)
        relan_same = RELAN(64, 64)
        out_a2_same = a2_same(x)
        out_relan_same = relan_same(x)
        print(f"   • Même taille OK: A2={out_a2_same.shape}, RELAN={out_relan_same.shape} ✅")
        
    except Exception as e:
        print(f"   ❌ ERREUR fonctionnelle: {e}")
        traceback.print_exc()
        return False
    
    # 5. Test de chargement de modèle
    print(f"\n🏗️ Test chargement modèle:")
    
    try:
        # Test avec modèle de base
        base_model_path = "ultralytics/cfg/models/v12/yolov12-face.yaml"
        if Path(base_model_path).exists():
            model_base = YOLO(base_model_path)
            print(f"   ✅ Modèle de base chargé: {base_model_path}")
        
        # Test avec modèle enhanced si disponible
        enhanced_model_path = "ultralytics/cfg/models/v12/yolov12-face-enhanced.yaml"
        if Path(enhanced_model_path).exists():
            model_enhanced = YOLO(enhanced_model_path)
            print(f"   ✅ Modèle Enhanced chargé: {enhanced_model_path}")
        else:
            print(f"   ⚠️ Modèle Enhanced non trouvé: {enhanced_model_path}")
            
    except Exception as e:
        print(f"   ❌ ERREUR chargement: {e}")
        traceback.print_exc()
        return False
    
    # 6. Informations système
    print(f"\n💻 Informations système:")
    print(f"   • Python: {sys.version.split()[0]}")
    print(f"   • PyTorch: {torch.__version__}")
    print(f"   • CUDA: {'✅' if torch.cuda.is_available() else '❌'}")
    if torch.cuda.is_available():
        print(f"   • GPU: {torch.cuda.get_device_name(0)}")
        print(f"   • VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 7. Résumé
    print(f"\n🎉 VALIDATION RÉUSSIE !")
    print("=" * 50)
    print("✅ Tous les composants sont fonctionnels")
    print("✅ Les modules Enhanced sont prêts")
    print("✅ YOLOv12-Face peut être entraîné")
    
    print(f"\n🚀 Prochaines étapes:")
    print("   1. Préparer le dataset WIDERFace")
    print("   2. Lancer l'entraînement avec le notebook corrigé")
    print("   3. Utiliser train_yolov12_enhanced_fixed.ipynb")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✅ Script terminé avec succès!")
        sys.exit(0)
    else:
        print(f"\n❌ Script terminé avec des erreurs!")
        sys.exit(1)
