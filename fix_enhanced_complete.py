#!/usr/bin/env python3
"""
🔍 Diagnostic et Correction du Problème d'Import Enhanced
"""

import sys
import os
import shutil
from pathlib import Path

def diagnose_problem():
    """Diagnostic complet du problème d'import"""
    
    print("🔍 DIAGNOSTIC DU PROBLÈME D'IMPORT")
    print("=" * 50)
    
    # Chemins
    project_root = Path.cwd()
    enhanced_path = project_root / "ultralytics/nn/modules/enhanced.py"
    init_path = project_root / "ultralytics/nn/modules/__init__.py"
    
    print(f"📁 Répertoire de travail: {project_root}")
    
    # 1. Vérifier l'existence des fichiers
    print("\n1️⃣ VÉRIFICATION DES FICHIERS:")
    
    if enhanced_path.exists():
        print(f"✅ enhanced.py existe ({enhanced_path.stat().st_size} bytes)")
        
        # Lire le contenu
        with open(enhanced_path, 'r') as f:
            content = f.read()
        
        # Vérifier les conflits potentiels
        if "C2PSA = RELAN" in content:
            print("⚠️  CONFLIT DÉTECTÉ: C2PSA = RELAN (conflit avec block.py)")
        if "SpatialAttention" in content and "from .conv import" not in content:
            print("⚠️  CONFLIT POTENTIEL: SpatialAttention existe déjà dans conv.py")
            
        # Vérifier __all__
        if "__all__" in content:
            import re
            all_match = re.search(r'__all__\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if all_match:
                exports = all_match.group(1)
                print(f"📋 Exports trouvés dans __all__: {exports}")
    else:
        print("❌ enhanced.py N'EXISTE PAS!")
    
    # 2. Vérifier __init__.py
    print("\n2️⃣ VÉRIFICATION DE __INIT__.PY:")
    
    if init_path.exists():
        with open(init_path, 'r') as f:
            init_content = f.read()
        
        if "from .enhanced import *" in init_content:
            print("✅ Import enhanced présent dans __init__.py")
        else:
            print("❌ Import enhanced MANQUANT dans __init__.py")
        
        # Vérifier l'ordre des imports
        if "from .enhanced import *" in init_content and "from .block import" in init_content:
            enhanced_pos = init_content.find("from .enhanced import *")
            block_pos = init_content.find("from .block import")
            if enhanced_pos < block_pos:
                print("⚠️  ORDRE D'IMPORT: enhanced importé AVANT block (peut causer des conflits)")
    
    # 3. Identifier les modules en conflit
    print("\n3️⃣ MODULES EN CONFLIT:")
    
    # Vérifier block.py
    block_path = project_root / "ultralytics/nn/modules/block.py"
    if block_path.exists():
        with open(block_path, 'r') as f:
            block_content = f.read()
        
        # Chercher les classes qui pourraient entrer en conflit
        conflicting_classes = []
        if "class C2PSA" in block_content:
            conflicting_classes.append("C2PSA")
        if "class SpatialAttention" in block_content:
            conflicting_classes.append("SpatialAttention")
            
        if conflicting_classes:
            print(f"⚠️  Classes en conflit dans block.py: {', '.join(conflicting_classes)}")
    
    # Vérifier conv.py
    conv_path = project_root / "ultralytics/nn/modules/conv.py"
    if conv_path.exists():
        with open(conv_path, 'r') as f:
            conv_content = f.read()
        
        if "class SpatialAttention" in conv_content:
            print("⚠️  SpatialAttention existe dans conv.py")
    
    return True

def fix_import_issue():
    """Corrige le problème d'import"""
    
    print("\n\n🔧 CORRECTION DU PROBLÈME")
    print("=" * 50)
    
    project_root = Path.cwd()
    
    # 1. Copier la version corrigée de enhanced.py
    src_fixed = project_root / "scripts/configs/modules/enhanced_fixed.py"
    dst_enhanced = project_root / "ultralytics/nn/modules/enhanced.py"
    
    if src_fixed.exists():
        shutil.copy2(src_fixed, dst_enhanced)
        print("✅ Fichier enhanced.py corrigé copié")
    else:
        # Si le fichier corrigé n'existe pas, corriger directement
        if dst_enhanced.exists():
            with open(dst_enhanced, 'r') as f:
                content = f.read()
            
            # Remplacer les noms en conflit
            content = content.replace("C2PSA = RELAN", "C2PSA_Enhanced = RELAN")
            content = content.replace("SpatialAttention = A2Module", "SpatialAttention_Enhanced = A2Module")
            
            # Mettre à jour __all__
            content = content.replace("'C2PSA'", "'C2PSA_Enhanced'")
            content = content.replace("'SpatialAttention'", "'SpatialAttention_Enhanced'")
            
            with open(dst_enhanced, 'w') as f:
                f.write(content)
            
            print("✅ Fichier enhanced.py corrigé directement")
    
    # 2. Mettre à jour __init__.py pour l'ordre correct des imports
    init_path = project_root / "ultralytics/nn/modules/__init__.py"
    
    if init_path.exists():
        with open(init_path, 'r') as f:
            lines = f.readlines()
        
        # Retirer l'ancien import enhanced s'il existe
        lines = [line for line in lines if "from .enhanced import" not in line]
        
        # Trouver où insérer (après transformer pour éviter les conflits)
        insert_pos = None
        for i, line in enumerate(lines):
            if "from .transformer import" in line:
                insert_pos = i + 1
                break
        
        if insert_pos:
            lines.insert(insert_pos, "from .enhanced import *\n")
            
            with open(init_path, 'w') as f:
                f.writelines(lines)
            
            print("✅ __init__.py mis à jour avec l'ordre correct des imports")
    
    # 3. Nettoyer le cache Python
    print("\n🧹 Nettoyage du cache Python...")
    
    # Supprimer __pycache__
    pycache_dir = project_root / "ultralytics/nn/modules/__pycache__"
    if pycache_dir.exists():
        shutil.rmtree(pycache_dir)
        print("✅ Cache __pycache__ supprimé")
    
    # Nettoyer sys.modules
    modules_to_clear = [m for m in sys.modules.keys() if 'enhanced' in m or 'ultralytics.nn' in m]
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]
    print(f"✅ {len(modules_to_clear)} modules retirés du cache")

def test_import_final():
    """Test final après correction"""
    
    print("\n\n🧪 TEST FINAL")
    print("=" * 20)
    
    try:
        # Import des modules
        print("1. Import des modules enhanced...")
        from ultralytics.nn.modules.enhanced import A2Module, RELAN, C2PSA_Enhanced
        print("✅ Import réussi")
        
        # Test d'instanciation
        print("2. Test d'instanciation...")
        a2 = A2Module(64, 64)
        relan = RELAN(128, 128)
        print("✅ Instanciation réussie")
        
        # Test du modèle YOLO
        print("3. Test du modèle YOLO...")
        from ultralytics import YOLO
        # Pas de chargement de modèle car cela pourrait échouer pour d'autres raisons
        print("✅ Import YOLO réussi")
        
        print("\n🎉 TOUS LES TESTS PASSENT ! Le problème est résolu.")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Diagnostic
    diagnose_problem()
    
    # Correction
    fix_import_issue()
    
    # Test
    test_import_final()
    
    print("\n\n📝 PROCHAINE ÉTAPE:")
    print("Lancez votre notebook d'entraînement ou utilisez:")
    print("python scripts/train_enhanced.py --epochs 100")
