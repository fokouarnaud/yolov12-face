#!/usr/bin/env python3
"""
🔧 Script de correction du problème d'import Enhanced
Résout le conflit de noms entre modules
"""

import sys
import os
import shutil
from pathlib import Path

def fix_enhanced_module():
    """Corrige le module enhanced pour éviter les conflits"""
    
    print("🔧 CORRECTION DU MODULE ENHANCED")
    print("=" * 50)
    
    # Chemins
    project_root = Path.cwd().parent if Path.cwd().name == 'debug' else Path.cwd()
    enhanced_path = project_root / "ultralytics/nn/modules/enhanced.py"
    
    print(f"📁 Projet: {project_root}")
    print(f"📄 Fichier enhanced: {enhanced_path}")
    
    if not enhanced_path.exists():
        print("❌ Fichier enhanced.py non trouvé!")
        return False
    
    # Lire le contenu actuel
    with open(enhanced_path, 'r') as f:
        content = f.read()
    
    # Corriger le conflit de noms
    # Remplacer C2PSA par C2PSA_Enhanced pour éviter le conflit
    new_content = content.replace(
        "C2PSA = RELAN", 
        "C2PSA_Enhanced = RELAN  # Renommé pour éviter conflit avec block.py"
    )
    
    # Mettre à jour __all__
    new_content = new_content.replace(
        "__all__ = ['A2Module', 'RELAN', 'FlashAttention', 'CrossScaleAttention', \n           'MicroExpressionAttention', 'SpatialAttention', 'C2PSA']",
        "__all__ = ['A2Module', 'RELAN', 'FlashAttention', 'CrossScaleAttention', \n           'MicroExpressionAttention', 'SpatialAttention', 'C2PSA_Enhanced']"
    )
    
    # Sauvegarder le fichier corrigé
    with open(enhanced_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Module enhanced.py corrigé")
    
    # Vérifier __init__.py
    init_path = project_root / "ultralytics/nn/modules/__init__.py"
    
    if init_path.exists():
        with open(init_path, 'r') as f:
            init_content = f.read()
        
        # S'assurer que l'import est correct
        if 'from .enhanced import *' not in init_content:
            # Trouver où insérer
            lines = init_content.split('\n')
            for i, line in enumerate(lines):
                if 'from .transformer import' in line:
                    lines.insert(i + 1, 'from .enhanced import *')
                    break
            
            with open(init_path, 'w') as f:
                f.write('\n'.join(lines))
            
            print("✅ __init__.py mis à jour")
    
    return True

def test_import():
    """Test l'import après correction"""
    
    print("\n🧪 TEST D'IMPORT")
    print("=" * 20)
    
    # Nettoyer le cache
    modules_to_clear = [m for m in sys.modules.keys() if 'enhanced' in m or 'ultralytics.nn' in m]
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    try:
        # Test import direct
        from ultralytics.nn.modules.enhanced import A2Module, RELAN
        print("✅ Import direct réussi")
        
        # Test instanciation
        a2 = A2Module(64, 64)
        relan = RELAN(128, 128)
        print("✅ Instanciation réussie")
        
        # Test import via __init__
        from ultralytics.nn.modules import A2Module as A2_test
        print("✅ Import via __init__ réussi")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Naviguer vers le répertoire du projet
    if Path.cwd().name == 'debug':
        os.chdir('..')
    
    success = fix_enhanced_module()
    if success:
        test_import()
