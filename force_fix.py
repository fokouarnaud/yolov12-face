#!/usr/bin/env python3
"""
🔄 Force Restore et Test Enhanced Modules
"""

import sys
import os
import shutil
import importlib
from pathlib import Path

def force_restore():
    """Force la restauration avec vérifications"""
    
    print("🔄 FORCE RESTORE - YOLOv12-Face Enhanced")
    print("=" * 50)
    
    # Chemins
    project_root = Path.cwd()
    
    print(f"📁 Projet: {project_root}")
    
    # 1. Vérifier et copier enhanced.py
    src_enhanced = project_root / "scripts/configs/modules/enhanced.py"
    dst_enhanced = project_root / "ultralytics/nn/modules/enhanced.py"
    
    print(f"\n📄 FICHIER ENHANCED.PY:")
    print(f"   Source: {src_enhanced} ({'✅' if src_enhanced.exists() else '❌'})")
    print(f"   Destination: {dst_enhanced} ({'✅' if dst_enhanced.exists() else '❌'})")
    
    if src_enhanced.exists():
        # Créer le dossier de destination
        dst_enhanced.parent.mkdir(parents=True, exist_ok=True)
        
        # Copier le fichier
        shutil.copy2(src_enhanced, dst_enhanced)
        print(f"✅ Fichier enhanced.py copié")
    else:
        print(f"❌ Fichier source enhanced.py manquant")
        return False
    
    # 2. Vérifier __init__.py
    init_file = project_root / "ultralytics/nn/modules/__init__.py"
    
    if init_file.exists():
        with open(init_file, 'r') as f:
            content = f.read()
        
        print(f"\n📝 __INIT__.PY:")
        
        if 'from .enhanced import *' in content:
            print("✅ Import enhanced présent")
        else:
            print("❌ Import enhanced manquant - ajout...")
            
            # Ajouter l'import
            lines = content.split('\n')
            
            # Trouver où insérer
            for i, line in enumerate(lines):
                if 'from .head import' in line:
                    lines.insert(i + 1, 'from .enhanced import *')
                    break
            
            # Sauvegarder
            with open(init_file, 'w') as f:
                f.write('\n'.join(lines))
            
            print("✅ Import enhanced ajouté")
    
    print(f"\n✅ Force restore terminé")
    return True

def force_test():
    """Test avec rechargement forcé"""
    
    print(f"\n🧪 FORCE TEST")
    print("=" * 20)
    
    # Nettoyer le cache des modules
    modules_to_clear = []
    for module_name in list(sys.modules.keys()):
        if 'enhanced' in module_name or 'ultralytics.nn.modules' in module_name:
            modules_to_clear.append(module_name)
    
    print(f"🧹 Nettoyage cache: {len(modules_to_clear)} modules")
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    # Test d'import
    try:
        print("1. Import direct enhanced...")
        
        # Import direct du fichier
        import importlib.util
        enhanced_path = Path("ultralytics/nn/modules/enhanced.py")
        
        spec = importlib.util.spec_from_file_location("enhanced", enhanced_path)
        enhanced_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enhanced_module)
        
        print("✅ Import direct réussi")
        
        # Test des classes
        A2Module = enhanced_module.A2Module
        RELAN = enhanced_module.RELAN
        
        print("2. Test instanciation...")
        a2 = A2Module(64, 64)
        relan = RELAN(128, 128)
        print("✅ Instanciation réussie")
        
        print("3. Import via ultralytics...")
        from ultralytics.nn.modules.enhanced import A2Module as A2_ultra
        print("✅ Import via ultralytics réussi")
        
        print("\n🎉 TOUS LES TESTS PASSENT !")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = force_restore()
    if success:
        force_test()
