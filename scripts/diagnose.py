#!/usr/bin/env python3
"""
🔍 Diagnostic YOLOv12-Face Enhanced
==================================

Script de diagnostic pour identifier et résoudre les problèmes d'import.
"""

import sys
import os
from pathlib import Path


def diagnose_environment():
    """Diagnostic de l'environnement Python et des dépendances"""
    
    print("🔍 DIAGNOSTIC ENVIRONNEMENT")
    print("=" * 40)
    
    # Info Python
    print(f"🐍 Python: {sys.version}")
    print(f"📍 Executable: {sys.executable}")
    print(f"📁 Working dir: {os.getcwd()}")
    
    # Vérifier PyTorch
    try:
        import torch
        print(f"🔥 PyTorch: {torch.__version__}")
        print(f"🎮 CUDA: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch non installé")
        return False
    
    # Vérifier les dépendances manquantes
    print("\n📦 VÉRIFICATION DES DÉPENDANCES")
    print("-" * 30)
    
    dependencies = [
        'ultralytics',
        'huggingface_hub', 
        'pillow',
        'opencv-python',
        'matplotlib',
        'seaborn',
        'pandas'
    ]
    
    missing = []
    for dep in dependencies:
        try:
            if dep == 'opencv-python':
                import cv2
                print(f"✅ OpenCV: {cv2.__version__}")
            elif dep == 'pillow':
                import PIL
                print(f"✅ Pillow: {PIL.__version__}")
            else:
                module = __import__(dep.replace('-', '_'))
                if hasattr(module, '__version__'):
                    print(f"✅ {dep}: {module.__version__}")
                else:
                    print(f"✅ {dep}: installé")
        except ImportError:
            print(f"❌ {dep}: MANQUANT")
            missing.append(dep)
    
    if missing:
        print(f"\n🚨 DÉPENDANCES MANQUANTES:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True


def diagnose_ultralytics():
    """Diagnostic spécifique à Ultralytics"""
    
    print("\n🔍 DIAGNOSTIC ULTRALYTICS")
    print("=" * 30)
    
    # Chercher Ultralytics local
    project_root = Path.cwd()
    local_ultralytics = project_root / 'ultralytics'
    
    if local_ultralytics.exists():
        print(f"📁 Ultralytics local trouvé: {local_ultralytics}")
        
        # Vérifier les fichiers clés
        key_files = [
            'ultralytics/__init__.py',
            'ultralytics/models/__init__.py',
            'ultralytics/nn/modules/__init__.py'
        ]
        
        for file_path in key_files:
            full_path = project_root / file_path
            if full_path.exists():
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} MANQUANT")
    else:
        print("⚠️ Ultralytics local non trouvé")
    
    # Test d'import
    try:
        sys.path.insert(0, str(project_root))
        from ultralytics import YOLO
        print("✅ Import Ultralytics réussi")
        return True
    except Exception as e:
        print(f"❌ Erreur import Ultralytics: {e}")
        return False


def diagnose_configs():
    """Diagnostic des fichiers de configuration"""
    
    print("\n🔍 DIAGNOSTIC CONFIGURATIONS")
    print("=" * 35)
    
    project_root = Path.cwd()
    configs_dir = project_root / 'scripts' / 'configs'
    
    if not configs_dir.exists():
        print(f"❌ Dossier configs manquant: {configs_dir}")
        return False
    
    print(f"📁 Configs trouvés: {configs_dir}")
    
    # Fichiers requis
    required_files = [
        'scripts/configs/datasets/widerface.yaml',
        'scripts/configs/models/v12/yolov12-face.yaml', 
        'scripts/configs/models/v12/yolov12-face-enhanced.yaml',
        'scripts/configs/modules/enhanced.py'
    ]
    
    all_present = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} MANQUANT")
            all_present = False
    
    return all_present


def provide_solutions():
    """Fournit des solutions aux problèmes identifiés"""
    
    print("\n💡 SOLUTIONS RECOMMANDÉES")
    print("=" * 30)
    
    print("1. 📦 INSTALLER LES DÉPENDANCES:")
    print("   pip install ultralytics huggingface_hub pillow opencv-python")
    print("   pip install matplotlib seaborn pandas")
    
    print("\n2. 🔄 RESTAURER LES CONFIGURATIONS:")
    print("   python scripts/restore_configs.py")
    
    print("\n3. 🧪 TESTER L'INSTALLATION:")
    print("   python -c \"from ultralytics import YOLO; print('✅ OK')\"")
    
    print("\n4. 📓 UTILISER LE NOTEBOOK:")
    print("   jupyter notebook train_yolov12_enhanced.ipynb")
    
    print("\n5. 🆘 SI PROBLÈMES PERSISTENT:")
    print("   - Créer un nouvel environnement virtuel")
    print("   - Installer depuis un environnement propre")
    print("   - Vérifier les conflits de versions")


def main():
    """Fonction principale de diagnostic"""
    
    print("🚀 YOLOv12-Face Enhanced - Diagnostic\n")
    
    # Diagnostic complet
    env_ok = diagnose_environment()
    ultralytics_ok = diagnose_ultralytics() 
    configs_ok = diagnose_configs()
    
    print(f"\n📊 RÉSUMÉ DIAGNOSTIC")
    print("=" * 25)
    print(f"Environnement: {'✅' if env_ok else '❌'}")
    print(f"Ultralytics: {'✅' if ultralytics_ok else '❌'}")
    print(f"Configurations: {'✅' if configs_ok else '❌'}")
    
    if env_ok and ultralytics_ok and configs_ok:
        print("\n🎉 DIAGNOSTIC POSITIF - Tout est prêt !")
    else:
        provide_solutions()
    
    return env_ok and ultralytics_ok and configs_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
