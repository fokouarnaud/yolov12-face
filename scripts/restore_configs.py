#!/usr/bin/env python3
"""
🔄 Script de Restauration Automatique YOLOv12-Face Enhanced
==========================================================

Ce script restaure automatiquement les fichiers de configuration
personnalisés après une réinstallation d'Ultralytics.

Usage:
    python restore_configs.py

Auteur: YOLOv12-Face Enhanced Project
Date: 2025
"""

import shutil
import sys
from pathlib import Path


def restore_configs():
    """Restaure automatiquement les configurations personnalisées"""
    
    print("🔄 RESTAURATION DES CONFIGURATIONS YOLOv12-FACE ENHANCED")
    print("=" * 60)
    
    # Chemins de base
    project_root = Path(__file__).parent.parent
    configs_dir = Path(__file__).parent / "configs"
    
    print(f"📁 Projet: {project_root}")
    print(f"📁 Configs: {configs_dir}")
    
    # Mapping des fichiers à restaurer
    files_map = [
        ('datasets/widerface.yaml', 'ultralytics/cfg/datasets/widerface.yaml'),
        ('models/v12/yolov12-face.yaml', 'ultralytics/cfg/models/v12/yolov12-face.yaml'),
        ('models/v12/yolov12-face-enhanced.yaml', 'ultralytics/cfg/models/v12/yolov12-face-enhanced.yaml'),
        ('modules/enhanced.py', 'ultralytics/nn/modules/enhanced.py')
    ]
    
    # Vérifier que le dossier configs existe
    if not configs_dir.exists():
        print(f"❌ Dossier configs non trouvé: {configs_dir}")
        return False
    
    # Restaurer les fichiers
    restored_count = 0
    for src_rel, dst_rel in files_map:
        src_path = configs_dir / src_rel
        dst_path = project_root / dst_rel
        
        if src_path.exists():
            # Créer le dossier de destination si nécessaire
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copier le fichier
            shutil.copy2(src_path, dst_path)
            print(f"🔄 Restauré: {dst_rel}")
            restored_count += 1
        else:
            print(f"⚠️  Fichier source manquant: {src_rel}")
    
    # Mettre à jour __init__.py
    init_file = project_root / 'ultralytics/nn/modules/__init__.py'
    
    if init_file.exists():
        print("\n🔧 Mise à jour de __init__.py...")
        
        # Lire le contenu actuel
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Ajouter notre import si pas déjà présent
        enhanced_import = "from .enhanced import *"
        
        if enhanced_import not in content:
            lines = content.split('\n')
            
            # Trouver la ligne __all__ et insérer avant
            for i, line in enumerate(lines):
                if line.strip().startswith('__all__'):
                    lines.insert(i, enhanced_import)
                    break
            else:
                # Si __all__ n'est pas trouvé, ajouter à la fin des imports
                lines.insert(-1, enhanced_import)
            
            # Écrire le nouveau contenu
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            print("✅ __init__.py mis à jour avec les modules enhanced")
        else:
            print("ℹ️  __init__.py déjà à jour")
    else:
        print("⚠️  Fichier __init__.py non trouvé - Ultralytics pas installé ?")
    
    # Résumé
    print(f"\n📊 RÉSUMÉ DE LA RESTAURATION")
    print(f"   Fichiers restaurés: {restored_count}/{len(files_map)}")
    
    if restored_count == len(files_map):
        print("✅ Restauration terminée avec succès !")
        print("\n🧪 Test de vérification:")
        print("from ultralytics.nn.modules.enhanced import A2Module")
        print("from ultralytics import YOLO")
        print("model = YOLO('ultralytics/cfg/models/v12/yolov12-face-enhanced.yaml')")
        return True
    else:
        print("⚠️  Restauration partielle - vérifiez les fichiers manquants")
        return False


def verify_installation():
    """Vérifie que la restauration a fonctionné"""
    
    print("\n🧪 VÉRIFICATION DE L'INSTALLATION")
    print("=" * 40)
    
    try:
        # Test 1: Import des modules enhanced
        from ultralytics.nn.modules.enhanced import A2Module, RELAN
        print("✅ Modules enhanced importés")
        
        # Test 2: Test d'instanciation
        import torch
        x = torch.randn(1, 64, 32, 32)
        a2 = A2Module(64, 64)
        out = a2(x)
        print(f"✅ Test A2Module: {x.shape} -> {out.shape}")
        
        # Test 3: Chargement du modèle
        from ultralytics import YOLO
        model = YOLO('ultralytics/cfg/models/v12/yolov12-face-enhanced.yaml')
        print("✅ Modèle enhanced chargé")
        
        print("\n🎉 Tous les tests passent - Configuration OK !")
        return True
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("💡 Exécutez à nouveau la restauration")
        return False
    except Exception as e:
        print(f"❌ Erreur de test: {e}")
        return False


if __name__ == "__main__":
    print("🚀 YOLOv12-Face Enhanced - Restauration des Configurations\n")
    
    # Restaurer les fichiers
    success = restore_configs()
    
    if success:
        # Vérifier l'installation
        verify_installation()
    
    print(f"\n{'🎯 RESTAURATION TERMINÉE' if success else '❌ RESTAURATION ÉCHOUÉE'}")
    sys.exit(0 if success else 1)
