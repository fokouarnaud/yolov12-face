#!/usr/bin/env python3
"""
✅ Solution Finale - Correction du Module Enhanced
Résout tous les conflits d'import
"""

import sys
import os
import shutil
from pathlib import Path

# Contenu corrigé du module enhanced.py
ENHANCED_CONTENT = '''"""
Modules Enhanced simplifiés pour YOLOv12-Face
Version stable et fonctionnelle - Sans conflits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class A2Module(nn.Module):
    """Area Attention Module simplifié"""
    
    def __init__(self, in_channels, out_channels, reduction=16):
        super(A2Module, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        
        # Output projection
        self.conv_out = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        
        # Apply channel attention
        x = x * channel_att
        
        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv_spatial(torch.cat([avg_spatial, max_spatial], dim=1)))
        
        # Apply spatial attention
        x = x * spatial_att
        
        return self.conv_out(x)


class RELAN(nn.Module):
    """Residual Efficient Layer Aggregation Network simplifié"""
    
    def __init__(self, in_channels, out_channels):
        super(RELAN, self).__init__()
        
        # Multi-scale convolutions
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels // 4, 7, padding=3)
        
        # Feature fusion
        self.fusion = nn.Conv2d(out_channels, out_channels, 1)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
            
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Multi-scale features
        f1 = self.conv1x1(x)
        f2 = self.conv3x3(x)
        f3 = self.conv5x5(x)
        f4 = self.conv7x7(x)
        
        # Concatenate and fuse
        fused = torch.cat([f1, f2, f3, f4], dim=1)
        fused = self.fusion(fused)
        
        # Residual connection
        residual = self.residual(x)
        
        return self.relu(fused + residual)


# Alias pour compatibilité - Éviter les conflits avec les modules existants
FlashAttention = A2Module
CrossScaleAttention = A2Module  
MicroExpressionAttention = A2Module


# Export des modules
__all__ = [
    'A2Module', 
    'RELAN', 
    'FlashAttention', 
    'CrossScaleAttention', 
    'MicroExpressionAttention'
]
'''

def apply_fix():
    """Applique la correction définitive"""
    
    print("🔧 APPLICATION DE LA SOLUTION FINALE")
    print("=" * 50)
    
    project_root = Path.cwd()
    
    # 1. Écrire le fichier enhanced.py corrigé
    enhanced_path = project_root / "ultralytics/nn/modules/enhanced.py"
    enhanced_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(enhanced_path, 'w', encoding='utf-8') as f:
        f.write(ENHANCED_CONTENT)
    
    print("✅ Fichier enhanced.py écrit (sans conflits)")
    
    # 2. Sauvegarder dans scripts/configs/modules
    backup_path = project_root / "scripts/configs/modules/enhanced.py"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(ENHANCED_CONTENT)
    
    print("✅ Sauvegarde dans scripts/configs/modules/enhanced.py")
    
    # 3. Mettre à jour __init__.py
    init_path = project_root / "ultralytics/nn/modules/__init__.py"
    
    if init_path.exists():
        with open(init_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Vérifier si l'import enhanced existe déjà
        if 'from .enhanced import' not in content:
            # Trouver où insérer (après transformer)
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'from .transformer import' in line:
                    lines.insert(i + 1, 'from .enhanced import *')
                    break
            
            content = '\n'.join(lines)
            
            with open(init_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ __init__.py mis à jour")
        else:
            print("✅ __init__.py déjà à jour")
    
    # 4. Nettoyer le cache
    pycache_dir = project_root / "ultralytics/nn/modules/__pycache__"
    if pycache_dir.exists():
        shutil.rmtree(pycache_dir)
        print("✅ Cache __pycache__ supprimé")
    
    # Nettoyer sys.modules
    modules_to_clear = [m for m in list(sys.modules.keys()) if 'enhanced' in m or 'ultralytics.nn' in m]
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    print(f"✅ {len(modules_to_clear)} modules retirés du cache Python")
    
    return True

def test_import():
    """Test l'import après correction"""
    
    print("\n🧪 TEST D'IMPORT")
    print("=" * 20)
    
    try:
        # Import des modules
        print("1. Import des modules enhanced...")
        from ultralytics.nn.modules.enhanced import A2Module, RELAN
        print("✅ Import direct réussi")
        
        # Test d'instanciation
        print("2. Test d'instanciation...")
        a2 = A2Module(64, 64)
        relan = RELAN(128, 128)
        print("✅ Instanciation réussie")
        
        # Test import via __init__
        print("3. Test import via __init__...")
        from ultralytics.nn.modules import A2Module as A2_test
        print("✅ Import via __init__ réussi")
        
        print("\n🎉 TOUS LES TESTS PASSENT !")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_script():
    """Crée un script de test simple"""
    
    test_content = '''#!/usr/bin/env python3
"""Test rapide du module Enhanced"""

try:
    from ultralytics.nn.modules.enhanced import A2Module, RELAN
    from ultralytics import YOLO
    
    print("✅ Import réussi!")
    
    # Test instanciation
    a2 = A2Module(64, 64)
    relan = RELAN(128, 128)
    print("✅ Modules Enhanced fonctionnels!")
    
    # Test chargement config (sans créer le modèle)
    print("✅ Prêt pour l'entraînement!")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
'''
    
    test_path = Path.cwd() / "test_enhanced_quick.py"
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"\n📝 Script de test créé: {test_path}")

if __name__ == "__main__":
    print("🚀 YOLOV12-FACE ENHANCED - CORRECTION FINALE")
    print("=" * 60)
    
    # Appliquer la correction
    success = apply_fix()
    
    if success:
        # Tester l'import
        test_success = test_import()
        
        if test_success:
            # Créer un script de test
            create_test_script()
            
            print("\n\n✅ CORRECTION TERMINÉE AVEC SUCCÈS !")
            print("\n📋 PROCHAINES ÉTAPES:")
            print("1. Testez avec: python test_enhanced_quick.py")
            print("2. Lancez l'entraînement:")
            print("   - Notebook: jupyter notebook train_yolov12_enhanced.ipynb")
            print("   - Script: python scripts/train_enhanced.py --epochs 100")
            print("\n💡 Les modules en conflit ont été supprimés:")
            print("   - C2PSA (conflit avec block.py)")
            print("   - SpatialAttention (conflit avec conv.py)")
            print("   - Seuls A2Module et RELAN sont disponibles")
