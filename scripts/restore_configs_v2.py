#!/usr/bin/env python3
"""
🔄 Script de restauration des configurations YOLOv12-Face Enhanced
Version corrigée pour éviter les conflits d'import
"""

import os
import shutil
from pathlib import Path
import sys

# Contenu corrigé du module enhanced.py (sans conflits)
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

def restore_configs():
    """Restaure les configurations personnalisées"""
    
    print("🔄 RESTAURATION DES CONFIGURATIONS")
    print("=" * 50)
    
    # Obtenir le répertoire racine du projet
    if hasattr(sys, '_MEIPASS'):
        # Si exécuté depuis un exécutable PyInstaller
        project_root = Path(os.path.dirname(os.path.abspath(__file__)))
    else:
        # Sinon, utiliser le répertoire courant
        project_root = Path.cwd()
    
    # Chemins source et destination
    configs_dir = project_root / "scripts/configs"
    ultralytics_dir = project_root / "ultralytics"
    
    # Vérifier que les répertoires existent
    if not configs_dir.exists():
        print(f"❌ Répertoire configs non trouvé: {configs_dir}")
        return False
    
    if not ultralytics_dir.exists():
        print(f"❌ Répertoire ultralytics non trouvé: {ultralytics_dir}")
        print("💡 Installez d'abord ultralytics: pip install ultralytics")
        return False
    
    # Copier les fichiers de configuration
    files_to_copy = [
        # Datasets
        ("datasets/widerface.yaml", "cfg/datasets/widerface.yaml"),
        # Models
        ("models/v12/yolov12-face.yaml", "cfg/models/v12/yolov12-face.yaml"),
        ("models/v12/yolov12-face-enhanced.yaml", "cfg/models/v12/yolov12-face-enhanced.yaml"),
    ]
    
    for src, dst in files_to_copy:
        src_path = configs_dir / src
        dst_path = ultralytics_dir / dst
        
        if src_path.exists():
            # Créer le répertoire de destination si nécessaire
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copier le fichier
            shutil.copy2(src_path, dst_path)
            print(f"✅ Copié: {src} → {dst}")
        else:
            print(f"⚠️  Fichier source non trouvé: {src}")
    
    # Écrire le module enhanced.py corrigé
    enhanced_path = ultralytics_dir / "nn/modules/enhanced.py"
    enhanced_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(enhanced_path, 'w', encoding='utf-8') as f:
        f.write(ENHANCED_CONTENT)
    
    print("✅ Module enhanced.py écrit (version corrigée)")
    
    # Modifier __init__.py pour importer les modules enhanced
    init_path = ultralytics_dir / "nn/modules/__init__.py"
    
    if init_path.exists():
        with open(init_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Vérifier si l'import existe déjà
        if 'from .enhanced import *' not in content:
            # Ajouter l'import après transformer
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'from .transformer import' in line:
                    lines.insert(i + 1, 'from .enhanced import *')
                    break
            
            # Sauvegarder
            with open(init_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            print("✅ __init__.py mis à jour avec l'import enhanced")
        else:
            print("✅ Import enhanced déjà présent dans __init__.py")
    
    # Nettoyer le cache Python
    pycache_dir = ultralytics_dir / "nn/modules/__pycache__"
    if pycache_dir.exists():
        shutil.rmtree(pycache_dir)
        print("✅ Cache Python nettoyé")
    
    print("\n✅ Restauration terminée!")
    return True

def test_import():
    """Test rapide de l'import"""
    
    print("\n🧪 TEST D'IMPORT")
    print("=" * 20)
    
    try:
        from ultralytics.nn.modules.enhanced import A2Module, RELAN
        print("✅ Import réussi!")
        
        # Test instanciation
        a2 = A2Module(64, 64)
        relan = RELAN(128, 128)
        print("✅ Modules fonctionnels!")
        
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    print("🚀 YOLOv12-Face Enhanced - Restauration")
    print("=" * 50)
    
    # Restaurer les configurations
    success = restore_configs()
    
    if success:
        # Tester l'import
        test_import()
        
        print("\n📋 UTILISATION:")
        print("1. Pour entraîner le modèle Enhanced:")
        print("   python scripts/train_enhanced.py --epochs 100")
        print("\n2. Pour utiliser le notebook:")
        print("   jupyter notebook train_yolov12_enhanced.ipynb")
