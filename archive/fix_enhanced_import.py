#!/usr/bin/env python3
"""
🔧 Script de correction définitive du problème d'import Enhanced
Version simplifiée et efficace
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess

def check_ultralytics_installation():
    """Vérifie l'installation d'Ultralytics"""
    try:
        import ultralytics
        print(f"✅ Ultralytics installé: {ultralytics.__version__}")
        print(f"📁 Chemin: {ultralytics.__file__}")
        return True
    except ImportError:
        print("❌ Ultralytics non installé")
        return False

def find_ultralytics_path():
    """Trouve le chemin réel d'Ultralytics"""
    try:
        import ultralytics
        return Path(ultralytics.__file__).parent
    except:
        # Si non installé, retourner le chemin local
        return Path.cwd() / "ultralytics"

def create_enhanced_module():
    """Crée le module enhanced.py avec le contenu correct"""
    
    content = '''"""
Modules Enhanced pour YOLOv12-Face
Version simplifiée sans conflits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class A2Module(nn.Module):
    """Area Attention Module simplifié"""
    
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
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
        x = x * channel_att
        
        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv_spatial(torch.cat([avg_spatial, max_spatial], dim=1)))
        x = x * spatial_att
        
        return self.conv_out(x)


class RELAN(nn.Module):
    """Residual Efficient Layer Aggregation Network"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Multi-scale convolutions
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels // 4, 7, padding=3)
        
        # Feature fusion
        self.fusion = nn.Conv2d(out_channels, out_channels, 1)
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Multi-scale features
        feats = [
            self.conv1x1(x),
            self.conv3x3(x),
            self.conv5x5(x),
            self.conv7x7(x)
        ]
        
        # Concatenate and fuse
        fused = self.fusion(torch.cat(feats, dim=1))
        
        # Residual connection
        return self.relu(fused + self.residual(x))


# Alias pour compatibilité
FlashAttention = A2Module
CrossScaleAttention = A2Module
MicroExpressionAttention = A2Module

__all__ = ['A2Module', 'RELAN', 'FlashAttention', 'CrossScaleAttention', 'MicroExpressionAttention']
'''
    return content

def fix_import_issue():
    """Corrige le problème d'import de manière définitive"""
    
    print("🔧 CORRECTION DU PROBLÈME D'IMPORT ENHANCED")
    print("=" * 50)
    
    # 1. Vérifier l'installation d'Ultralytics
    if not check_ultralytics_installation():
        print("📦 Installation d'Ultralytics...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics", "--upgrade"])
    
    # 2. Trouver le chemin réel d'Ultralytics
    ultra_path = find_ultralytics_path()
    print(f"📁 Chemin Ultralytics: {ultra_path}")
    
    # 3. Créer le module enhanced dans le bon répertoire
    enhanced_path = ultra_path / "nn" / "modules" / "enhanced.py"
    
    # Créer le répertoire si nécessaire
    enhanced_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Écrire le fichier
    with open(enhanced_path, 'w', encoding='utf-8') as f:
        f.write(create_enhanced_module())
    
    print(f"✅ Module enhanced.py créé: {enhanced_path}")
    
    # 4. Mettre à jour __init__.py
    init_path = enhanced_path.parent / "__init__.py"
    
    if init_path.exists():
        with open(init_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'from .enhanced import' not in content:
            # Ajouter l'import à la fin de la section des imports
            lines = content.split('\n')
            
            # Trouver la dernière ligne d'import
            last_import_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('from .') and 'import' in line:
                    last_import_idx = i
            
            # Insérer après la dernière ligne d'import
            lines.insert(last_import_idx + 1, 'from .enhanced import *')
            
            # Écrire le fichier
            with open(init_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            print("✅ __init__.py mis à jour")
    
    # 5. Copier aussi dans le répertoire local si différent
    local_path = Path.cwd() / "ultralytics"
    if local_path.exists() and local_path != ultra_path:
        local_enhanced = local_path / "nn" / "modules" / "enhanced.py"
        local_enhanced.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(enhanced_path, local_enhanced)
        print(f"✅ Copie locale créée: {local_enhanced}")
    
    # 6. Nettoyer le cache Python
    for cache_dir in [
        ultra_path / "nn" / "modules" / "__pycache__",
        local_path / "nn" / "modules" / "__pycache__" if local_path.exists() else None
    ]:
        if cache_dir and cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"✅ Cache supprimé: {cache_dir}")
    
    # 7. Sauvegarder dans scripts/configs
    backup_path = Path.cwd() / "scripts" / "configs" / "modules" / "enhanced.py"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(create_enhanced_module())
    print(f"✅ Sauvegarde créée: {backup_path}")
    
    # 8. Forcer le rechargement des modules
    modules_to_remove = [k for k in sys.modules.keys() if 'ultralytics' in k or 'enhanced' in k]
    for module in modules_to_remove:
        sys.modules.pop(module, None)
    
    print(f"✅ {len(modules_to_remove)} modules retirés du cache")

def test_import():
    """Test l'import après correction"""
    
    print("\n🧪 TEST D'IMPORT")
    print("=" * 20)
    
    # Forcer le rechargement
    if 'ultralytics' in sys.modules:
        del sys.modules['ultralytics']
    
    try:
        # Test 1: Import Ultralytics
        import ultralytics
        print("✅ Import ultralytics réussi")
        
        # Test 2: Import direct du module
        from ultralytics.nn.modules.enhanced import A2Module, RELAN
        print("✅ Import direct enhanced réussi")
        
        # Test 3: Instanciation
        a2 = A2Module(64, 64)
        relan = RELAN(128, 128)
        print("✅ Instanciation des modules réussie")
        
        # Test 4: Import via __init__
        from ultralytics.nn.modules import A2Module as A2Test
        print("✅ Import via __init__ réussi")
        
        # Test 5: YOLO
        from ultralytics import YOLO
        print("✅ Import YOLO réussi")
        
        print("\n🎉 TOUS LES TESTS PASSENT !")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_test():
    """Crée un script de test simple"""
    
    test_content = '''#!/usr/bin/env python3
"""Test simple du module Enhanced"""

import sys
sys.path.insert(0, '.')

try:
    from ultralytics.nn.modules.enhanced import A2Module, RELAN
    print("✅ Import Enhanced réussi!")
    
    # Test basique
    a2 = A2Module(64, 64)
    relan = RELAN(128, 128)
    print("✅ Modules fonctionnels!")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    print("💡 Exécutez: python fix_enhanced_import.py")
'''
    
    with open('test_enhanced.py', 'w') as f:
        f.write(test_content)
    
    print("\n📝 Script de test créé: test_enhanced.py")

if __name__ == "__main__":
    # Nettoyer l'écran
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("🚀 YOLOv12-FACE ENHANCED - CORRECTION DÉFINITIVE")
    print("=" * 60)
    
    # Appliquer la correction
    fix_import_issue()
    
    # Tester
    success = test_import()
    
    if success:
        create_simple_test()
        
        print("\n✅ PROBLÈME RÉSOLU !")
        print("\n📋 PROCHAINES ÉTAPES:")
        print("1. Test rapide: python test_enhanced.py")
        print("2. Entraînement: python scripts/train_enhanced.py")
        print("3. Notebook: jupyter notebook train_yolov12_enhanced.ipynb")
    else:
        print("\n❌ Le problème persiste.")
        print("💡 Essayez:")
        print("1. pip uninstall ultralytics")
        print("2. pip install ultralytics")
        print("3. python fix_enhanced_import.py")
