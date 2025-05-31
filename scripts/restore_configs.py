#!/usr/bin/env python3
"""
🔄 Script de restauration optimisé pour YOLOv12-Face Enhanced
Version corrigée pour éviter les erreurs de syntaxe
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess

class ConfigRestorer:
    def __init__(self):
        self.project_root = Path.cwd()
        self.configs_dir = self.project_root / "scripts" / "configs"
        self.success_count = 0
        self.total_count = 0
        
    def check_environment(self):
        """Vérifie l'environnement d'exécution"""
        print("🔍 VÉRIFICATION DE L'ENVIRONNEMENT")
        print("=" * 50)
        
        print(f"📁 Répertoire de travail: {self.project_root}")
        print(f"🐍 Python: {sys.version.split()[0]}")
        
        # Vérifier Ultralytics
        try:
            import ultralytics
            self.ultra_path = Path(ultralytics.__file__).parent
            print(f"✅ Ultralytics {ultralytics.__version__} installé")
            print(f"📁 Chemin: {self.ultra_path}")
            return True
        except ImportError:
            print("❌ Ultralytics non installé")
            return False
    
    def install_ultralytics(self):
        """Installe ou met à jour Ultralytics"""
        print("\n📦 Installation d'Ultralytics...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics", "--upgrade"], check=True)
            # Recharger pour obtenir le chemin
            import ultralytics
            self.ultra_path = Path(ultralytics.__file__).parent
            return True
        except:
            print("❌ Échec de l'installation")
            return False
    
    def create_enhanced_module(self):
        """Crée le contenu du module enhanced.py"""
        return '''"""
Modules Enhanced pour YOLOv12-Face
Version stable sans conflits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class A2Module(nn.Module):
    """Area Attention Module pour améliorer la détection de visages"""
    
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
    
    def restore_file(self, src_path, dst_path):
        """Restaure un fichier avec gestion d'erreurs"""
        self.total_count += 1
        try:
            # Créer le répertoire parent si nécessaire
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copier le fichier
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                print(f"✅ {src_path.name} → {dst_path.relative_to(self.ultra_path)}")
                self.success_count += 1
                return True
            else:
                print(f"❌ Source manquante: {src_path.name}")
                return False
        except Exception as e:
            print(f"❌ Erreur pour {src_path.name}: {e}")
            return False
    
    def restore_configs(self):
        """Restaure tous les fichiers de configuration"""
        print("\n🔄 RESTAURATION DES CONFIGURATIONS")
        print("=" * 50)
        
        # Liste des fichiers à restaurer
        files = [
            ("datasets/widerface.yaml", "cfg/datasets/widerface.yaml"),
            ("models/v12/yolov12-face.yaml", "cfg/models/v12/yolov12-face.yaml"),
            ("models/v12/yolov12-face-enhanced.yaml", "cfg/models/v12/yolov12-face-enhanced.yaml"),
        ]
        
        # Restaurer les fichiers YAML
        for src, dst in files:
            src_path = self.configs_dir / src
            dst_path = self.ultra_path / dst
            self.restore_file(src_path, dst_path)
        
        # Restaurer le module enhanced.py
        self.total_count += 1
        enhanced_path = self.ultra_path / "nn" / "modules" / "enhanced.py"
        enhanced_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(enhanced_path, 'w', encoding='utf-8') as f:
            f.write(self.create_enhanced_module())
        
        print(f"✅ enhanced.py → nn/modules/enhanced.py")
        self.success_count += 1
        
        # Sauvegarder aussi dans configs
        backup_path = self.configs_dir / "modules" / "enhanced.py"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(self.create_enhanced_module())
    
    def update_init_file(self):
        """Met à jour le fichier __init__.py de manière plus sûre"""
        print("\n🔧 MISE À JOUR DE __INIT__.PY")
        print("=" * 30)
        
        init_path = self.ultra_path / "nn" / "modules" / "__init__.py"
        
        if not init_path.exists():
            print("❌ __init__.py non trouvé")
            return False
        
        try:
            with open(init_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'from .enhanced import *' in content:
                print("✅ Import enhanced déjà présent")
                return True
            
            # Méthode plus sûre : ajouter l'import juste avant la définition de __all__
            lines = content.split('\n')
            
            # Trouver la ligne __all__
            all_line_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith('__all__'):
                    all_line_idx = i
                    break
            
            if all_line_idx is None:
                # Si pas de __all__, ajouter à la fin des imports
                import_line_idx = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith('from') and not line.startswith('import'):
                        import_line_idx = i - 1
                        break
                
                if import_line_idx > 0:
                    lines.insert(import_line_idx, 'from .enhanced import *')
                else:
                    # Ajouter après le dernier import
                    for i in range(len(lines)-1, -1, -1):
                        if lines[i].startswith('from .') or lines[i].startswith('import'):
                            lines.insert(i + 1, 'from .enhanced import *')
                            break
            else:
                # Ajouter juste avant __all__
                # Trouver la dernière ligne d'import avant __all__
                last_import_idx = 0
                for i in range(all_line_idx - 1, -1, -1):
                    if lines[i].strip() and (lines[i].startswith('from') or lines[i].startswith('import')):
                        last_import_idx = i
                        break
                
                # Insérer après le dernier import
                lines.insert(last_import_idx + 1, 'from .enhanced import *')
            
            # Écrire le fichier modifié
            with open(init_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            print("✅ Import enhanced ajouté")
            
            # Vérifier la syntaxe
            try:
                compile(open(init_path).read(), init_path, 'exec')
                print("✅ Syntaxe vérifiée")
                return True
            except SyntaxError as e:
                print(f"❌ Erreur de syntaxe: {e}")
                # Restaurer le fichier original
                with open(init_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return False
                
        except Exception as e:
            print(f"❌ Erreur lors de la mise à jour: {e}")
            return False
    
    def clear_cache(self):
        """Nettoie le cache Python"""
        print("\n🧹 NETTOYAGE DU CACHE")
        print("=" * 25)
        
        # Supprimer __pycache__
        cache_dirs = [
            self.ultra_path / "nn" / "modules" / "__pycache__",
            self.project_root / "ultralytics" / "nn" / "modules" / "__pycache__"
        ]
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                print(f"✅ Supprimé: {cache_dir.name}")
        
        # Nettoyer sys.modules
        modules_to_clear = [k for k in list(sys.modules.keys()) if 'ultralytics' in k or 'enhanced' in k]
        for module in modules_to_clear:
            sys.modules.pop(module, None)
        
        print(f"✅ {len(modules_to_clear)} modules retirés du cache")
    
    def test_import(self):
        """Test l'import des modules"""
        print("\n🧪 TEST D'IMPORT")
        print("=" * 20)
        
        # Forcer le rechargement
        if 'ultralytics' in sys.modules:
            del sys.modules['ultralytics']
        
        try:
            # Importer et tester
            from ultralytics.nn.modules.enhanced import A2Module, RELAN
            print("✅ Import enhanced réussi")
            
            # Test d'instanciation
            a2 = A2Module(64, 64)
            relan = RELAN(128, 128)
            print("✅ Modules fonctionnels")
            
            return True
        except Exception as e:
            print(f"❌ Erreur: {e}")
            
            # Essayer une méthode alternative
            print("\n🔧 Tentative de correction alternative...")
            try:
                # Importer directement le fichier
                import importlib.util
                enhanced_path = self.ultra_path / "nn" / "modules" / "enhanced.py"
                
                if enhanced_path.exists():
                    spec = importlib.util.spec_from_file_location("enhanced", enhanced_path)
                    enhanced = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(enhanced)
                    
                    # Ajouter au sys.modules
                    sys.modules['ultralytics.nn.modules.enhanced'] = enhanced
                    
                    # Tester
                    A2Module = enhanced.A2Module
                    RELAN = enhanced.RELAN
                    
                    a2 = A2Module(64, 64)
                    relan = RELAN(128, 128)
                    
                    print("✅ Import alternatif réussi")
                    print("⚠️  Note: L'import via __init__.py a échoué, mais le module est fonctionnel")
                    return True
                else:
                    print("❌ Fichier enhanced.py non trouvé")
                    return False
                    
            except Exception as e2:
                print(f"❌ Échec de l'import alternatif: {e2}")
                return False
    
    def run(self):
        """Exécute le processus complet de restauration"""
        print("🚀 YOLOv12-Face Enhanced - Restauration Optimisée")
        print("=" * 60)
        
        # 1. Vérifier l'environnement
        if not self.check_environment():
            if not self.install_ultralytics():
                print("\n❌ Impossible d'installer Ultralytics")
                return False
        
        # 2. Restaurer les configurations
        self.restore_configs()
        
        # 3. Mettre à jour __init__.py (avec gestion d'erreur améliorée)
        init_success = self.update_init_file()
        if not init_success:
            print("⚠️  Mise à jour de __init__.py échouée, mais continuons...")
        
        # 4. Nettoyer le cache
        self.clear_cache()
        
        # 5. Tester l'import
        import_ok = self.test_import()
        
        # 6. Résumé
        print("\n📊 RÉSUMÉ")
        print("=" * 20)
        print(f"✅ Fichiers restaurés: {self.success_count}/{self.total_count}")
        print(f"{'✅' if import_ok else '❌'} Import fonctionnel")
        
        if import_ok:
            print("\n🎉 RESTAURATION RÉUSSIE !")
            print("\n📋 Le modèle Enhanced est prêt à être utilisé")
            print("⚠️  Si l'import via __init__.py échoue, utilisez l'import direct:")
            print("    import sys")
            print("    sys.path.append('ultralytics/nn/modules')")
            print("    from enhanced import A2Module, RELAN")
        else:
            print("\n❌ Problème d'import persistant")
            print("💡 Solution manuelle:")
            print("1. Vérifiez que ultralytics est installé: pip install ultralytics")
            print("2. Copiez manuellement enhanced.py dans ultralytics/nn/modules/")
            print("3. Utilisez l'import direct comme indiqué ci-dessus")
        
        return import_ok


if __name__ == "__main__":
    restorer = ConfigRestorer()
    restorer.run()
