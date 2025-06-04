"""Test des modules Enhanced avec différentes configurations de canaux"""

import torch
import sys
from pathlib import Path

# Ajouter le répertoire au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics.nn.modules.enhanced import A2Module, RELAN

print("Test des modules Enhanced avec différentes tailles de canaux")
print("="*60)

# Test avec différentes tailles de canaux
channel_configs = [
    (64, 64),    # Petit nombre de canaux
    (128, 256),  # Configuration asymétrique
    (256, 256),  # Configuration normale
    (512, 512),  # Grande taille
    (3, 64),     # Très petit (entrée RGB)
]

print("\nTest A2Module:")
print("-"*40)
for in_ch, out_ch in channel_configs:
    try:
        module = A2Module(in_ch, out_ch, n=1)
        x = torch.randn(1, in_ch, 32, 32)
        out = module(x)
        print(f"✅ {in_ch} -> {out_ch}: Input {x.shape} -> Output {out.shape}")
        
        # Vérifier les dimensions internes
        mid_channels = max(1, in_ch // 16)
        print(f"   Canaux intermédiaires: {mid_channels}")
        
    except Exception as e:
        print(f"❌ {in_ch} -> {out_ch}: Erreur - {e}")

print("\nTest RELAN:")
print("-"*40)
for in_ch, out_ch in channel_configs:
    try:
        module = RELAN(in_ch, out_ch, n=1)
        x = torch.randn(1, in_ch, 32, 32)
        out = module(x)
        print(f"✅ {in_ch} -> {out_ch}: Input {x.shape} -> Output {out.shape}")
        
        # Vérifier les dimensions internes
        branch_channels = max(1, out_ch // 4)
        print(f"   Canaux par branche: {branch_channels}, Total concat: {branch_channels * 4}")
        
    except Exception as e:
        print(f"❌ {in_ch} -> {out_ch}: Erreur - {e}")

print("\nTest du modèle YOLO complet:")
print("-"*40)
try:
    from ultralytics import YOLO
    
    config_path = Path(__file__).parent / "ultralytics" / "cfg" / "models" / "v12" / "yolov12-face-enhanced.yaml"
    model = YOLO(str(config_path))
    print("✅ Modèle chargé avec succès!")
    
    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        _ = model.model(x)
    print("✅ Forward pass réussi!")
    
except Exception as e:
    print(f"❌ Erreur modèle: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Tests terminés!")
