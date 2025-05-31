#!/usr/bin/env python3
"""
Script de test pour diagnostiquer les problèmes d'import des modules Enhanced
"""

import sys
import os

# Ajouter le répertoire courant au path
sys.path.insert(0, os.getcwd())

print("=== Test d'Import des Modules Enhanced ===")
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}...")

try:
    print("\n1. Test import ultralytics...")
    import ultralytics
    print(f"✅ ultralytics importé - Version: {ultralytics.__version__}")
    
    print("\n2. Test import ultralytics.nn...")
    from ultralytics import nn
    print("✅ ultralytics.nn importé")
    
    print("\n3. Test import ultralytics.nn.modules...")
    from ultralytics.nn import modules
    print("✅ ultralytics.nn.modules importé")
    
    print("\n4. Test import enhanced directement...")
    from ultralytics.nn.modules import enhanced
    print("✅ ultralytics.nn.modules.enhanced importé")
    
    print("\n5. Test import A2Module et RELAN...")
    from ultralytics.nn.modules.enhanced import A2Module, RELAN
    print("✅ A2Module et RELAN importés directement")
    
    print("\n6. Test import via modules.__init__...")
    from ultralytics.nn.modules import A2Module, RELAN
    print("✅ A2Module et RELAN importés via modules")
    
    print("\n7. Test de création d'instances...")
    a2 = A2Module(64, 64)
    relan = RELAN(64, 64)
    print(f"✅ Instances créées - A2Module: {type(a2)}, RELAN: {type(relan)}")
    
    print("\n8. Test avec données factices...")
    import torch
    x = torch.randn(1, 64, 32, 32)
    
    # Test A2Module
    out_a2 = a2(x)
    print(f"✅ A2Module forward - Input: {x.shape}, Output: {out_a2.shape}")
    
    # Test RELAN
    out_relan = relan(x)
    print(f"✅ RELAN forward - Input: {x.shape}, Output: {out_relan.shape}")
    
    print("\n🎉 TOUS LES TESTS RÉUSSIS ! 🎉")
    
except Exception as e:
    print(f"❌ ERREUR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
