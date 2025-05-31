#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Changer vers le bon répertoire
project_dir = Path(r"C:\Users\cedric\Desktop\box\01-Projects\Face-Recognition\yolov12-face")
os.chdir(project_dir)
sys.path.insert(0, str(project_dir))

print(f"Working directory: {os.getcwd()}")

# Test rapide
try:
    from ultralytics.nn.modules.enhanced import A2Module, RELAN
    print("✅ Import successful!")
    
    import torch
    x = torch.randn(1, 64, 32, 32)
    a2 = A2Module(64, 64)
    out = a2(x)
    print(f"✅ A2Module test: {x.shape} -> {out.shape}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
