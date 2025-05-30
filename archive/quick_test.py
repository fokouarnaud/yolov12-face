"""Test rapide des imports Enhanced"""

print("🧪 Test des imports Enhanced...")

try:
    # Test 1: Vérifier que le fichier enhanced.py existe
    import os
    enhanced_path = "ultralytics/nn/modules/enhanced.py"
    if os.path.exists(enhanced_path):
        print("✅ Fichier enhanced.py trouvé")
    else:
        print("❌ Fichier enhanced.py manquant")
        
    # Test 2: Import direct du module
    import sys
    sys.path.insert(0, "ultralytics/nn/modules")
    
    import enhanced
    print("✅ Import direct du module enhanced réussi")
    
    # Test 3: Test des classes
    a2 = enhanced.A2Module(64, 64)
    relan = enhanced.RELAN(128, 128)
    print("✅ Classes A2Module et RELAN instanciées")
    
    # Test 4: Import via ultralytics
    try:
        from ultralytics.nn.modules.enhanced import A2Module, RELAN
        print("✅ Import via ultralytics.nn.modules.enhanced réussi")
    except ImportError as e:
        print(f"❌ Import via ultralytics échoué: {e}")
        
        # Vérifier __init__.py
        with open("ultralytics/nn/modules/__init__.py", "r") as f:
            content = f.read()
        
        if "from .enhanced import *" in content:
            print("✅ Import enhanced présent dans __init__.py")
        else:
            print("❌ Import enhanced manquant dans __init__.py")
            
except Exception as e:
    print(f"❌ Erreur générale: {e}")

print("\n🔧 Pour corriger, exécutez:")
print("python scripts/test_enhanced.py")
