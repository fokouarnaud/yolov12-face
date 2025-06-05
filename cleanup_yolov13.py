"""
Script de nettoyage pour supprimer les imports YOLOv13 et fichiers inutiles
"""
import os
import re

def clean_tasks_py():
    """Nettoie le fichier tasks.py en supprimant les imports YOLOv13"""
    tasks_path = r"C:\Users\cedric\Desktop\box\01-Projects\Face-Recognition\yolov12-face\ultralytics\nn\tasks.py"
    
    # Lire le fichier
    with open(tasks_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Supprimer la section d'import YOLOv13
    start_marker = "# Import des modules YOLOv13 après LOGGER"
    end_marker = 'LOGGER.warning(f"WARNING ⚠️ YOLOv13 modules not found: {e}")'
    
    # Trouver les positions
    start_pos = content.find(start_marker)
    if start_pos != -1:
        end_pos = content.find(end_marker)
        if end_pos != -1:
            # Inclure la ligne complète après end_marker
            end_pos = content.find('\n', end_pos) + 1
            # Supprimer toute la section
            content = content[:start_pos] + content[end_pos:]
    
    # Supprimer aussi les références dans parse_model
    # Supprimer les elif pour les modules YOLOv13
    patterns_to_remove = [
        r'elif m is TripletFaceAttention:.*?args = \[c1, \*args\].*?\n',
        r'elif m is EfficientFaceTransformer:.*?args = \[c1, \*args\].*?\n',
        r'elif m is NeuralArchitectureSearchBlock:.*?args = \[c1, \*args\].*?\n',
        r'elif m is MixtureOfExpertsBlock:.*?args = \[c1, \*args\].*?\n',
        r'elif m is C2fTransformer:.*?args = \[c1, \*args\].*?\n',
        r'elif m is FaceDetect:.*?c2 = args\[0\] \* 5.*?\n',
    ]
    
    for pattern in patterns_to_remove:
        content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Écrire le fichier nettoyé
    with open(tasks_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Fichier tasks.py nettoyé")

def remove_yolov13_files():
    """Supprime les fichiers YOLOv13 inutiles"""
    files_to_remove = [
        r"C:\Users\cedric\Desktop\box\01-Projects\Face-Recognition\yolov12-face\ultralytics\nn\modules\yolov13_face.py",
        r"C:\Users\cedric\Desktop\box\01-Projects\Face-Recognition\yolov12-face\ultralytics\nn\modules\yolov13_modules.py",
        r"C:\Users\cedric\Desktop\box\01-Projects\Face-Recognition\yolov12-face\ultralytics\nn\modules\enhanced.py",
        r"C:\Users\cedric\Desktop\box\01-Projects\Face-Recognition\yolov12-face\ultralytics\nn\modules\enhanced_v2.py"
    ]
    
    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✓ Supprimé: {os.path.basename(file_path)}")
            else:
                print(f"✗ N'existe pas: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"✗ Erreur: {e}")

if __name__ == "__main__":
    print("Nettoyage des fichiers YOLOv13...")
    clean_tasks_py()
    remove_yolov13_files()
    print("\nNettoyage terminé!")
