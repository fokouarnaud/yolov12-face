import os

# Chemins des fichiers à supprimer
files_to_delete = [
    r"C:\Users\cedric\Desktop\box\01-Projects\Face-Recognition\yolov12-face\ultralytics\nn\modules\enhanced.py",
    r"C:\Users\cedric\Desktop\box\01-Projects\Face-Recognition\yolov12-face\ultralytics\nn\modules\enhanced_v2.py"
]

for file_path in files_to_delete:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✓ Supprimé: {os.path.basename(file_path)}")
        else:
            print(f"✗ N'existe pas: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"✗ Erreur lors de la suppression de {os.path.basename(file_path)}: {str(e)}")
