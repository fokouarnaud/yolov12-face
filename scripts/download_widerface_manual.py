"""
Script de téléchargement manuel du dataset WIDERFace
Utilise wget ou curl pour télécharger depuis les sources alternatives
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def download_with_wget(url, output_file):
    """Télécharge avec wget"""
    cmd = ['wget', '-c', '-O', output_file, url]
    return subprocess.run(cmd).returncode == 0


def download_with_curl(url, output_file):
    """Télécharge avec curl"""
    cmd = ['curl', '-L', '-o', output_file, url]
    return subprocess.run(cmd).returncode == 0


def download_with_python(url, output_file):
    """Télécharge avec Python requests"""
    try:
        import requests
        from tqdm import tqdm
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_file, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_file) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Erreur: {e}")
        return False


def main():
    # Créer le dossier de sortie
    output_dir = Path('datasets/widerface_temp')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 Téléchargement manuel du dataset WIDERFace")
    print("="*60)
    
    # URLs alternatives
    files = {
        'WIDER_train.zip': [
            'https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip',
            'https://drive.google.com/uc?export=download&id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M',
        ],
        'WIDER_val.zip': [
            'https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip',
            'https://drive.google.com/uc?export=download&id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q',
        ],
        'wider_face_split.zip': [
            'https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip',
            'https://drive.google.com/uc?export=download&id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q',
        ]
    }
    
    # Déterminer la méthode de téléchargement
    system = platform.system().lower()
    
    for filename, urls in files.items():
        output_file = str(output_dir / filename)
        
        if Path(output_file).exists():
            print(f"✅ {filename} existe déjà")
            continue
        
        print(f"\n📥 Téléchargement de {filename}...")
        
        success = False
        for url in urls:
            print(f"Tentative avec: {url[:50]}...")
            
            # Essayer différentes méthodes
            if system != 'windows' and subprocess.run(['which', 'wget'], capture_output=True).returncode == 0:
                success = download_with_wget(url, output_file)
            elif system != 'windows' and subprocess.run(['which', 'curl'], capture_output=True).returncode == 0:
                success = download_with_curl(url, output_file)
            else:
                success = download_with_python(url, output_file)
            
            if success:
                print(f"✅ Téléchargé avec succès")
                break
        
        if not success:
            print(f"❌ Impossible de télécharger {filename}")
            print(f"💡 Téléchargez manuellement depuis:")
            print(f"   http://shuoyang1213.me/WIDERFACE/")
            print(f"   Et placez le fichier dans: {output_dir}")
    
    print("\n" + "="*60)
    print("✅ Téléchargements terminés")
    print(f"📁 Fichiers dans: {output_dir}")
    print("\n💡 Maintenant, exécutez:")
    print("   python scripts/prepare_widerface.py")


if __name__ == '__main__':
    main()
