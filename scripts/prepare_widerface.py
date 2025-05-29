"""
Script de téléchargement et préparation des datasets WIDERFace
Compatible avec YOLOv12 et la structure Ultralytics
Version mise à jour avec les liens corrects
"""

import os
import sys
import zipfile
import requests
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import yaml
import argparse


class WIDERFaceDownloader:
    """
    Télécharge et prépare le dataset WIDERFace pour YOLOv12
    """
    
    # URLs mises à jour (décembre 2024)
    URLS = {
        # HuggingFace (utilise resolve au lieu de blob)
        'train_images': 'https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip',
        'val_images': 'https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip',
        'test_images': 'https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_test.zip',
        'annotations': 'https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip',
        
        # URLs alternatives directes depuis le site officiel
        'train_gdrive_new': 'https://drive.google.com/file/d/15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M/view?usp=sharing',
        'val_gdrive_new': 'https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view?usp=sharing',
        'anno_gdrive_new': 'https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view?usp=sharing'
    }
    
    # Google Drive IDs extraits des nouveaux liens
    GDRIVE_IDS = {
        'train_images': '15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M',
        'val_images': '1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q',
        'test_images': '1HIfDbVEWKmsYKJZm4lchTBDLW5N7dY5T',
        'annotations': '1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q'
    }
    
    def __init__(self, output_dir: str = 'datasets/widerface'):
        """
        Args:
            output_dir: Répertoire de sortie pour le dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer la structure de dossiers
        self.dirs = {
            'images': {
                'train': self.output_dir / 'images' / 'train',
                'val': self.output_dir / 'images' / 'val',
                'test': self.output_dir / 'images' / 'test'
            },
            'labels': {
                'train': self.output_dir / 'labels' / 'train',
                'val': self.output_dir / 'labels' / 'val',
                'test': self.output_dir / 'labels' / 'test'
            }
        }
        
        # Créer tous les dossiers
        for split_dirs in self.dirs.values():
            for dir_path in split_dirs.values():
                dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_from_url(self, url: str, filename: str, max_retries: int = 3) -> bool:
        """
        Télécharge un fichier depuis une URL avec retry
        """
        for attempt in range(max_retries):
            try:
                print(f"Tentative {attempt + 1}/{max_retries}...")
                
                # Headers pour éviter les blocages
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(url, stream=True, headers=headers, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filename, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                # Vérifier la taille du fichier
                if Path(filename).stat().st_size > 1000:  # Au moins 1KB
                    return True
                else:
                    print(f"⚠️ Fichier trop petit, nouvelle tentative...")
                    Path(filename).unlink()
                    
            except Exception as e:
                print(f"⚠️ Tentative {attempt + 1} échouée: {str(e)}")
                if Path(filename).exists():
                    Path(filename).unlink()
                
                if attempt < max_retries - 1:
                    print("⏳ Attente de 5 secondes avant nouvelle tentative...")
                    import time
                    time.sleep(5)
        
        return False
    
    def download_from_gdrive(self, file_id: str, filename: str) -> bool:
        """
        Télécharge depuis Google Drive avec gdown
        """
        try:
            import gdown
            
            # URL avec paramètres pour éviter les confirmations
            url = f'https://drive.google.com/uc?export=download&id={file_id}'
            
            # Télécharger avec gdown
            output = gdown.download(url, filename, quiet=False, fuzzy=True)
            
            if output and Path(output).exists() and Path(output).stat().st_size > 1000:
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Erreur Google Drive: {e}")
            
            # Essayer la méthode alternative
            try:
                # Construction de l'URL alternative
                alt_url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
                return self.download_from_url(alt_url, filename)
            except:
                return False
    
    def download_dataset(self, use_gdrive: bool = False):
        """
        Télécharge tous les fichiers du dataset
        """
        print("📥 Téléchargement du dataset WIDERFace...")
        
        # Fichiers à télécharger
        files_to_download = [
            ('train_images', 'WIDER_train.zip', 'train'),
            ('val_images', 'WIDER_val.zip', 'val'),
            ('annotations', 'wider_face_split.zip', 'annotations')
        ]
        
        success_count = 0
        
        for file_key, filename, split in files_to_download:
            filepath = self.output_dir / filename
            
            # Vérifier si le fichier existe déjà et a une taille raisonnable
            if filepath.exists() and filepath.stat().st_size > 1000000:  # > 1MB
                print(f"✅ {filename} existe déjà")
                success_count += 1
                continue
            
            print(f"\n📥 Téléchargement de {filename}...")
            success = False
            
            # Essayer HuggingFace d'abord (sauf si gdrive forcé)
            if not use_gdrive and file_key in self.URLS:
                print("🔄 Tentative via HuggingFace...")
                success = self.download_from_url(self.URLS[file_key], str(filepath))
            
            # Si échec ou gdrive forcé, essayer Google Drive
            if not success and file_key in self.GDRIVE_IDS:
                print("🔄 Tentative via Google Drive...")
                success = self.download_from_gdrive(self.GDRIVE_IDS[file_key], str(filepath))
            
            if success:
                success_count += 1
                print(f"✅ {filename} téléchargé avec succès")
            else:
                print(f"❌ Impossible de télécharger {filename}")
                print(f"💡 Vous pouvez télécharger manuellement depuis:")
                print(f"   - http://shuoyang1213.me/WIDERFACE/")
                print(f"   - https://huggingface.co/datasets/wider_face/tree/main/data")
        
        # Extraire les fichiers zip
        if success_count > 0:
            print("\n📦 Extraction des archives...")
            for file_key, filename, split in files_to_download:
                filepath = self.output_dir / filename
                if filepath.exists() and filepath.suffix == '.zip':
                    try:
                        print(f"   Extraction de {filename}...")
                        with zipfile.ZipFile(filepath, 'r') as zip_ref:
                            zip_ref.extractall(self.output_dir)
                        filepath.unlink()  # Supprimer le zip après extraction
                        print(f"   ✅ {filename} extrait")
                    except Exception as e:
                        print(f"   ❌ Erreur lors de l'extraction de {filename}: {e}")
        
        return success_count >= 2  # Au moins train et annotations
    
    def convert_annotations(self):
        """
        Convertit les annotations WIDERFace au format YOLO
        """
        print("\n🔄 Conversion des annotations au format YOLO...")
        
        converted_splits = 0
        
        for split in ['train', 'val']:
            print(f"\n📝 Traitement du split '{split}'...")
            
            # Fichier d'annotations
            anno_file = self.output_dir / 'wider_face_split' / f'wider_face_{split}_bbx_gt.txt'
            
            if not anno_file.exists():
                print(f"⚠️ Fichier d'annotations non trouvé: {anno_file}")
                # Essayer un autre emplacement
                anno_file = self.output_dir / f'wider_face_{split}_bbx_gt.txt'
                if not anno_file.exists():
                    continue
            
            # Lire les annotations
            with open(anno_file, 'r') as f:
                lines = f.readlines()
            
            i = 0
            num_images = 0
            num_faces = 0
            
            pbar = tqdm(total=len(lines), desc=f"Conversion {split}")
            
            while i < len(lines):
                # Nom de l'image
                img_name = lines[i].strip()
                i += 1
                
                if not img_name or i >= len(lines):
                    pbar.update(1)
                    continue
                
                # Chemin de l'image source
                img_src = self.output_dir / f'WIDER_{split}' / 'images' / img_name
                
                if not img_src.exists():
                    pbar.update(1)
                    continue
                
                # Lire l'image pour obtenir les dimensions
                img = cv2.imread(str(img_src))
                if img is None:
                    pbar.update(1)
                    continue
                
                h, w = img.shape[:2]
                
                # Copier l'image vers le dossier de destination
                img_dst = self.dirs['images'][split] / img_name.replace('/', '_')
                shutil.copy2(img_src, img_dst)
                
                # Nombre de faces
                try:
                    num_bbox = int(lines[i].strip())
                    i += 1
                except:
                    pbar.update(1)
                    continue
                
                # Créer le fichier de labels YOLO
                label_file = self.dirs['labels'][split] / (img_dst.stem + '.txt')
                
                valid_faces = 0
                with open(label_file, 'w') as lf:
                    for j in range(num_bbox):
                        if i >= len(lines):
                            break
                            
                        bbox_info = lines[i].strip().split()
                        i += 1
                        
                        if len(bbox_info) < 4:
                            continue
                        
                        x1, y1, bbox_w, bbox_h = map(float, bbox_info[:4])
                        
                        # Filtrer les bboxes invalides
                        if bbox_w <= 0 or bbox_h <= 0 or bbox_w < 5 or bbox_h < 5:
                            continue
                        
                        # Vérifier si c'est un visage valide (pas marqué comme invalide)
                        if len(bbox_info) >= 9:
                            invalid = int(bbox_info[8])
                            if invalid == 1:
                                continue
                        
                        # Convertir au format YOLO (normalisé)
                        x_center = (x1 + bbox_w / 2) / w
                        y_center = (y1 + bbox_h / 2) / h
                        width = bbox_w / w
                        height = bbox_h / h
                        
                        # Vérifier que les valeurs sont valides
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))
                        
                        # Écrire au format YOLO
                        lf.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        valid_faces += 1
                
                if valid_faces > 0:
                    num_images += 1
                    num_faces += valid_faces
                else:
                    # Supprimer l'image et le label s'il n'y a pas de faces valides
                    img_dst.unlink()
                    label_file.unlink()
                
                pbar.update(1)
            
            pbar.close()
            
            if num_images > 0:
                converted_splits += 1
                print(f"✅ Split '{split}': {num_images} images, {num_faces} faces")
            else:
                print(f"⚠️ Split '{split}': Aucune image convertie")
        
        return converted_splits > 0
    
    def create_yaml_config(self):
        """
        Crée le fichier de configuration YAML pour YOLOv12
        """
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            
            'nc': 1,  # number of classes
            'names': ['face'],
            
            # Informations sur le dataset
            'download': 'Dataset WIDERFace déjà téléchargé et converti',
            'dataset_info': {
                'description': 'WIDERFace dataset for face detection',
                'url': 'http://shuoyang1213.me/WIDERFACE/',
                'version': '1.0',
                'year': 2016,
                'contributor': 'WIDER FACE team',
                'date_created': '2016-01-01'
            }
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✅ Configuration YAML créée: {yaml_path}")
        
        # Créer aussi dans ultralytics/cfg/datasets/
        cfg_dir = Path(__file__).parent.parent / 'ultralytics' / 'cfg' / 'datasets'
        if cfg_dir.exists():
            shutil.copy2(yaml_path, cfg_dir / 'widerface.yaml')
            print(f"✅ Configuration copiée vers: {cfg_dir / 'widerface.yaml'}")
    
    def verify_dataset(self):
        """
        Vérifie l'intégrité du dataset
        """
        print("\n🔍 Vérification du dataset...")
        
        total_images = 0
        total_labels = 0
        
        for split in ['train', 'val']:
            img_dir = self.dirs['images'][split]
            label_dir = self.dirs['labels'][split]
            
            if not img_dir.exists() or not label_dir.exists():
                continue
            
            images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            labels = list(label_dir.glob('*.txt'))
            
            print(f"\n📊 Split '{split}':")
            print(f"   • Images: {len(images)}")
            print(f"   • Labels: {len(labels)}")
            
            total_images += len(images)
            total_labels += len(labels)
            
            # Vérifier quelques labels
            if labels and len(labels) > 0:
                sample_label = labels[0]
                with open(sample_label, 'r') as f:
                    lines = f.readlines()
                print(f"   • Exemple ({sample_label.name}): {len(lines)} faces")
        
        print(f"\n📊 Total:")
        print(f"   • Images: {total_images}")
        print(f"   • Labels: {total_labels}")
        
        return total_images > 0 and total_labels > 0
    
    def run(self, use_gdrive: bool = False):
        """
        Exécute le pipeline complet
        """
        print("🚀 Préparation du dataset WIDERFace pour YOLOv12")
        print("="*60)
        
        # Télécharger
        download_success = self.download_dataset(use_gdrive)
        
        if not download_success:
            print("\n❌ Échec du téléchargement")
            print("💡 Solutions alternatives:")
            print("1. Téléchargez manuellement depuis http://shuoyang1213.me/WIDERFACE/")
            print("2. Placez les fichiers dans:", self.output_dir)
            print("3. Relancez ce script")
            return False
        
        # Convertir
        convert_success = self.convert_annotations()
        
        if not convert_success:
            print("\n❌ Échec de la conversion")
            return False
        
        # Créer la configuration
        self.create_yaml_config()
        
        # Vérifier
        success = self.verify_dataset()
        
        if success:
            print("\n✅ Dataset WIDERFace prêt pour l'entraînement!")
            print(f"📁 Emplacement: {self.output_dir.absolute()}")
            print(f"📝 Configuration: {self.output_dir / 'data.yaml'}")
        else:
            print("\n⚠️ Dataset partiellement préparé. Vérifiez les logs.")
        
        return success


def main():
    """
    Point d'entrée principal
    """
    parser = argparse.ArgumentParser(description='Préparer le dataset WIDERFace pour YOLOv12')
    parser.add_argument('--output', type=str, default='datasets/widerface',
                      help='Répertoire de sortie pour le dataset')
    parser.add_argument('--gdrive', action='store_true',
                      help='Forcer l\'utilisation de Google Drive pour le téléchargement')
    
    args = parser.parse_args()
    
    # Installer gdown si nécessaire
    try:
        import gdown
    except ImportError:
        print("📦 Installation de gdown...")
        os.system(f"{sys.executable} -m pip install gdown")
    
    # Créer le downloader
    downloader = WIDERFaceDownloader(args.output)
    
    # Exécuter
    success = downloader.run(use_gdrive=args.gdrive)
    
    if success:
        print("\n💡 Pour entraîner YOLOv12:")
        print(f"   yolo detect train data={args.output}/data.yaml model=yolov12n.yaml")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
