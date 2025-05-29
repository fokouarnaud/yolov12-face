"""
Script de téléchargement et préparation des datasets WIDERFace
Compatible avec YOLOv12 et la structure Ultralytics
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
    
    # URLs officielles du dataset WIDERFace
    URLS = {
        'train_images': 'https://huggingface.co/datasets/wider_face/resolve/main/WIDER_train.zip',
        'val_images': 'https://huggingface.co/datasets/wider_face/resolve/main/WIDER_val.zip',
        'test_images': 'https://huggingface.co/datasets/wider_face/resolve/main/WIDER_test.zip',
        'annotations': 'http://shuoyang1213.me/WIDERFACE/support/wider_face_split.zip'
    }
    
    # URLs alternatives (Google Drive)
    GDRIVE_IDS = {
        'train_images': '0B6eKvaijfFUOQUUwd21EckhUbWs',
        'val_images': '0B6eKvaijfFUDd3dIRmpvSk8tLUk',
        'test_images': '0B6eKvaijfFUDbnF0ei1jMmhLT1U',
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
    
    def download_from_url(self, url: str, filename: str) -> bool:
        """
        Télécharge un fichier depuis une URL
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
        except Exception as e:
            print(f"❌ Erreur lors du téléchargement depuis {url}: {e}")
            return False
    
    def download_from_gdrive(self, file_id: str, filename: str) -> bool:
        """
        Télécharge depuis Google Drive
        """
        try:
            import gdown
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filename, quiet=False)
            return True
        except Exception as e:
            print(f"❌ Erreur Google Drive: {e}")
            print("💡 Installation de gdown...")
            os.system("pip install gdown")
            try:
                import gdown
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, filename, quiet=False)
                return True
            except:
                return False
    
    def download_dataset(self, use_gdrive: bool = False):
        """
        Télécharge tous les fichiers du dataset
        """
        print("📥 Téléchargement du dataset WIDERFace...")
        
        downloads = []
        
        # Déterminer les sources
        if use_gdrive:
            downloads = [
                (self.GDRIVE_IDS['train_images'], 'WIDER_train.zip', 'train'),
                (self.GDRIVE_IDS['val_images'], 'WIDER_val.zip', 'val'),
                (self.GDRIVE_IDS['annotations'], 'wider_face_split.zip', 'annotations')
            ]
        else:
            downloads = [
                (self.URLS['train_images'], 'WIDER_train.zip', 'train'),
                (self.URLS['val_images'], 'WIDER_val.zip', 'val'),
                (self.URLS['annotations'], 'wider_face_split.zip', 'annotations')
            ]
        
        # Télécharger chaque fichier
        for source, filename, split in downloads:
            filepath = self.output_dir / filename
            
            if filepath.exists():
                print(f"✅ {filename} existe déjà")
                continue
            
            print(f"\n📥 Téléchargement de {filename}...")
            
            # Essayer de télécharger
            if use_gdrive:
                success = self.download_from_gdrive(source, str(filepath))
            else:
                success = self.download_from_url(source, str(filepath))
            
            if not success:
                print(f"❌ Échec du téléchargement de {filename}")
                # Essayer l'autre méthode
                if not use_gdrive:
                    print("🔄 Tentative avec Google Drive...")
                    success = self.download_from_gdrive(self.GDRIVE_IDS[split + '_images'], str(filepath))
                
                if not success:
                    raise Exception(f"Impossible de télécharger {filename}")
            
            # Extraire le zip
            if filepath.exists() and filepath.suffix == '.zip':
                print(f"📦 Extraction de {filename}...")
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(self.output_dir)
                
                # Supprimer le zip après extraction
                filepath.unlink()
                print(f"✅ {filename} extrait avec succès")
    
    def convert_annotations(self):
        """
        Convertit les annotations WIDERFace au format YOLO
        """
        print("\n🔄 Conversion des annotations au format YOLO...")
        
        for split in ['train', 'val']:
            print(f"\n📝 Traitement du split '{split}'...")
            
            # Fichier d'annotations
            anno_file = self.output_dir / 'wider_face_split' / f'wider_face_{split}_bbx_gt.txt'
            
            if not anno_file.exists():
                print(f"❌ Fichier d'annotations non trouvé: {anno_file}")
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
                num_bbox = int(lines[i].strip())
                i += 1
                
                # Créer le fichier de labels YOLO
                label_file = self.dirs['labels'][split] / (img_dst.stem + '.txt')
                
                with open(label_file, 'w') as lf:
                    for j in range(num_bbox):
                        # Format WIDERFace: x1 y1 w h blur expression occlusion pose invalid
                        bbox_info = lines[i].strip().split()
                        i += 1
                        
                        if len(bbox_info) < 4:
                            continue
                        
                        x1, y1, bbox_w, bbox_h = map(float, bbox_info[:4])
                        
                        # Filtrer les bboxes invalides
                        if len(bbox_info) >= 9:
                            invalid = int(bbox_info[8])
                            if invalid == 1:
                                continue
                        
                        # Ignorer les bboxes trop petites
                        if bbox_w <= 0 or bbox_h <= 0 or bbox_w < 5 or bbox_h < 5:
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
                        num_faces += 1
                
                num_images += 1
                pbar.update(1)
            
            pbar.close()
            print(f"✅ Split '{split}': {num_images} images, {num_faces} faces")
    
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
            
            images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            labels = list(label_dir.glob('*.txt'))
            
            print(f"\n📊 Split '{split}':")
            print(f"   • Images: {len(images)}")
            print(f"   • Labels: {len(labels)}")
            
            total_images += len(images)
            total_labels += len(labels)
            
            # Vérifier quelques labels
            if labels:
                sample_label = labels[0]
                with open(sample_label, 'r') as f:
                    lines = f.readlines()
                print(f"   • Exemple ({sample_label.name}): {len(lines)} faces")
        
        print(f"\n📊 Total:")
        print(f"   • Images: {total_images}")
        print(f"   • Labels: {total_labels}")
        
        if total_images == total_labels and total_images > 0:
            print("\n✅ Dataset vérifié avec succès!")
            return True
        else:
            print("\n❌ Problème détecté dans le dataset")
            return False
    
    def run(self, use_gdrive: bool = False):
        """
        Exécute le pipeline complet
        """
        print("🚀 Préparation du dataset WIDERFace pour YOLOv12")
        print("="*60)
        
        # Télécharger
        self.download_dataset(use_gdrive)
        
        # Convertir
        self.convert_annotations()
        
        # Créer la configuration
        self.create_yaml_config()
        
        # Vérifier
        success = self.verify_dataset()
        
        if success:
            print("\n✅ Dataset WIDERFace prêt pour l'entraînement!")
            print(f"📁 Emplacement: {self.output_dir.absolute()}")
            print(f"📝 Configuration: {self.output_dir / 'data.yaml'}")
        else:
            print("\n❌ Erreurs détectées. Vérifiez les logs.")
        
        return success


def main():
    """
    Point d'entrée principal
    """
    parser = argparse.ArgumentParser(description='Préparer le dataset WIDERFace pour YOLOv12')
    parser.add_argument('--output', type=str, default='datasets/widerface',
                      help='Répertoire de sortie pour le dataset')
    parser.add_argument('--gdrive', action='store_true',
                      help='Utiliser Google Drive pour le téléchargement')
    
    args = parser.parse_args()
    
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
