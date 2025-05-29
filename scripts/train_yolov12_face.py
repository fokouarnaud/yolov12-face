"""
Script d'entraînement YOLOv12-Face
Utilise la structure Ultralytics existante avec nos améliorations
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import argparse
from datetime import datetime

# Ajouter le chemin du repo au PYTHONPATH
repo_path = Path(__file__).parent.parent
sys.path.insert(0, str(repo_path))

# Import Ultralytics
try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER, colorstr
    from ultralytics.utils.checks import check_requirements
except ImportError:
    print("❌ Ultralytics non trouvé. Installation...")
    os.system(f"{sys.executable} -m pip install ultralytics")
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER, colorstr
    from ultralytics.utils.checks import check_requirements


def train_yolov12_face(
    model_config: str = 'yolov12n.yaml',
    data_config: str = 'datasets/widerface/data.yaml',
    epochs: int = 300,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = None,
    project: str = 'runs/train',
    name: str = None,
    pretrained: bool = True,
    resume: bool = False,
    patience: int = 50,
    workers: int = 8,
    optimizer: str = 'AdamW',
    lr0: float = 0.01,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    warmup_momentum: float = 0.8,
    warmup_bias_lr: float = 0.1,
    box: float = 0.05,
    cls: float = 0.5,
    dfl: float = 1.5,
    pose: float = 12.0,
    kobj: float = 1.0,
    label_smoothing: float = 0.0,
    nbs: int = 64,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    flipud: float = 0.0,
    fliplr: float = 0.5,
    mosaic: float = 1.0,
    mixup: float = 0.15,
    copy_paste: float = 0.0,
    save: bool = True,
    save_period: int = -1,
    cache: bool = False,
    imgsz: int = 640,
    rect: bool = False,
    cos_lr: bool = False,
    close_mosaic: int = 10,
    amp: bool = True,
    fraction: float = 1.0,
    profile: bool = False,
    overlap_mask: bool = True,
    mask_ratio: int = 4,
    dropout: float = 0.0,
    val: bool = True,
    save_json: bool = False,
    save_hybrid: bool = False,
    conf: float = None,
    iou: float = 0.7,
    max_det: int = 300,
    half: bool = False,
    dnn: bool = False,
    plots: bool = True,
    source: str = None,
    show: bool = False,
    save_txt: bool = False,
    save_conf: bool = False,
    save_crop: bool = False,
    hide_labels: bool = False,
    hide_conf: bool = False,
    vid_stride: int = 1,
    line_thickness: int = 3,
    visualize: bool = False,
    augment: bool = False,
    agnostic_nms: bool = False,
    classes: list = None,
    retina_masks: bool = False,
    boxes: bool = True,
    format: str = 'torchscript',
    keras: bool = False,
    optimize: bool = False,
    int8: bool = False,
    dynamic: bool = False,
    simplify: bool = False,
    opset: int = None,
    workspace: float = 4,
    nms: bool = False,
    lr_find: bool = False,
    freeze: list = None,
    save_dir: str = None
):
    """
    Entraîne YOLOv12 pour la détection de visages
    
    Args:
        Tous les paramètres d'entraînement YOLOv12
    """
    
    # Configuration du device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Nom automatique si non spécifié
    if name is None:
        name = f'yolov12-face-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    
    # Vérifier que le fichier de données existe
    data_path = Path(data_config)
    if not data_path.exists():
        # Chercher dans différents emplacements
        possible_paths = [
            Path('datasets/widerface/data.yaml'),
            Path('ultralytics/cfg/datasets/widerface.yaml'),
            Path(repo_path) / 'datasets' / 'widerface' / 'data.yaml',
            Path(repo_path) / 'ultralytics' / 'cfg' / 'datasets' / 'widerface.yaml'
        ]
        
        for p in possible_paths:
            if p.exists():
                data_path = p
                break
        else:
            raise FileNotFoundError(f"❌ Fichier de données non trouvé: {data_config}")
    
    # Vérifier la configuration du modèle
    model_path = Path(model_config)
    if not model_path.exists():
        # Chercher dans ultralytics/cfg/models/v12/
        possible_model_paths = [
            Path(f'ultralytics/cfg/models/v12/{model_config}'),
            Path(repo_path) / 'ultralytics' / 'cfg' / 'models' / 'v12' / model_config
        ]
        
        for p in possible_model_paths:
            if p.exists():
                model_path = p
                break
    
    print("🚀 Démarrage de l'entraînement YOLOv12-Face")
    print("="*60)
    print(f"📊 Configuration:")
    print(f"   • Modèle: {model_path}")
    print(f"   • Données: {data_path}")
    print(f"   • Epochs: {epochs}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Image size: {img_size}")
    print(f"   • Device: {device}")
    print(f"   • Project: {project}/{name}")
    print("="*60)
    
    # Charger le modèle
    if resume:
        # Reprendre l'entraînement
        model = YOLO(f'{project}/{name}/weights/last.pt')
        print("♻️ Reprise de l'entraînement depuis le dernier checkpoint")
    else:
        # Nouveau modèle
        model = YOLO(str(model_path))
        print(f"✅ Modèle chargé: {model_path}")
    
    # Arguments d'entraînement
    train_args = {
        'data': str(data_path),
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'project': project,
        'name': name,
        'pretrained': pretrained,
        'optimizer': optimizer,
        'verbose': True,
        'patience': patience,
        'workers': workers,
        'cos_lr': cos_lr,
        'lr0': lr0,
        'lrf': lrf,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'warmup_epochs': warmup_epochs,
        'warmup_momentum': warmup_momentum,
        'warmup_bias_lr': warmup_bias_lr,
        'box': box,
        'cls': cls,
        'dfl': dfl,
        'label_smoothing': label_smoothing,
        'nbs': nbs,
        'hsv_h': hsv_h,
        'hsv_s': hsv_s,
        'hsv_v': hsv_v,
        'degrees': degrees,
        'translate': translate,
        'scale': scale,
        'shear': shear,
        'perspective': perspective,
        'flipud': flipud,
        'fliplr': fliplr,
        'mosaic': mosaic,
        'mixup': mixup,
        'copy_paste': copy_paste,
        'amp': amp,
        'close_mosaic': close_mosaic,
        'resume': resume,
        'fraction': fraction,
        'profile': profile,
        'dropout': dropout,
        'val': val,
        'save': save,
        'save_period': save_period,
        'cache': cache,
        'plots': plots,
        'overlap_mask': overlap_mask,
        'mask_ratio': mask_ratio,
        'max_det': max_det,
        'save_json': save_json,
        'save_hybrid': save_hybrid,
        'conf': conf,
        'iou': iou,
        'exist_ok': True
    }
    
    # Filtrer les None
    train_args = {k: v for k, v in train_args.items() if v is not None}
    
    # Entraîner
    print("\n🏋️ Début de l'entraînement...")
    results = model.train(**train_args)
    
    print("\n✅ Entraînement terminé!")
    print(f"📁 Résultats sauvegardés dans: {project}/{name}")
    
    # Afficher les métriques finales
    if hasattr(results, 'results_dict'):
        print("\n📊 Métriques finales:")
        for key, value in results.results_dict.items():
            if isinstance(value, (int, float)):
                print(f"   • {key}: {value:.4f}")
    
    return model, results


def export_model(model_path: str, formats: list = None):
    """
    Exporte le modèle dans différents formats
    """
    if formats is None:
        formats = ['onnx', 'torchscript']
    
    print(f"\n📦 Export du modèle: {model_path}")
    
    # Charger le modèle
    model = YOLO(model_path)
    
    for fmt in formats:
        print(f"\n🔄 Export en {fmt.upper()}...")
        try:
            model.export(format=fmt)
            print(f"✅ Export {fmt} réussi")
        except Exception as e:
            print(f"❌ Erreur export {fmt}: {e}")


def validate_model(model_path: str, data_config: str = 'datasets/widerface/data.yaml'):
    """
    Valide le modèle sur le dataset
    """
    print(f"\n🔍 Validation du modèle: {model_path}")
    
    # Charger le modèle
    model = YOLO(model_path)
    
    # Valider
    results = model.val(data=data_config)
    
    print("\n📊 Résultats de validation:")
    if hasattr(results, 'results_dict'):
        for key, value in results.results_dict.items():
            if isinstance(value, (int, float)):
                print(f"   • {key}: {value:.4f}")
    
    return results


def main():
    """
    Point d'entrée principal
    """
    parser = argparse.ArgumentParser(description='Entraîner YOLOv12 pour la détection de visages')
    
    # Arguments principaux
    parser.add_argument('--model', type=str, default='yolov12n.yaml',
                      help='Configuration du modèle YOLOv12 (n/s/m/l/x)')
    parser.add_argument('--data', type=str, default='datasets/widerface/data.yaml',
                      help='Fichier de configuration du dataset')
    parser.add_argument('--epochs', type=int, default=300,
                      help='Nombre d\'epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Taille du batch')
    parser.add_argument('--img-size', type=int, default=640,
                      help='Taille des images')
    parser.add_argument('--device', type=str, default=None,
                      help='Device (cuda/cpu)')
    parser.add_argument('--project', type=str, default='runs/train',
                      help='Dossier du projet')
    parser.add_argument('--name', type=str, default=None,
                      help='Nom de l\'expérience')
    parser.add_argument('--pretrained', action='store_true', default=True,
                      help='Utiliser des poids pré-entraînés')
    parser.add_argument('--resume', action='store_true',
                      help='Reprendre l\'entraînement')
    
    # Hyperparamètres
    parser.add_argument('--patience', type=int, default=50,
                      help='Patience pour early stopping')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                      help='Optimiseur')
    parser.add_argument('--lr0', type=float, default=0.01,
                      help='Learning rate initial')
    parser.add_argument('--lrf', type=float, default=0.01,
                      help='Learning rate final')
    
    # Augmentations
    parser.add_argument('--mosaic', type=float, default=1.0,
                      help='Probabilité de mosaic')
    parser.add_argument('--mixup', type=float, default=0.15,
                      help='Probabilité de mixup')
    
    # Modes
    parser.add_argument('--mode', choices=['train', 'val', 'export'],
                      default='train', help='Mode d\'exécution')
    parser.add_argument('--weights', type=str, default=None,
                      help='Chemin des poids pour validation/export')
    parser.add_argument('--formats', nargs='+', default=['onnx', 'torchscript'],
                      help='Formats d\'export')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Entraîner
        model, results = train_yolov12_face(
            model_config=args.model,
            data_config=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
            project=args.project,
            name=args.name,
            pretrained=args.pretrained,
            resume=args.resume,
            patience=args.patience,
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            mosaic=args.mosaic,
            mixup=args.mixup
        )
        
        # Export automatique après entraînement
        best_weights = f"{args.project}/{args.name}/weights/best.pt"
        if Path(best_weights).exists():
            export_model(best_weights, formats=['onnx'])
    
    elif args.mode == 'val':
        # Valider
        if not args.weights:
            print("❌ --weights requis pour la validation")
            return
        validate_model(args.weights, args.data)
    
    elif args.mode == 'export':
        # Exporter
        if not args.weights:
            print("❌ --weights requis pour l'export")
            return
        export_model(args.weights, args.formats)


if __name__ == '__main__':
    main()
