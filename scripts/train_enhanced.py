#!/usr/bin/env python3
"""
YOLOv12-Face Enhanced Training Script
Entraînement du modèle YOLOv12-Face avec modules d'attention avancés
"""

import os
import sys
import argparse
import torch
import time
import json
from pathlib import Path
from datetime import datetime

# Ajouter le répertoire parent au path pour importer ultralytics
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def parse_args():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='YOLOv12-Face Enhanced Training')
    parser.add_argument('--data', type=str, default='datasets/widerface/data.yaml',
                        help='Path to dataset YAML')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (0 for GPU, cpu for CPU)')
    parser.add_argument('--project', type=str, default='runs/face',
                        help='Project name')
    parser.add_argument('--name', type=str, default='yolov12-face-enhanced',
                        help='Experiment name')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with baseline model')
    return parser.parse_args()


def check_enhanced_modules():
    """Vérifie que les modules enhanced sont disponibles"""
    try:
        from ultralytics.nn.modules.enhanced import (
            A2Module, RELAN, FlashAttention, 
            CrossScaleAttention, MicroExpressionAttention
        )
        print("✅ Modules Enhanced trouvés et importés avec succès")
        return True
    except ImportError as e:
        print(f"⚠️ Modules Enhanced non trouvés: {e}")
        print("Tentative d'ajout des modules au système...")
        
        # Ajouter les modules à __init__.py
        init_path = parent_dir / 'ultralytics' / 'nn' / 'modules' / '__init__.py'
        if init_path.exists():
            with open(init_path, 'r') as f:
                content = f.read()
            
            # Ajouter les imports si nécessaire
            if 'enhanced' not in content:
                imports_to_add = """
# Enhanced modules for YOLOv12-Face
try:
    from .enhanced import (
        A2Module, RELAN, FlashAttention,
        CrossScaleAttention, MicroExpressionAttention
    )
    __all__.extend(['A2Module', 'RELAN', 'FlashAttention', 
                    'CrossScaleAttention', 'MicroExpressionAttention'])
except ImportError:
    pass  # Les modules enhanced sont optionnels
"""
                with open(init_path, 'a') as f:
                    f.write(imports_to_add)
                print("✅ Modules ajoutés à __init__.py")
        
        return False


def train_baseline(args):
    """Entraîne le modèle de base pour comparaison"""
    print("\n🔄 Entraînement du modèle de base (baseline)...")
    
    # Configuration du modèle de base
    cfg_path = parent_dir / 'ultralytics' / 'cfg' / 'models' / 'v12' / 'yolov12-face.yaml'
    data_path = parent_dir / args.data
    
    # Créer le modèle
    model = YOLO(str(cfg_path))
    
    # Entraîner le modèle
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        project=args.project,
        name=f"{args.name}_baseline",
        patience=args.patience,
        workers=args.workers,
        verbose=True,
        exist_ok=True
    )
    
    return results, model


def train_enhanced(args):
    """Entraîne le modèle enhanced avec modules d'attention"""
    print("\n🚀 Entraînement du modèle Enhanced...")
    
    # Configuration du modèle enhanced
    cfg_path = parent_dir / 'ultralytics' / 'cfg' / 'models' / 'v12' / 'yolov12-face-enhanced.yaml'
    data_path = parent_dir / args.data
    
    # Vérifier que le fichier existe
    if not cfg_path.exists():
        print(f"❌ Fichier de configuration non trouvé: {cfg_path}")
        return None, None
    
    # Créer le modèle
    model = YOLO(str(cfg_path))
    
    # Entraîner le modèle
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        project=args.project,
        name=f"{args.name}_enhanced",
        patience=args.patience,
        workers=args.workers,
        verbose=True,
        exist_ok=True,
        resume=args.resume
    )
    
    return results, model


def compare_models(baseline_model, enhanced_model, args):
    """Compare les performances des deux modèles"""
    print("\n📊 Comparaison des modèles...")
    
    # Créer le dossier de résultats
    results_dir = parent_dir / 'results' / f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger les données de validation
    data_path = parent_dir / args.data
    
    # Évaluer les modèles
    print("Évaluation du modèle baseline...")
    baseline_metrics = baseline_model.val(data=str(data_path))
    
    print("Évaluation du modèle enhanced...")
    enhanced_metrics = enhanced_model.val(data=str(data_path))
    
    # Créer un rapport de comparaison
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'baseline': {
            'mAP50': float(baseline_metrics.box.map50),
            'mAP50-95': float(baseline_metrics.box.map),
            'precision': float(baseline_metrics.box.p),
            'recall': float(baseline_metrics.box.r),
        },
        'enhanced': {
            'mAP50': float(enhanced_metrics.box.map50),
            'mAP50-95': float(enhanced_metrics.box.map),
            'precision': float(enhanced_metrics.box.p),
            'recall': float(enhanced_metrics.box.r),
        },
        'improvements': {
            'mAP50': float(enhanced_metrics.box.map50 - baseline_metrics.box.map50),
            'mAP50-95': float(enhanced_metrics.box.map - baseline_metrics.box.map),
            'precision': float(enhanced_metrics.box.p - baseline_metrics.box.p),
            'recall': float(enhanced_metrics.box.r - baseline_metrics.box.r),
        }
    }
    
    # Sauvegarder le rapport JSON
    with open(results_dir / 'comparison_report.json', 'w') as f:
        json.dump(comparison, f, indent=4)
    
    # Créer des graphiques de comparaison
    create_comparison_plots(comparison, results_dir)
    
    # Test de latence
    test_inference_speed(baseline_model, enhanced_model, results_dir, args)
    
    # Afficher le résumé
    print("\n" + "="*60)
    print("📈 RÉSUMÉ DE LA COMPARAISON")
    print("="*60)
    print(f"\n{'Métrique':<20} {'Baseline':<15} {'Enhanced':<15} {'Amélioration':<15}")
    print("-"*60)
    
    for metric in ['mAP50', 'mAP50-95', 'precision', 'recall']:
        baseline_val = comparison['baseline'][metric]
        enhanced_val = comparison['enhanced'][metric]
        improvement = comparison['improvements'][metric]
        improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
        
        print(f"{metric:<20} {baseline_val:<15.4f} {enhanced_val:<15.4f} "
              f"{improvement:+.4f} ({improvement_pct:+.1f}%)")
    
    print("="*60)
    print(f"\n📁 Résultats sauvegardés dans: {results_dir}")
    
    return comparison


def create_comparison_plots(comparison, results_dir):
    """Crée des graphiques de comparaison"""
    metrics = ['mAP50', 'mAP50-95', 'precision', 'recall']
    baseline_values = [comparison['baseline'][m] for m in metrics]
    enhanced_values = [comparison['enhanced'][m] for m in metrics]
    
    # Graphique en barres
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='#3498db')
    bars2 = ax.bar(x + width/2, enhanced_values, width, label='Enhanced', color='#2ecc71')
    
    # Ajouter les valeurs sur les barres
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
    
    ax.set_xlabel('Métriques')
    ax.set_ylabel('Valeur')
    ax.set_title('Comparaison YOLOv12-Face: Baseline vs Enhanced')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'comparison_bars.png', dpi=300)
    plt.close()
    
    # Graphique radar
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    baseline_values += baseline_values[:1]
    enhanced_values += enhanced_values[:1]
    angles += angles[:1]
    
    ax.plot(angles, baseline_values, 'o-', linewidth=2, label='Baseline', color='#3498db')
    ax.fill(angles, baseline_values, alpha=0.25, color='#3498db')
    
    ax.plot(angles, enhanced_values, 'o-', linewidth=2, label='Enhanced', color='#2ecc71')
    ax.fill(angles, enhanced_values, alpha=0.25, color='#2ecc71')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Radar Chart', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'comparison_radar.png', dpi=300)
    plt.close()


def test_inference_speed(baseline_model, enhanced_model, results_dir, args):
    """Test la vitesse d'inférence des modèles"""
    print("\n⏱️ Test de vitesse d'inférence...")
    
    # Créer une image de test
    dummy_img = torch.randn(1, 3, args.imgsz, args.imgsz)
    if args.device != 'cpu':
        dummy_img = dummy_img.cuda()
    
    # Warmup
    for _ in range(10):
        baseline_model(dummy_img, verbose=False)
        enhanced_model(dummy_img, verbose=False)
    
    # Test baseline
    baseline_times = []
    for _ in tqdm(range(100), desc="Baseline inference"):
        start = time.time()
        baseline_model(dummy_img, verbose=False)
        baseline_times.append((time.time() - start) * 1000)  # ms
    
    # Test enhanced
    enhanced_times = []
    for _ in tqdm(range(100), desc="Enhanced inference"):
        start = time.time()
        enhanced_model(dummy_img, verbose=False)
        enhanced_times.append((time.time() - start) * 1000)  # ms
    
    # Calculer les statistiques
    speed_report = {
        'baseline': {
            'mean_ms': np.mean(baseline_times),
            'std_ms': np.std(baseline_times),
            'min_ms': np.min(baseline_times),
            'max_ms': np.max(baseline_times),
            'fps': 1000 / np.mean(baseline_times)
        },
        'enhanced': {
            'mean_ms': np.mean(enhanced_times),
            'std_ms': np.std(enhanced_times),
            'min_ms': np.min(enhanced_times),
            'max_ms': np.max(enhanced_times),
            'fps': 1000 / np.mean(enhanced_times)
        }
    }
    
    # Sauvegarder le rapport
    with open(results_dir / 'speed_report.json', 'w') as f:
        json.dump(speed_report, f, indent=4)
    
    # Afficher les résultats
    print(f"\n{'Modèle':<15} {'Latence Moy (ms)':<20} {'FPS':<10}")
    print("-"*45)
    print(f"{'Baseline':<15} {speed_report['baseline']['mean_ms']:<20.2f} "
          f"{speed_report['baseline']['fps']:<10.1f}")
    print(f"{'Enhanced':<15} {speed_report['enhanced']['mean_ms']:<20.2f} "
          f"{speed_report['enhanced']['fps']:<10.1f}")
    
    # Créer un graphique de distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(baseline_times, bins=30, alpha=0.7, label='Baseline', color='#3498db')
    ax.hist(enhanced_times, bins=30, alpha=0.7, label='Enhanced', color='#2ecc71')
    ax.axvline(np.mean(baseline_times), color='#2980b9', linestyle='--', 
               label=f'Baseline Mean: {np.mean(baseline_times):.2f}ms')
    ax.axvline(np.mean(enhanced_times), color='#27ae60', linestyle='--',
               label=f'Enhanced Mean: {np.mean(enhanced_times):.2f}ms')
    
    ax.set_xlabel('Latence (ms)')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution des Latences d\'Inférence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'latency_distribution.png', dpi=300)
    plt.close()


def main():
    """Fonction principale"""
    args = parse_args()
    
    print("🚀 YOLOv12-Face Enhanced Training Script")
    print("="*60)
    print(f"📁 Projet: {args.project}")
    print(f"🏷️  Nom: {args.name}")
    print(f"📊 Dataset: {args.data}")
    print(f"🔢 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📐 Image size: {args.imgsz}")
    print(f"🖥️  Device: {args.device}")
    print("="*60)
    
    # Vérifier les modules enhanced
    modules_ok = check_enhanced_modules()
    if not modules_ok:
        print("\n⚠️ Les modules enhanced ne sont pas disponibles.")
        print("Assurez-vous que le fichier 'enhanced.py' est dans ultralytics/nn/modules/")
        if not args.compare:
            return
    
    # Si on veut comparer les modèles
    if args.compare:
        # Entraîner le modèle de base
        baseline_results, baseline_model = train_baseline(args)
        
        # Entraîner le modèle enhanced
        enhanced_results, enhanced_model = train_enhanced(args)
        
        if baseline_model and enhanced_model:
            # Comparer les modèles
            compare_models(baseline_model, enhanced_model, args)
    else:
        # Entraîner seulement le modèle enhanced
        enhanced_results, enhanced_model = train_enhanced(args)
        
        if enhanced_model:
            print("\n✅ Entraînement terminé avec succès!")
            print(f"📁 Modèle sauvegardé dans: {args.project}/{args.name}_enhanced/")


if __name__ == '__main__':
    main()
