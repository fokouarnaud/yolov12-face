#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de comparaison de performance pour YOLOv12-Face
Compare le modèle de base avec le modèle enhanced
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Ajouter le répertoire parent au path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from ultralytics import YOLO
from tqdm import tqdm


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description='YOLOv12-Face Performance Comparison')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Path to baseline model weights')
    parser.add_argument('--enhanced', type=str, required=True,
                        help='Path to enhanced model weights')
    parser.add_argument('--test-images', type=str, default='test_images',
                        help='Directory with test images')
    parser.add_argument('--output', type=str, default='comparison_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--save-images', action='store_true',
                        help='Save detection results')
    return parser.parse_args()


def load_models(baseline_path, enhanced_path, device):
    """Charge les deux modèles"""
    print("📥 Chargement des modèles...")
    
    # Charger le modèle baseline
    baseline_model = YOLO(baseline_path)
    baseline_model.to(device)
    
    # Charger le modèle enhanced
    enhanced_model = YOLO(enhanced_path)
    enhanced_model.to(device)
    
    return baseline_model, enhanced_model


def benchmark_speed(model, test_images, device, desc="Model"):
    """Benchmark la vitesse d'un modèle"""
    times = []
    
    # Warmup
    if len(test_images) > 0:
        for _ in range(5):
            model(test_images[0], verbose=False)
    
    # Test
    for img_path in tqdm(test_images, desc=f"Testing {desc}"):
        start = time.time()
        results = model(img_path, verbose=False)
        inference_time = (time.time() - start) * 1000  # ms
        times.append(inference_time)
    
    return times


def analyze_detections(baseline_model, enhanced_model, test_images, conf, iou, save_dir=None):
    """Analyse les détections des deux modèles"""
    baseline_stats = {
        'total_detections': 0,
        'avg_confidence': [],
        'detection_counts': []
    }
    
    enhanced_stats = {
        'total_detections': 0,
        'avg_confidence': [],
        'detection_counts': []
    }
    
    comparison_data = []
    
    for img_path in tqdm(test_images, desc="Analyzing detections"):
        # Détections baseline
        baseline_results = baseline_model(img_path, conf=conf, iou=iou, verbose=False)
        baseline_boxes = baseline_results[0].boxes
        
        if baseline_boxes is not None:
            n_baseline = len(baseline_boxes)
            baseline_stats['total_detections'] += n_baseline
            baseline_stats['detection_counts'].append(n_baseline)
            if n_baseline > 0:
                baseline_stats['avg_confidence'].extend(baseline_boxes.conf.cpu().numpy().tolist())
        else:
            baseline_stats['detection_counts'].append(0)
        
        # Détections enhanced
        enhanced_results = enhanced_model(img_path, conf=conf, iou=iou, verbose=False)
        enhanced_boxes = enhanced_results[0].boxes
        
        if enhanced_boxes is not None:
            n_enhanced = len(enhanced_boxes)
            enhanced_stats['total_detections'] += n_enhanced
            enhanced_stats['detection_counts'].append(n_enhanced)
            if n_enhanced > 0:
                enhanced_stats['avg_confidence'].extend(enhanced_boxes.conf.cpu().numpy().tolist())
        else:
            enhanced_stats['detection_counts'].append(0)
        
        # Sauvegarder les images si demandé
        if save_dir:
            save_comparison_image(
                img_path, 
                baseline_results[0], 
                enhanced_results[0],
                save_dir
            )
        
        # Données de comparaison
        comparison_data.append({
            'image': Path(img_path).name,
            'baseline_detections': baseline_stats['detection_counts'][-1],
            'enhanced_detections': enhanced_stats['detection_counts'][-1],
            'difference': enhanced_stats['detection_counts'][-1] - baseline_stats['detection_counts'][-1]
        })
    
    return baseline_stats, enhanced_stats, comparison_data


def save_comparison_image(img_path, baseline_result, enhanced_result, save_dir):
    """Sauvegarde une image avec les détections des deux modèles côte à côte"""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    # Créer une image double largeur
    comparison = np.zeros((h, w*2 + 10, 3), dtype=np.uint8)
    
    # Copier l'image originale deux fois
    comparison[:, :w] = img
    comparison[:, w+10:] = img
    
    # Ajouter une ligne de séparation
    comparison[:, w:w+10] = [255, 255, 255]
    
    # Dessiner les détections baseline (côté gauche)
    if baseline_result.boxes is not None:
        for box in baseline_result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cv2.rectangle(comparison, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(comparison, f'{conf:.2f}', (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Dessiner les détections enhanced (côté droit)
    if enhanced_result.boxes is not None:
        for box in enhanced_result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            # Décaler pour le côté droit
            x1 += w + 10
            x2 += w + 10
            cv2.rectangle(comparison, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(comparison, f'{conf:.2f}', (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Ajouter les labels
    cv2.putText(comparison, 'Baseline', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, 'Enhanced', (w+20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Sauvegarder
    output_path = save_dir / f'comparison_{Path(img_path).name}'
    cv2.imwrite(str(output_path), comparison)


def create_performance_report(baseline_times, enhanced_times, baseline_stats, 
                            enhanced_stats, comparison_data, output_dir):
    """Crée un rapport de performance détaillé"""
    
    # Créer le répertoire de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculer les statistiques
    report = {
        'timestamp': datetime.now().isoformat(),
        'speed_metrics': {
            'baseline': {
                'mean_ms': np.mean(baseline_times),
                'std_ms': np.std(baseline_times),
                'min_ms': np.min(baseline_times),
                'max_ms': np.max(baseline_times),
                'median_ms': np.median(baseline_times),
                'fps': 1000 / np.mean(baseline_times)
            },
            'enhanced': {
                'mean_ms': np.mean(enhanced_times),
                'std_ms': np.std(enhanced_times),
                'min_ms': np.min(enhanced_times),
                'max_ms': np.max(enhanced_times),
                'median_ms': np.median(enhanced_times),
                'fps': 1000 / np.mean(enhanced_times)
            }
        },
        'detection_metrics': {
            'baseline': {
                'total_detections': baseline_stats['total_detections'],
                'avg_detections_per_image': np.mean(baseline_stats['detection_counts']),
                'avg_confidence': np.mean(baseline_stats['avg_confidence']) if baseline_stats['avg_confidence'] else 0
            },
            'enhanced': {
                'total_detections': enhanced_stats['total_detections'],
                'avg_detections_per_image': np.mean(enhanced_stats['detection_counts']),
                'avg_confidence': np.mean(enhanced_stats['avg_confidence']) if enhanced_stats['avg_confidence'] else 0
            }
        },
        'improvements': {
            'speed_improvement_pct': ((np.mean(baseline_times) - np.mean(enhanced_times)) / np.mean(baseline_times) * 100),
            'detection_improvement_pct': ((enhanced_stats['total_detections'] - baseline_stats['total_detections']) / baseline_stats['total_detections'] * 100) if baseline_stats['total_detections'] > 0 else 0,
            'confidence_improvement': (np.mean(enhanced_stats['avg_confidence']) - np.mean(baseline_stats['avg_confidence'])) if enhanced_stats['avg_confidence'] and baseline_stats['avg_confidence'] else 0
        }
    }
    
    # Sauvegarder le rapport JSON
    with open(output_dir / 'performance_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    # Créer des visualisations
    create_visualizations(baseline_times, enhanced_times, baseline_stats, 
                         enhanced_stats, comparison_data, output_dir)
    
    return report


def create_visualizations(baseline_times, enhanced_times, baseline_stats, 
                         enhanced_stats, comparison_data, output_dir):
    """Crée des graphiques de visualisation"""
    
    # Configuration du style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Distribution des temps d'inférence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogramme
    ax1.hist(baseline_times, bins=30, alpha=0.7, label='Baseline', color='#3498db', density=True)
    ax1.hist(enhanced_times, bins=30, alpha=0.7, label='Enhanced', color='#2ecc71', density=True)
    ax1.axvline(np.mean(baseline_times), color='#2980b9', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(enhanced_times), color='#27ae60', linestyle='--', linewidth=2)
    ax1.set_xlabel('Temps d\'inférence (ms)')
    ax1.set_ylabel('Densité')
    ax1.set_title('Distribution des Temps d\'Inférence')
    ax1.legend()
    
    # Box plot
    ax2.boxplot([baseline_times, enhanced_times], labels=['Baseline', 'Enhanced'])
    ax2.set_ylabel('Temps d\'inférence (ms)')
    ax2.set_title('Comparaison des Temps d\'Inférence')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Comparaison des détections
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Nombre de détections par image
    x = range(len(baseline_stats['detection_counts']))
    ax1.plot(x, baseline_stats['detection_counts'], 'o-', label='Baseline', alpha=0.7, color='#3498db')
    ax1.plot(x, enhanced_stats['detection_counts'], 's-', label='Enhanced', alpha=0.7, color='#2ecc71')
    ax1.set_xlabel('Image Index')
    ax1.set_ylabel('Nombre de Détections')
    ax1.set_title('Détections par Image')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot des détections
    baseline_counts = baseline_stats['detection_counts']
    enhanced_counts = enhanced_stats['detection_counts']
    ax2.scatter(baseline_counts, enhanced_counts, alpha=0.6, s=50)
    
    # Ligne de référence y=x
    max_val = max(max(baseline_counts), max(enhanced_counts))
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    
    ax2.set_xlabel('Détections Baseline')
    ax2.set_ylabel('Détections Enhanced')
    ax2.set_title('Corrélation des Détections')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detection_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribution des confidences
    if baseline_stats['avg_confidence'] and enhanced_stats['avg_confidence']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Violin plot
        data = [baseline_stats['avg_confidence'], enhanced_stats['avg_confidence']]
        parts = ax.violinplot(data, positions=[1, 2], showmeans=True, showmedians=True)
        
        # Personnaliser les couleurs
        colors = ['#3498db', '#2ecc71']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Baseline', 'Enhanced'])
        ax.set_ylabel('Confidence Score')
        ax.set_title('Distribution des Scores de Confiance')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Graphique de performance globale
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculer les métriques normalisées
    baseline_fps = 1000 / np.mean(baseline_times)
    enhanced_fps = 1000 / np.mean(enhanced_times)
    
    baseline_detections = baseline_stats['total_detections']
    enhanced_detections = enhanced_stats['total_detections']
    
    baseline_conf = np.mean(baseline_stats['avg_confidence']) if baseline_stats['avg_confidence'] else 0
    enhanced_conf = np.mean(enhanced_stats['avg_confidence']) if enhanced_stats['avg_confidence'] else 0
    
    # Normaliser les valeurs (0-100)
    max_fps = max(baseline_fps, enhanced_fps)
    max_det = max(baseline_detections, enhanced_detections)
    max_conf = max(baseline_conf, enhanced_conf)
    
    categories = ['FPS\n(normalized)', 'Total Detections\n(normalized)', 'Avg Confidence\n(normalized)']
    baseline_values = [
        baseline_fps / max_fps * 100,
        baseline_detections / max_det * 100 if max_det > 0 else 0,
        baseline_conf / max_conf * 100 if max_conf > 0 else 0
    ]
    enhanced_values = [
        enhanced_fps / max_fps * 100,
        enhanced_detections / max_det * 100 if max_det > 0 else 0,
        enhanced_conf / max_conf * 100 if max_conf > 0 else 0
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='#3498db')
    bars2 = ax.bar(x + width/2, enhanced_values, width, label='Enhanced', color='#2ecc71')
    
    # Ajouter les valeurs sur les barres
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    ax.set_xlabel('Métriques')
    ax.set_ylabel('Performance Relative (%)')
    ax.set_title('Comparaison Globale des Performances')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'global_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Heatmap des différences de détection
    if len(comparison_data) > 20:
        # Limiter à 20 images pour la lisibilité
        comparison_subset = comparison_data[:20]
    else:
        comparison_subset = comparison_data
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    image_names = [d['image'] for d in comparison_subset]
    differences = [d['difference'] for d in comparison_subset]
    
    # Créer un gradient de couleur
    colors = ['#e74c3c' if d < 0 else '#2ecc71' if d > 0 else '#95a5a6' for d in differences]
    
    bars = ax.barh(image_names, differences, color=colors)
    ax.set_xlabel('Différence de Détections (Enhanced - Baseline)')
    ax.set_title('Différence de Détections par Image')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Ajouter les valeurs
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        if diff != 0:
            ax.text(diff/2, i, str(diff), ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detection_differences.png', dpi=300, bbox_inches='tight')
    plt.close()


def print_summary(report):
    """Affiche un résumé des résultats"""
    print("\n" + "="*80)
    print("📊 RÉSUMÉ DES PERFORMANCES")
    print("="*80)
    
    # Métriques de vitesse
    print("\n⏱️  Vitesse d'Inférence:")
    print(f"{'Modèle':<15} {'Moyenne (ms)':<15} {'FPS':<10} {'Min (ms)':<10} {'Max (ms)':<10}")
    print("-"*60)
    
    for model in ['baseline', 'enhanced']:
        metrics = report['speed_metrics'][model]
        print(f"{model.capitalize():<15} {metrics['mean_ms']:<15.2f} {metrics['fps']:<10.1f} "
              f"{metrics['min_ms']:<10.2f} {metrics['max_ms']:<10.2f}")
    
    # Métriques de détection
    print("\n🎯 Détections:")
    print(f"{'Modèle':<15} {'Total':<15} {'Moy/Image':<15} {'Confiance Moy':<15}")
    print("-"*60)
    
    for model in ['baseline', 'enhanced']:
        metrics = report['detection_metrics'][model]
        print(f"{model.capitalize():<15} {metrics['total_detections']:<15} "
              f"{metrics['avg_detections_per_image']:<15.2f} "
              f"{metrics['avg_confidence']:<15.3f}")
    
    # Améliorations
    print("\n📈 Améliorations:")
    print("-"*60)
    improvements = report['improvements']
    
    speed_imp = improvements['speed_improvement_pct']
    print(f"Vitesse: {speed_imp:+.1f}% " + 
          ("(plus rapide)" if speed_imp > 0 else "(plus lent)" if speed_imp < 0 else ""))
    
    det_imp = improvements['detection_improvement_pct']
    print(f"Détections: {det_imp:+.1f}% " +
          ("(plus de détections)" if det_imp > 0 else "(moins de détections)" if det_imp < 0 else ""))
    
    conf_imp = improvements['confidence_improvement']
    print(f"Confiance: {conf_imp:+.3f} " +
          ("(meilleure)" if conf_imp > 0 else "(moins bonne)" if conf_imp < 0 else ""))
    
    print("="*80)


def main():
    """Fonction principale"""
    args = parse_args()
    
    # Créer le répertoire de sortie
    output_dir = Path(args.output) / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Créer le répertoire pour les images si nécessaire
    if args.save_images:
        images_dir = output_dir / 'detection_images'
        images_dir.mkdir(exist_ok=True)
    else:
        images_dir = None
    
    # Charger les modèles
    baseline_model, enhanced_model = load_models(args.baseline, args.enhanced, args.device)
    
    # Obtenir la liste des images de test
    test_dir = Path(args.test_images)
    if not test_dir.exists():
        print(f"❌ Répertoire de test non trouvé: {test_dir}")
        print("Création d'images de test synthétiques...")
        test_dir.mkdir(exist_ok=True)
        # Créer quelques images de test
        for i in range(5):
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(test_dir / f'test_{i}.jpg'), img)
    
    test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
    print(f"\n📸 {len(test_images)} images de test trouvées")
    
    if len(test_images) == 0:
        print("❌ Aucune image de test trouvée!")
        return
    
    # Benchmark de vitesse
    print("\n⏱️  Benchmark de vitesse...")
    baseline_times = benchmark_speed(baseline_model, test_images, args.device, "Baseline")
    enhanced_times = benchmark_speed(enhanced_model, test_images, args.device, "Enhanced")
    
    # Analyse des détections
    print("\n🔍 Analyse des détections...")
    baseline_stats, enhanced_stats, comparison_data = analyze_detections(
        baseline_model, enhanced_model, test_images, 
        args.conf, args.iou, images_dir
    )
    
    # Créer le rapport
    print("\n📊 Génération du rapport...")
    report = create_performance_report(
        baseline_times, enhanced_times,
        baseline_stats, enhanced_stats,
        comparison_data, output_dir
    )
    
    # Afficher le résumé
    print_summary(report)
    
    print(f"\n✅ Analyse terminée!")
    print(f"📁 Résultats sauvegardés dans: {output_dir}")


if __name__ == '__main__':
    main()
