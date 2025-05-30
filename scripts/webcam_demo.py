#!/usr/bin/env python3
"""
YOLOv12-Face Real-time Webcam Demo
Teste le modèle YOLOv12-Face en temps réel avec une webcam
"""

import cv2
import numpy as np
import torch
import time
import argparse
from pathlib import Path
import sys

# Ajouter le répertoire parent au path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from ultralytics import YOLO


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description='YOLOv12-Face Webcam Demo')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--source', type=int, default=0,
                        help='Webcam source (0 for default camera)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (0 for GPU, cpu for CPU)')
    parser.add_argument('--save-video', type=str, default=None,
                        help='Save output video to file')
    parser.add_argument('--show-fps', action='store_true',
                        help='Show FPS counter')
    parser.add_argument('--show-info', action='store_true',
                        help='Show detection info')
    return parser.parse_args()


def draw_boxes(image, boxes, conf_threshold=0.5, show_info=False):
    """Dessine les boîtes de détection sur l'image"""
    if boxes is None:
        return image, 0
    
    detections = 0
    for box in boxes:
        # Extraire les coordonnées et la confiance
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].cpu().numpy()
        
        if conf >= conf_threshold:
            detections += 1
            
            # Dessiner la boîte
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Ajouter le score de confiance
            label = f'Face: {conf:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Fond pour le texte
            cv2.rectangle(image, (x1, y1 - label_size[1] - 4), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Texte
            cv2.putText(image, label, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            if show_info:
                # Afficher la taille de la boîte
                width = x2 - x1
                height = y2 - y1
                size_label = f'{width}x{height}'
                cv2.putText(image, size_label, (x1, y2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return image, detections


def main():
    """Fonction principale"""
    args = parse_args()
    
    print("🚀 YOLOv12-Face Webcam Demo")
    print("="*50)
    print(f"📁 Modèle: {args.model}")
    print(f"📷 Source: Camera {args.source}")
    print(f"🎯 Seuil de confiance: {args.conf}")
    print(f"🖥️  Device: {args.device}")
    print("="*50)
    print("\nAppuyez sur 'q' pour quitter")
    print("Appuyez sur 's' pour prendre une capture d'écran")
    print("Appuyez sur 'ESPACE' pour pause/reprise\n")
    
    # Charger le modèle
    print("📥 Chargement du modèle...")
    model = YOLO(args.model)
    model.to(args.device)
    
    # Ouvrir la webcam
    cap = cv2.VideoCapture(args.source)
    
    # Configurer la capture vidéo
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Vérifier que la webcam est ouverte
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam!")
        return
    
    # Obtenir les propriétés de la vidéo
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Résolution: {width}x{height} @ {fps:.1f} FPS")
    
    # Configurer l'enregistrement vidéo si demandé
    out = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.save_video, fourcc, fps, (width, height))
        print(f"📼 Enregistrement vidéo: {args.save_video}")
    
    # Variables pour le FPS
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    paused = False
    screenshot_count = 0
    
    print("\n🎬 Démarrage de la détection en temps réel...")
    
    try:
        while True:
            # Lire une frame
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Impossible de lire la frame!")
                    break
                
                # Faire la détection
                results = model(frame, conf=args.conf, iou=args.iou, verbose=False)
                
                # Dessiner les boîtes
                frame, num_faces = draw_boxes(frame, results[0].boxes, args.conf, args.show_info)
                
                # Calculer le FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    end_time = time.time()
                    fps_display = 30 / (end_time - start_time)
                    start_time = end_time
                
                # Afficher les informations
                info_y = 30
                
                # Nombre de visages détectés
                cv2.putText(frame, f'Faces: {num_faces}', (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                info_y += 30
                
                # FPS
                if args.show_fps:
                    cv2.putText(frame, f'FPS: {fps_display:.1f}', (10, info_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    info_y += 30
                
                # Modèle utilisé
                model_name = Path(args.model).stem
                cv2.putText(frame, f'Model: {model_name}', (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Statut pause
                if paused:
                    cv2.putText(frame, 'PAUSED', (width//2 - 50, height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # Enregistrer la frame si nécessaire
                if out is not None:
                    out.write(frame)
            
            # Afficher la frame
            cv2.imshow('YOLOv12-Face Detection', frame)
            
            # Gérer les touches
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Arrêt demandé par l'utilisateur")
                break
            elif key == ord('s'):
                # Prendre une capture d'écran
                screenshot_name = f'screenshot_{screenshot_count:04d}.jpg'
                cv2.imwrite(screenshot_name, frame)
                print(f"📸 Capture d'écran sauvegardée: {screenshot_name}")
                screenshot_count += 1
            elif key == ord(' '):
                # Pause/Reprise
                paused = not paused
                if paused:
                    print("⏸️  Pause")
                else:
                    print("▶️  Reprise")
                    start_time = time.time()
    
    except KeyboardInterrupt:
        print("\n⚠️  Interruption clavier détectée")
    
    finally:
        # Nettoyer
        print("\n🧹 Nettoyage...")
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        # Afficher les statistiques finales
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print("\n📊 Statistiques de session:")
        print(f"  - Frames traitées: {frame_count}")
        print(f"  - Temps total: {total_time:.1f}s")
        print(f"  - FPS moyen: {avg_fps:.1f}")
        print(f"  - Captures d'écran: {screenshot_count}")
        
        print("\n✅ Terminé!")


if __name__ == '__main__':
    main()
