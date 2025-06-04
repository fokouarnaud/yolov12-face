# Guide d'Implémentation YOLOv13-Face dans train_evaluate_yolov_face.ipynb

## Configuration Initiale du Notebook

### Cellule 1: Imports et Configuration
```python
import os
import sys
from pathlib import Path

# Configuration du path pour le fork local
PROJECT_ROOT = Path.cwd()
ULTRALYTICS_PATH = PROJECT_ROOT / 'ultralytics'
sys.path.insert(0, str(ULTRALYTICS_PATH))

# Imports essentiels
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import select_device

# Configuration GPU optimale
device = select_device('0')  # ou 'cpu'
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.amp.autocast(enabled=True)

# Vérifier l'installation
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {device}")
```

### Cellule 2: Préparation des Données WIDERFace
```python
# Configuration des données
DATA_CONFIG = 'ultralytics/cfg/datasets/widerface.yaml'

# Vérifier la structure des données
import yaml
with open(DATA_CONFIG, 'r') as f:
    data_config = yaml.safe_load(f)
    
print("Configuration WIDERFace:")
print(f"Train: {data_config['train']}")
print(f"Val: {data_config['val']}")
print(f"Test: {data_config['test']}")
print(f"Classes: {data_config['names']}")

# Fonction pour vérifier les annotations
def verify_annotations(data_path):
    """Vérifie l'intégrité des annotations WIDERFace"""
    from ultralytics.data.utils import check_det_dataset
    dataset = check_det_dataset(data_path)
    print(f"Dataset vérifié: {dataset['nc']} classes")
    print(f"Train: {len(dataset['train'])} images")
    print(f"Val: {len(dataset['val'])} images")
    return dataset

dataset_info = verify_annotations(DATA_CONFIG)
```

### Cellule 3: Comparaison des Architectures
```python
# Charger les modèles pour comparaison
models = {
    'YOLOv12n-Face': 'ultralytics/cfg/models/v12/yolov12n-face.yaml',
    'YOLOv13n-Face-v3': 'ultralytics/cfg/models/v13/yolov13n-face-v3.yaml',
    'YOLOv13n-Face-v4': 'ultralytics/cfg/models/v13/yolov13n-face-v4.yaml'
}

# Analyser les architectures
def analyze_model_architecture(model_path):
    """Analyse la complexité du modèle"""
    model = YOLO(model_path)
    
    # Calculer les paramètres et FLOPs
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    print(f"\nModèle: {Path(model_path).stem}")
    print(f"Total paramètres: {total_params:,}")
    print(f"Paramètres entraînables: {trainable_params:,}")
    
    # Profile du modèle
    import torch.profiler as profiler
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    
    with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            _ = model.model(dummy_input)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
    
    return model

# Analyser chaque modèle
model_analysis = {}
for name, path in models.items():
    model_analysis[name] = analyze_model_architecture(path)
```

### Cellule 4: Entraînement avec Curriculum Learning
```python
class WIDERFaceCurriculumTrainer:
    """Entraînement progressif sur WIDERFace"""
    
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.stages = {
            'warmup': {
                'epochs': 10,
                'lr0': 0.001,
                'imgsz': 320,
                'batch': 64,
                'subset': 'easy'
            },
            'main': {
                'epochs': 50,
                'lr0': 0.01,
                'imgsz': 640,
                'batch': 32,
                'subset': 'all'
            },
            'finetune': {
                'epochs': 20,
                'lr0': 0.001,
                'imgsz': 640,
                'batch': 16,
                'subset': 'hard'
            }
        }
        
    def train_stage(self, stage_name):
        """Entraîne une étape spécifique"""
        config = self.stages[stage_name]
        
        print(f"\n{'='*50}")
        print(f"Stage: {stage_name}")
        print(f"Config: {config}")
        print(f"{'='*50}\n")
        
        model = YOLO(self.model_path)
        
        # Configuration d'entraînement
        results = model.train(
            data=self.data_path,
            epochs=config['epochs'],
            imgsz=config['imgsz'],
            batch=config['batch'],
            lr0=config['lr0'],
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.2,
            copy_paste=0.1,
            auto_augment='randaugment',
            amp=True,
            patience=10,
            save=True,
            save_period=5,
            cache=True,
            device=device,
            workers=8,
            project='runs/face_detection',
            name=f'yolov13_face_{stage_name}',
            exist_ok=True,
            resume=stage_name != 'warmup'
        )
        
        return results
    
    def train_full_curriculum(self):
        """Entraînement complet avec curriculum"""
        all_results = {}
        
        for stage in ['warmup', 'main', 'finetune']:
            results = self.train_stage(stage)
            all_results[stage] = results
            
            # Sauvegarder les métriques
            self.save_stage_metrics(stage, results)
            
        return all_results
    
    def save_stage_metrics(self, stage, results):
        """Sauvegarde les métriques de chaque étape"""
        import json
        metrics_path = f'runs/face_detection/yolov13_face_{stage}/metrics.json'
        
        metrics = {
            'stage': stage,
            'best_fitness': float(results.best_fitness),
            'final_epoch': int(results.epoch),
            'training_time': float(results.t),
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

# Lancer l'entraînement
trainer = WIDERFaceCurriculumTrainer(
    model_path='ultralytics/cfg/models/v13/yolov13n-face-v4.yaml',
    data_path=DATA_CONFIG
)

# Pour entraîner complètement (décommenter)
# results = trainer.train_full_curriculum()

# Pour entraîner une seule étape
results_warmup = trainer.train_stage('warmup')
```

### Cellule 5: Évaluation Avancée
```python
class AdvancedFaceEvaluator:
    """Évaluation complète sur WIDERFace"""
    
    def __init__(self, model_path, data_path):
        self.model = YOLO(model_path)
        self.data_path = data_path
        self.subsets = ['easy', 'medium', 'hard']
        
    def evaluate_subset(self, subset):
        """Évalue sur un sous-ensemble spécifique"""
        print(f"\nÉvaluation sur WIDERFace {subset.upper()}")
        
        results = self.model.val(
            data=self.data_path,
            split='val',
            batch=32,
            imgsz=640,
            conf=0.001,
            iou=0.5,
            max_det=1000,
            device=device,
            workers=8,
            save_json=True,
            save_hybrid=True,
            plots=True,
            rect=True,
            task='detect'
        )
        
        return results
    
    def evaluate_all(self):
        """Évaluation complète"""
        all_results = {}
        
        for subset in self.subsets:
            results = self.evaluate_subset(subset)
            all_results[subset] = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.p),
                'recall': float(results.box.r),
                'speed': {
                    'preprocess': float(results.speed['preprocess']),
                    'inference': float(results.speed['inference']),
                    'postprocess': float(results.speed['postprocess'])
                }
            }
            
        return all_results
    
    def compare_models(self, model_paths):
        """Compare plusieurs modèles"""
        comparison_results = {}
        
        for model_name, model_path in model_paths.items():
            print(f"\n{'='*50}")
            print(f"Évaluation de {model_name}")
            print(f"{'='*50}")
            
            self.model = YOLO(model_path)
            comparison_results[model_name] = self.evaluate_all()
            
        return comparison_results
    
    def visualize_results(self, results):
        """Visualise les résultats de comparaison"""
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Préparer les données pour visualisation
        data = []
        for model_name, model_results in results.items():
            for subset, metrics in model_results.items():
                data.append({
                    'Model': model_name,
                    'Subset': subset,
                    'mAP50': metrics['mAP50'],
                    'mAP50-95': metrics['mAP50-95'],
                    'FPS': 1000 / sum(metrics['speed'].values())
                })
        
        df = pd.DataFrame(data)
        
        # Graphiques
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # mAP50 par subset
        df.pivot(index='Subset', columns='Model', values='mAP50').plot(
            kind='bar', ax=axes[0], title='mAP50 par Subset'
        )
        
        # mAP50-95 par subset
        df.pivot(index='Subset', columns='Model', values='mAP50-95').plot(
            kind='bar', ax=axes[1], title='mAP50-95 par Subset'
        )
        
        # FPS par modèle
        df.groupby('Model')['FPS'].mean().plot(
            kind='bar', ax=axes[2], title='FPS Moyen'
        )
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300)
        plt.show()
        
        return df

# Évaluer un modèle
evaluator = AdvancedFaceEvaluator(
    model_path='runs/face_detection/yolov13_face_main/weights/best.pt',
    data_path=DATA_CONFIG
)

# Évaluation simple
# results = evaluator.evaluate_all()

# Comparaison de modèles
# comparison_results = evaluator.compare_models({
#     'YOLOv12n': 'path/to/yolov12n/best.pt',
#     'YOLOv13n-v3': 'path/to/yolov13n-v3/best.pt',
#     'YOLOv13n-v4': 'path/to/yolov13n-v4/best.pt'
# })
```

### Cellule 6: Inférence et Visualisation
```python
def inference_on_challenging_cases(model_path, test_images_path):
    """Test sur des cas difficiles"""
    model = YOLO(model_path)
    
    # Catégories de cas difficiles
    challenging_cases = {
        'occlusion': 'images avec occlusions',
        'small_faces': 'petits visages',
        'crowd': 'foules',
        'low_light': 'faible éclairage',
        'extreme_pose': 'poses extrêmes'
    }
    
    results_by_category = {}
    
    for category, description in challenging_cases.items():
        print(f"\nTest sur {description}")
        
        # Obtenir les images de cette catégorie
        category_path = Path(test_images_path) / category
        if category_path.exists():
            images = list(category_path.glob('*.jpg'))
            
            # Inférence
            results = model.predict(
                source=images,
                conf=0.25,
                iou=0.45,
                imgsz=640,
                max_det=1000,
                device=device,
                agnostic_nms=True,
                retina_masks=True,
                classes=[0],  # Face seulement
                save=True,
                save_txt=True,
                save_conf=True,
                project='runs/inference',
                name=f'{category}_results'
            )
            
            results_by_category[category] = results
            
            # Analyser les performances
            analyze_category_performance(results, category)
    
    return results_by_category

def analyze_category_performance(results, category):
    """Analyse les performances par catégorie"""
    detections = []
    for r in results:
        if r.boxes is not None:
            detections.append(len(r.boxes))
    
    if detections:
        print(f"  - Détections moyennes: {np.mean(detections):.2f}")
        print(f"  - Min/Max détections: {min(detections)}/{max(detections)}")
        print(f"  - Images sans détection: {detections.count(0)}")

# Test d'inférence
# results = inference_on_challenging_cases(
#     'runs/face_detection/yolov13_face_main/weights/best.pt',
#     'path/to/test/images'
# )
```

### Cellule 7: Export et Optimisation
```python
def export_and_optimize_model(model_path, export_formats=['onnx', 'tflite', 'engine']):
    """Export et optimisation du modèle"""
    model = YOLO(model_path)
    
    export_results = {}
    
    for format in export_formats:
        print(f"\nExport au format {format.upper()}")
        
        try:
            # Export
            exported_path = model.export(
                format=format,
                imgsz=640,
                keras=False,
                optimize=True,
                half=True if format == 'engine' else False,
                int8=True if format == 'tflite' else False,
                dynamic=True if format == 'onnx' else False,
                simplify=True if format == 'onnx' else False,
                opset=12 if format == 'onnx' else None,
                workspace=4 if format == 'engine' else None,
                nms=True,
                batch=1
            )
            
            export_results[format] = {
                'success': True,
                'path': exported_path,
                'size_mb': os.path.getsize(exported_path) / 1024 / 1024
            }
            
            # Benchmark du modèle exporté
            if format == 'onnx':
                benchmark_onnx_model(exported_path)
                
        except Exception as e:
            export_results[format] = {
                'success': False,
                'error': str(e)
            }
    
    return export_results

def benchmark_onnx_model(onnx_path):
    """Benchmark du modèle ONNX"""
    import onnxruntime as ort
    import time
    
    # Créer une session ONNX
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Input factice
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_input})
    
    # Benchmark
    times = []
    for _ in range(100):
        start = time.time()
        _ = session.run(None, {input_name: dummy_input})
        times.append(time.time() - start)
    
    print(f"  ONNX Inference Time: {np.mean(times)*1000:.2f} ms")
    print(f"  ONNX FPS: {1/np.mean(times):.2f}")

# Export du modèle
# export_results = export_and_optimize_model(
#     'runs/face_detection/yolov13_face_main/weights/best.pt'
# )
```

### Cellule 8: Rapport Final
```python
def generate_final_report(model_name, training_results, eval_results, export_results):
    """Génère un rapport complet"""
    
    report = f"""
# Rapport d'Évaluation - {model_name}
    
## Résultats d'Entraînement
- Epochs: {training_results.get('epochs', 'N/A')}
- Best mAP50: {training_results.get('best_map50', 'N/A'):.4f}
- Training Time: {training_results.get('training_time', 'N/A'):.2f} heures
    
## Performance sur WIDERFace
### Easy
- mAP50: {eval_results['easy']['mAP50']:.4f}
- mAP50-95: {eval_results['easy']['mAP50-95']:.4f}
- FPS: {1000/sum(eval_results['easy']['speed'].values()):.2f}
    
### Medium
- mAP50: {eval_results['medium']['mAP50']:.4f}
- mAP50-95: {eval_results['medium']['mAP50-95']:.4f}
- FPS: {1000/sum(eval_results['medium']['speed'].values()):.2f}
    
### Hard
- mAP50: {eval_results['hard']['mAP50']:.4f}
- mAP50-95: {eval_results['hard']['mAP50-95']:.4f}
- FPS: {1000/sum(eval_results['hard']['speed'].values()):.2f}
    
## Export et Déploiement
"""
    
    for format, result in export_results.items():
        if result['success']:
            report += f"- {format.upper()}: ✓ ({result['size_mb']:.2f} MB)\n"
        else:
            report += f"- {format.upper()}: ✗ ({result['error']})\n"
    
    # Sauvegarder le rapport
    with open(f'{model_name}_report.md', 'w') as f:
        f.write(report)
    
    print(report)
    
    return report

# Générer le rapport final
# report = generate_final_report(
#     'YOLOv13n-Face-v4',
#     training_results,
#     eval_results,
#     export_results
# )
```

## Instructions d'Utilisation

1. **Installation des dépendances** (dans la première cellule du notebook):
```python
!pip install -e .  # Depuis le répertoire racine avec pyproject.toml
```

2. **Ordre d'exécution recommandé**:
   - Configuration initiale (Cellule 1)
   - Vérification des données (Cellule 2)
   - Analyse architecturale (Cellule 3)
   - Entraînement avec curriculum (Cellule 4)
   - Évaluation complète (Cellule 5)
   - Tests d'inférence (Cellule 6)
   - Export et optimisation (Cellule 7)
   - Rapport final (Cellule 8)

3. **Personnalisation**:
   - Ajuster les hyperparamètres dans la Cellule 4
   - Modifier les seuils de détection dans la Cellule 6
   - Choisir les formats d'export dans la Cellule 7

4. **Monitoring**:
   - Utiliser TensorBoard: `tensorboard --logdir runs/face_detection`
   - Vérifier les logs dans `runs/face_detection/*/`

Ce guide fournit une implémentation complète et robuste de YOLOv13-Face dans votre notebook existant.
