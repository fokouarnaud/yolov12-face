"""
Face-Aware Geometric Attention (FAGA) Module
Une innovation pour YOLOv12-Face qui exploite la géométrie faciale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FaceAwareGeometricAttention(nn.Module):
    """
    FAGA: Face-Aware Geometric Attention Module
    
    Innovation clé: Exploite la structure géométrique des visages humains
    - Attention guidée par les points clés faciaux (yeux, nez, bouche)
    - Mécanisme de pondération adaptatif basé sur les rapports géométriques
    - Intégration de la symétrie faciale
    """
    
    def __init__(self, in_channels, out_channels, n=1, num_landmarks=5, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_landmarks = num_landmarks  # 5 points: 2 yeux, nez, 2 coins bouche
        
        # 1. Landmark Prediction Branch
        self.landmark_predictor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_landmarks * 2, 1)  # x,y pour chaque landmark
        )
        
        # 2. Geometric Relationship Encoder
        self.geometric_encoder = nn.Sequential(
            nn.Linear(num_landmarks * (num_landmarks - 1) // 2, 64),  # Distances entre paires
            nn.ReLU(inplace=True),
            nn.Linear(64, in_channels // 4)
        )
        
        # 3. Symmetry-Aware Attention
        self.left_branch = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.right_branch = nn.Conv2d(in_channels, in_channels // 2, 1)
        
        # 4. Multi-Scale Face-Specific Kernels
        # Tailles optimisées pour les composants faciaux
        self.eye_kernel = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)  # Petits détails
        self.nose_kernel = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)  # Taille moyenne
        self.mouth_kernel = nn.Conv2d(in_channels, out_channels // 4, 7, padding=3)  # Plus large
        self.face_kernel = nn.Conv2d(in_channels, out_channels // 4, 9, padding=4)  # Contexte global
        
        # 5. Adaptive Fusion
        self.fusion = nn.Conv2d(out_channels + in_channels // 4, out_channels, 1)
        
        # 6. Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def compute_geometric_features(self, landmarks):
        """Calcule les distances entre points clés pour encoder la géométrie faciale"""
        B, _, H, W = landmarks.shape
        landmarks = landmarks.view(B, self.num_landmarks, 2, H, W)
        
        # Calculer les centres de masse des heatmaps
        coords = []
        for i in range(self.num_landmarks):
            heatmap = landmarks[:, i].sum(dim=1)  # Sum x,y
            heatmap = F.softmax(heatmap.view(B, -1), dim=1).view(B, H, W)
            
            # Coordonnées pondérées
            x_coords = torch.arange(W, device=landmarks.device).float()
            y_coords = torch.arange(H, device=landmarks.device).float()
            x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='xy')
            
            x_center = (heatmap * x_grid).sum(dim=[1, 2])
            y_center = (heatmap * y_grid).sum(dim=[1, 2])
            
            coords.append(torch.stack([x_center, y_center], dim=1))
        
        coords = torch.stack(coords, dim=1)  # B x num_landmarks x 2
        
        # Calculer toutes les distances par paires
        distances = []
        for i in range(self.num_landmarks):
            for j in range(i + 1, self.num_landmarks):
                dist = torch.norm(coords[:, i] - coords[:, j], dim=1)
                distances.append(dist)
        
        return torch.stack(distances, dim=1)  # B x (num_landmarks * (num_landmarks-1) / 2)
    
    def apply_symmetry_attention(self, x):
        """Applique une attention tenant compte de la symétrie faciale"""
        B, C, H, W = x.shape
        
        # Diviser l'image en deux moitiés
        left_half = x[:, :, :, :W//2]
        right_half = x[:, :, :, W//2:]
        
        # Flip la moitié droite pour comparaison
        right_half_flipped = torch.flip(right_half, dims=[3])
        
        # Calculer la similarité entre les deux moitiés
        similarity = F.cosine_similarity(
            self.left_branch(left_half).mean(dim=[2, 3]),
            self.right_branch(right_half_flipped).mean(dim=[2, 3]),
            dim=1
        ).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        # Utiliser la similarité comme poids d'attention
        return x * similarity.expand_as(x)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Prédire les landmarks
        landmarks = self.landmark_predictor(x)
        
        # 2. Extraire les caractéristiques géométriques
        geometric_features = self.compute_geometric_features(landmarks)
        geometric_encoding = self.geometric_encoder(geometric_features)
        geometric_attention = geometric_encoding.unsqueeze(2).unsqueeze(3).expand(B, -1, H, W)
        
        # 3. Appliquer l'attention de symétrie
        x_sym = self.apply_symmetry_attention(x)
        
        # 4. Extractions multi-échelles spécifiques aux visages
        eye_features = self.eye_kernel(x_sym)
        nose_features = self.nose_kernel(x_sym)
        mouth_features = self.mouth_kernel(x_sym)
        face_features = self.face_kernel(x_sym)
        
        # 5. Concaténer toutes les caractéristiques
        multi_scale = torch.cat([eye_features, nose_features, mouth_features, face_features], dim=1)
        combined = torch.cat([multi_scale, geometric_attention], dim=1)
        
        # 6. Fusion adaptive
        out = self.fusion(combined)
        
        # 7. Connection résiduelle
        return out + self.residual(x)


class FacePyramidAttention(nn.Module):
    """
    Module d'attention pyramidale optimisé pour les visages
    Traite différentes échelles de visages efficacement
    """
    
    def __init__(self, in_channels, out_channels, n=1, pyramid_levels=3, *args, **kwargs):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        
        # Branches pour différentes échelles de visages
        self.pyramid_branches = nn.ModuleList()
        for i in range(pyramid_levels):
            scale_factor = 2 ** i
            branch = nn.Sequential(
                nn.AvgPool2d(scale_factor) if scale_factor > 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels // pyramid_levels, 3, padding=1),
                nn.BatchNorm2d(out_channels // pyramid_levels),
                nn.ReLU(inplace=True)
            )
            self.pyramid_branches.append(branch)
        
        # Attention weights pour chaque échelle
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, pyramid_levels, 1),
            nn.Softmax(dim=1)
        )
        
        # Fusion finale
        self.fusion = nn.Conv2d(out_channels, out_channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Calculer les poids d'attention pour chaque échelle
        scale_weights = self.scale_attention(x)  # B x pyramid_levels x 1 x 1
        
        # Traiter chaque échelle
        pyramid_features = []
        for i, branch in enumerate(self.pyramid_branches):
            feat = branch(x)
            
            # Upsampler si nécessaire
            if feat.shape[2] != H or feat.shape[3] != W:
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            
            # Appliquer le poids d'échelle
            weight = scale_weights[:, i:i+1, :, :]
            feat = feat * weight
            
            pyramid_features.append(feat)
        
        # Concaténer et fusionner
        combined = torch.cat(pyramid_features, dim=1)
        return self.fusion(combined)


# Export des nouveaux modules
__all__ = ['FaceAwareGeometricAttention', 'FacePyramidAttention', 'A2Module', 'RELAN']
