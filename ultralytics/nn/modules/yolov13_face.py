"""
YOLOv13-Face: Next-Generation Face Detection Architecture for 2025
Combining Efficient Vision Transformers with Neural Architecture Search
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class EfficientFaceTransformer(nn.Module):
    """
    Efficient Face Transformer (EFT) - Innovation principale de YOLOv13-Face
    
    Inspiré par:
    - Vision Transformers efficaces (2024)
    - Multi-head attention triplet pour la détection de visages
    - Knowledge distillation pour l'efficacité
    """
    
    def __init__(self, dim, num_heads=8, window_size=7, shift_size=0, 
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        # Triplet Multi-Head Attention pour les visages
        self.face_attn = TripletFaceAttention(
            dim, num_heads=num_heads, window_size=window_size,
            qkv_bias=qkv_bias, attn_drop=attn_drop
        )
        
        # MLP efficace
        self.mlp = EfficientMLP(dim, int(dim * mlp_ratio), drop=drop)
        
        # Normalisation adaptative
        self.norm1 = AdaptiveLayerNorm(dim)
        self.norm2 = AdaptiveLayerNorm(dim)
        
        # Face-specific token mixing
        self.token_mixer = FaceTokenMixer(dim)
        
    def forward(self, x, face_priors=None):
        """
        Args:
            x: Input features [B, C, H, W]
            face_priors: Optional face region priors
        """
        B, C, H, W = x.shape
        
        # Window partition si nécessaire
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        
        # Attention avec priors faciaux
        shortcut = x
        x = self.norm1(x)
        x = self.face_attn(x, face_priors)
        x = shortcut + x
        
        # Token mixing spécifique aux visages
        x = x + self.token_mixer(self.norm2(x))
        
        # MLP
        x = x + self.mlp(x)
        
        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        
        return x


class TripletFaceAttention(nn.Module):
    """
    Attention triplet pour les visages: Query-Key-Value avec contraintes géométriques
    """
    
    def __init__(self, dim, num_heads, window_size, qkv_bias=True, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        # Assurer que dim est divisible par num_heads
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        head_dim = dim // num_heads
        assert head_dim > 0, f"head dimension must be > 0, got {head_dim}"
        
        self.scale = head_dim ** -0.5
        
        # Projections Q, K, V avec biais adaptatifs pour les visages
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Relative position encoding pour capturer la structure faciale
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Face structure prior - encode les positions typiques des composants faciaux
        self.face_structure_prior = nn.Parameter(
            torch.zeros(1, num_heads, window_size * window_size, window_size * window_size)
        )
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        
        # Geometric consistency loss pour maintenir la cohérence faciale
        self.geometric_mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 5 * 2)  # 5 landmarks x 2 coords
        )
        
    def forward(self, x, face_priors=None):
        B, C, H, W = x.shape
        N = H * W
        
        # Reshape pour l'attention
        x_flat = x.flatten(2).transpose(1, 2)  # B, N, C
        
        # Compute Q, K, V
        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention avec scaling
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Ajouter le prior de structure faciale
        attn = attn + self.face_structure_prior
        
        # Ajouter les biais de position relative
        relative_position_bias = self._get_relative_position_bias(H, W)
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Appliquer les priors faciaux si disponibles
        if face_priors is not None:
            attn = attn * face_priors.unsqueeze(1)
        
        # Softmax et dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Appliquer l'attention
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        # Prédire les landmarks pour la cohérence géométrique
        landmarks = self.geometric_mlp(x.mean(dim=1))
        
        # Reshape back
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        return x
    
    def _get_relative_position_bias(self, H, W):
        """Calcule les biais de position relative"""
        coords_h = torch.arange(H)
        coords_w = torch.arange(W)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += H - 1
        relative_coords[:, :, 1] += W - 1
        relative_coords[:, :, 0] *= 2 * W - 1
        
        relative_position_index = relative_coords.sum(-1)
        return self.relative_position_bias_table[relative_position_index.view(-1)].view(
            H * W, H * W, -1).permute(2, 0, 1).contiguous()


class FaceTokenMixer(nn.Module):
    """
    Mélangeur de tokens optimisé pour les visages
    Utilise des convolutions séparables en profondeur avec patterns spécifiques aux visages
    """
    
    def __init__(self, dim):
        super().__init__()
        # Convolutions pour différentes parties du visage
        self.eye_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.nose_conv = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.mouth_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        
        # Fusion adaptative
        self.fusion = nn.Conv2d(dim * 3, dim, 1)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim * 3, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extraire features pour différentes parties
        eye_feat = self.eye_conv(x)
        nose_feat = self.nose_conv(x)
        mouth_feat = self.mouth_conv(x)
        
        # Concaténer
        concat_feat = torch.cat([eye_feat, nose_feat, mouth_feat], dim=1)
        
        # Gating adaptatif
        gates = self.gate(x)
        gated_feat = concat_feat * gates
        
        # Fusion finale
        return self.fusion(gated_feat)


class AdaptiveLayerNorm(nn.Module):
    """Layer Normalization adaptative pour les visages"""
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        
        # Paramètres adaptatifs basés sur le contenu
        self.adapt_mlp = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim * 2, 1)
        )
        
    def forward(self, x):
        # Calcul des paramètres adaptatifs
        adapt_params = self.adapt_mlp(x.mean(dim=[2, 3], keepdim=True))
        adapt_weight, adapt_bias = adapt_params.chunk(2, dim=1)
        
        # Layer norm standard
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        
        # Application des poids adaptatifs
        weight = self.weight.view(1, -1, 1, 1) * (1 + adapt_weight)
        bias = self.bias.view(1, -1, 1, 1) + adapt_bias
        
        return weight * x + bias


class EfficientMLP(nn.Module):
    """MLP efficace avec mixture of experts pour différentes échelles"""
    
    def __init__(self, in_features, hidden_features, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        
        # Mixture of Experts pour différentes échelles de visages
        self.expert_small = nn.Sequential(
            nn.Conv2d(in_features, hidden_features // 3, 1),
            nn.GELU(),
            nn.Conv2d(hidden_features // 3, out_features, 1)
        )
        
        self.expert_medium = nn.Sequential(
            nn.Conv2d(in_features, hidden_features // 3, 1),
            nn.GELU(),
            nn.Conv2d(hidden_features // 3, out_features, 1)
        )
        
        self.expert_large = nn.Sequential(
            nn.Conv2d(in_features, hidden_features // 3, 1),
            nn.GELU(),
            nn.Conv2d(hidden_features // 3, out_features, 1)
        )
        
        # Router pour sélectionner l'expert
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, 3, 1),
            nn.Softmax(dim=1)
        )
        
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        # Calcul des poids du router
        router_weights = self.router(x)
        
        # Appliquer chaque expert
        out_small = self.expert_small(x)
        out_medium = self.expert_medium(x)
        out_large = self.expert_large(x)
        
        # Combinaison pondérée
        out = (router_weights[:, 0:1] * out_small + 
               router_weights[:, 1:2] * out_medium + 
               router_weights[:, 2:3] * out_large)
        
        return self.drop(out)


class NeuralArchitectureSearchBlock(nn.Module):
    """
    Bloc avec recherche d'architecture neuronale (NAS)
    Permet l'optimisation automatique de l'architecture
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Opérations candidates
        self.ops = nn.ModuleList([
            # Identity
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1),
            # Conv 3x3
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),
            # Conv 5x5 séparable
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),
            # Face Transformer
            EfficientFaceTransformer(out_channels) if in_channels == out_channels else 
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                EfficientFaceTransformer(out_channels)
            )
        ])
        
        # Paramètres d'architecture (appris pendant l'entraînement)
        self.arch_params = nn.Parameter(torch.ones(len(self.ops)) / len(self.ops))
        
    def forward(self, x):
        # Softmax sur les paramètres d'architecture
        weights = F.softmax(self.arch_params, dim=0)
        
        # Combinaison pondérée des opérations
        out = 0
        for w, op in zip(weights, self.ops):
            out = out + w * op(x)
            
        return out


# Module principal YOLOv13-Face
class YOLOv13FaceBackbone(nn.Module):
    """
    Backbone YOLOv13-Face avec innovations 2025:
    - Efficient Face Transformers
    - Neural Architecture Search
    - Multi-scale Face Experts
    - Geometric Consistency
    """
    
    def __init__(self, channels=[64, 128, 256, 512, 1024], depths=[2, 2, 6, 2]):
        super().__init__()
        
        # Stem avec attention précoce
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 4, stride=4),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            TripletFaceAttention(channels[0], num_heads=4, window_size=7)
        )
        
        # Stages avec NAS et Transformers
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = nn.Sequential(
                # Downsample
                nn.Conv2d(channels[i], channels[i+1], 3, stride=2, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(),
                # Blocks avec NAS
                *[NeuralArchitectureSearchBlock(channels[i+1], channels[i+1]) 
                  for _ in range(depths[i])]
            )
            self.stages.append(stage)
        
        # Face Prior Generator
        self.face_prior_gen = nn.Sequential(
            nn.Conv2d(channels[1], 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Générer les priors faciaux
        face_priors = None
        
        outputs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            
            # Générer les priors après le premier stage
            if i == 0:
                face_priors = self.face_prior_gen(x)
            
            outputs.append(x)
        
        return outputs, face_priors


# Export
__all__ = ['YOLOv13FaceBackbone', 'EfficientFaceTransformer', 'NeuralArchitectureSearchBlock']
