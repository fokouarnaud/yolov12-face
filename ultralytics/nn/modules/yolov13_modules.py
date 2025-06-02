"""
Modules complémentaires pour YOLOv13-Face
Inclut C2fTransformer, FaceFeatureRefinement, GeometricConsistency, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv, DWConv
from .block import C2f, Bottleneck


class C2fTransformer(nn.Module):
    """C2f avec intégration de transformers efficaces"""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        # Remplacer les bottlenecks par des blocs transformer/convolution hybrides
        self.m = nn.ModuleList()
        for i in range(n):
            if i % 2 == 0:  # Alterner transformer et convolution
                self.m.append(MiniTransformerBlock(self.c, self.c))
            else:
                self.m.append(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0))
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MiniTransformerBlock(nn.Module):
    """Bloc transformer léger pour intégration dans C2f"""
    
    def __init__(self, dim, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SimplifiedAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape pour transformer
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Attention
        x_flat = x_flat + self.attn(self.norm1(x_flat))
        
        # MLP
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        
        # Reshape back
        return x_flat.transpose(1, 2).reshape(B, C, H, W)


class SimplifiedAttention(nn.Module):
    """Attention simplifiée pour efficacité"""
    
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class FaceFeatureRefinement(nn.Module):
    """Raffinement des features spécifique aux visages"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Branches pour différentes parties du visage
        self.upper_face = nn.Sequential(
            Conv(in_channels, out_channels // 2, 3, 1, 1),
            Conv(out_channels // 2, out_channels // 2, 3, 1, 1)
        )
        
        self.lower_face = nn.Sequential(
            Conv(in_channels, out_channels // 2, 5, 1, 2),
            Conv(out_channels // 2, out_channels // 2, 3, 1, 1)
        )
        
        # Fusion avec attention
        self.fusion = nn.Sequential(
            Conv(out_channels, out_channels, 1),
            nn.Sigmoid()
        )
        
        # Projection finale
        self.project = Conv(out_channels, out_channels, 1)
        
    def forward(self, x):
        # Diviser spatialement (approximatif)
        h = x.shape[2]
        upper = x[:, :, :h//2, :]
        lower = x[:, :, h//2:, :]
        
        # Traiter séparément
        upper_feat = self.upper_face(F.pad(upper, (0, 0, 0, h//2)))
        lower_feat = self.lower_face(F.pad(lower, (0, 0, h//2, 0)))
        
        # Combiner avec attention
        combined = torch.cat([upper_feat[:, :out_channels//2], 
                            lower_feat[:, out_channels//2:]], dim=1)
        
        attention = self.fusion(combined)
        refined = combined * attention
        
        return self.project(refined) + x


class GeometricConsistency(nn.Module):
    """Module de cohérence géométrique pour les landmarks"""
    
    def __init__(self, in_channels, num_landmarks=5):
        super().__init__()
        self.num_landmarks = num_landmarks
        
        # Prédicteur de landmarks
        self.landmark_pred = nn.Sequential(
            Conv(in_channels, 256, 3, 1, 1),
            Conv(256, 128, 3, 1, 1),
            nn.Conv2d(128, num_landmarks * 2, 1)  # x, y pour chaque landmark
        )
        
        # Encodeur de relations géométriques
        self.geometric_encoder = nn.Sequential(
            nn.Linear(num_landmarks * (num_landmarks - 1) // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Fusion avec features originales
        self.fusion = nn.Sequential(
            Conv(in_channels + 64, in_channels, 1),
            nn.ReLU()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Prédire les heatmaps de landmarks
        landmark_maps = self.landmark_pred(x)  # B, num_landmarks*2, H, W
        
        # Calculer les centres de masse des heatmaps
        landmarks = self._compute_landmark_positions(landmark_maps)
        
        # Encoder les relations géométriques
        geometric_features = self._encode_geometric_relations(landmarks)
        
        # Étendre les features géométriques spatialement
        geom_expanded = geometric_features.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
        
        # Fusionner avec les features originales
        enhanced = torch.cat([x, geom_expanded], dim=1)
        output = self.fusion(enhanced)
        
        return output
    
    def _compute_landmark_positions(self, heatmaps):
        """Calcule les positions des landmarks à partir des heatmaps"""
        B, _, H, W = heatmaps.shape
        heatmaps = heatmaps.view(B, self.num_landmarks, 2, H, W)
        
        positions = []
        for i in range(self.num_landmarks):
            # Softmax spatial pour obtenir une distribution de probabilité
            hmap = heatmaps[:, i].sum(dim=1)  # Sum x,y channels
            hmap_flat = F.softmax(hmap.view(B, -1), dim=1).view(B, H, W)
            
            # Coordonnées pondérées
            x_coords = torch.arange(W, device=heatmaps.device).float()
            y_coords = torch.arange(H, device=heatmaps.device).float()
            x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='xy')
            
            x_pos = (hmap_flat * x_grid).sum(dim=[1, 2])
            y_pos = (hmap_flat * y_grid).sum(dim=[1, 2])
            
            positions.append(torch.stack([x_pos, y_pos], dim=1))
        
        return torch.stack(positions, dim=1)  # B, num_landmarks, 2
    
    def _encode_geometric_relations(self, landmarks):
        """Encode les distances entre paires de landmarks"""
        B = landmarks.shape[0]
        distances = []
        
        for i in range(self.num_landmarks):
            for j in range(i + 1, self.num_landmarks):
                dist = torch.norm(landmarks[:, i] - landmarks[:, j], dim=1)
                distances.append(dist)
        
        distances = torch.stack(distances, dim=1)  # B, num_pairs
        return self.geometric_encoder(distances)


class MixtureOfExpertsBlock(nn.Module):
    """Mixture of Experts pour gérer différentes échelles de visages"""
    
    def __init__(self, in_channels, out_channels, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        
        # Experts pour différentes échelles
        self.experts = nn.ModuleList([
            nn.Sequential(
                Conv(in_channels, out_channels, k=2*i+3, s=1, p=i+1),
                Conv(out_channels, out_channels, 1)
            ) for i in range(num_experts)
        ])
        
        # Router pour sélectionner les experts
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_experts, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Calculer les poids du router
        router_weights = self.router(x)  # B, num_experts, 1, 1
        
        # Appliquer chaque expert
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            out = expert(x)
            weighted = out * router_weights[:, i:i+1]
            expert_outputs.append(weighted)
        
        # Sommer les sorties pondérées
        return sum(expert_outputs)


class FaceDetect(nn.Module):
    """Head de détection spécialisé pour les visages avec landmarks"""
    
    def __init__(self, nc=1, num_landmarks=5, anchors=(), ch=()):
        super().__init__()
        self.nc = nc  # number of classes (1 for face)
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2 if anchors else 1  # number of anchors
        self.num_landmarks = num_landmarks
        
        # Outputs: x, y, w, h, objectness, landmarks (x,y) * 5
        self.no = 5 + num_landmarks * 2  # number of outputs per anchor
        
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        
        # Face-specific priors
        self.face_prior = nn.Parameter(torch.ones(1, self.na, 1, 1, self.no))
        
    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            # Appliquer les priors faciaux
            x[i] = x[i] * self.face_prior
            
        return x


# Export des modules
__all__ = ['C2fTransformer', 'FaceFeatureRefinement', 'GeometricConsistency', 
           'MixtureOfExpertsBlock', 'FaceDetect']
