"""
Modules personnalisés pour YOLOv12-Face Enhanced
Basé sur les innovations 2025 en vision transformers et attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, C3, C2f, SPPF, Concat, autopad
import math


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module pour améliorer la détection des petits visages
    Inspiré par les travaux sur micro-expression detection (2025)
    """
    def __init__(self, c1, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Calcul des statistiques spatiales
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_spatial = torch.cat([avg_out, max_out], dim=1)
        
        # Génération du masque d'attention
        attention = self.sigmoid(self.conv(x_spatial))
        return x * attention


class A2Module(nn.Module):
    """
    Area Attention Module (A2) - Innovation clé de YOLOv12
    Divise la feature map en zones pour préserver le receptive field
    tout en réduisant la complexité computationnelle
    """
    def __init__(self, c1, num_areas=4):
        super().__init__()
        self.num_areas = num_areas
        self.area_size = int(math.sqrt(num_areas))
        
        # Attention pour chaque zone
        self.area_attentions = nn.ModuleList([
            nn.MultiheadAttention(c1, num_heads=8, batch_first=True)
            for _ in range(num_areas)
        ])
        
        # Fusion des zones
        self.fusion = Conv(c1 * num_areas, c1, 1)
        self.norm = nn.LayerNorm(c1)

    def forward(self, x):
        B, C, H, W = x.shape
        area_h, area_w = H // self.area_size, W // self.area_size
        
        # Diviser en zones
        areas = []
        for i in range(self.area_size):
            for j in range(self.area_size):
                area = x[:, :, i*area_h:(i+1)*area_h, j*area_w:(j+1)*area_w]
                area_flat = area.flatten(2).transpose(1, 2)  # B, HW, C
                
                # Appliquer l'attention à cette zone
                idx = i * self.area_size + j
                attn_out, _ = self.area_attentions[idx](area_flat, area_flat, area_flat)
                attn_out = self.norm(attn_out + area_flat)  # Residual connection
                
                # Reshape
                attn_out = attn_out.transpose(1, 2).reshape(B, C, area_h, area_w)
                areas.append(attn_out)
        
        # Reconstruire la feature map
        rows = []
        for i in range(self.area_size):
            row = torch.cat(areas[i*self.area_size:(i+1)*self.area_size], dim=3)
            rows.append(row)
        x_areas = torch.cat(rows, dim=2)
        
        return x_areas


class RELAN(nn.Module):
    """
    Residual Efficient Layer Aggregation Network
    Améliore la stabilité d'entraînement avec des connexions résiduelles
    au niveau des blocs
    """
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        
        # Blocs avec connexions résiduelles
        self.m = nn.Sequential(*(
            ResidualBlock(c_, c_, shortcut) for _ in range(n)
        ))
        
        # Scaling factor pour la stabilité
        self.scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.m(self.cv2(x))
        out = self.cv3(torch.cat((y1, y2), 1))
        
        # Residual connection avec scaling
        if x.shape[1] == out.shape[1]:
            out = out + self.scale * x
            
        return out


class ResidualBlock(nn.Module):
    """Bloc résiduel pour RELAN"""
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.conv1 = Conv(c1, c2, 3, 1)
        self.conv2 = Conv(c2, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class FlashAttention(nn.Module):
    """
    Flash Attention optimisé pour GPU
    Réduit l'utilisation mémoire et améliore la vitesse
    """
    def __init__(self, c1, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = c1 // num_heads
        
        self.qkv = nn.Linear(c1, 3 * c1, bias=False)
        self.proj = nn.Linear(c1, c1)
        self.norm = nn.LayerNorm(c1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        # Flatten spatial dimensions
        x_flat = x.flatten(2).transpose(1, 2)  # B, N, C
        
        # QKV projection
        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, H, N, D
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention (optimisé pour mémoire)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        
        # Utiliser Flash Attention si disponible
        try:
            from flash_attn import flash_attn_qkvpacked_func
            # Code Flash Attention optimisé
            attn = flash_attn_qkvpacked_func(qkv.permute(1, 2, 0, 3, 4))
        except:
            # Fallback vers attention standard
            attn = F.softmax(attn, dim=-1)
            attn = attn @ v
        
        # Reshape et projection
        attn = attn.transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(attn)
        x_attn = self.norm(x_attn + x_flat)  # Residual
        
        # Reshape vers format spatial
        return x_attn.transpose(1, 2).reshape(B, C, H, W)


class C2PSA(nn.Module):
    """
    Cross-Stage Partial with Spatial Attention
    Innovation de YOLOv11 pour améliorer la détection des petits objets
    """
    def __init__(self, c1, n=1):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        
        # Spatial attention branch
        self.spatial_attn = nn.Sequential(
            Conv(c_, c_ // 4, 1),
            nn.ReLU(),
            Conv(c_ // 4, c_, 1),
            nn.Sigmoid()
        )
        
        # Feature processing
        self.m = nn.Sequential(*(
            C2f(c_, c_, 1, shortcut=True) for _ in range(n)
        ))
        
        self.cv3 = Conv(2 * c_, c1, 1)
        
    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        
        # Appliquer spatial attention
        attn = self.spatial_attn(y2)
        y2 = y2 * attn
        
        # Process features
        y2 = self.m(y2)
        
        return self.cv3(torch.cat((y1, y2), 1))


class CrossScaleAttention(nn.Module):
    """
    Attention cross-scale pour fusionner les features multi-échelles
    Améliore la détection à différentes tailles de visages
    """
    def __init__(self, c1):
        super().__init__()
        self.conv_query = Conv(c1, c1 // 8, 1)
        self.conv_key = Conv(c1, c1 // 8, 1)
        self.conv_value = Conv(c1, c1, 1)
        self.conv_out = Conv(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Générer Q, K, V
        query = self.conv_query(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.conv_key(x).view(B, -1, H * W)
        value = self.conv_value(x).view(B, -1, H * W)
        
        # Attention
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.conv_out(out)
        
        # Weighted sum avec input
        return self.gamma * out + x


class MicroExpressionAttention(nn.Module):
    """
    Module d'attention spécialisé pour les micro-expressions faciales
    Basé sur les recherches 2025 en reconnaissance de micro-expressions
    """
    def __init__(self, c1):
        super().__init__()
        # Attention temporelle et spatiale combinée
        self.temporal_conv = nn.Conv2d(c1, c1, (3, 3), padding=1, groups=c1)
        self.spatial_conv = nn.Conv2d(c1, c1, (1, 1))
        
        # Attention par régions faciales
        self.region_attention = nn.ModuleList([
            Conv(c1, c1 // 4, 1) for _ in range(4)  # 4 régions principales du visage
        ])
        
        self.fusion = Conv(c1 + c1 // 4 * 4, c1, 1)
        
    def forward(self, x):
        # Caractéristiques temporelles (mouvement subtil)
        temporal_feat = self.temporal_conv(x)
        
        # Caractéristiques spatiales
        spatial_feat = self.spatial_conv(x)
        
        # Attention par régions
        B, C, H, W = x.shape
        h_mid, w_mid = H // 2, W // 2
        
        regions = [
            x[:, :, :h_mid, :w_mid],      # Haut gauche (œil)
            x[:, :, :h_mid, w_mid:],       # Haut droit (œil)
            x[:, :, h_mid:, :w_mid],       # Bas gauche (bouche)
            x[:, :, h_mid:, w_mid:]        # Bas droit (bouche)
        ]
        
        region_feats = []
        for i, (region, attn_module) in enumerate(zip(regions, self.region_attention)):
            feat = attn_module(region)
            # Upscale to original size
            feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            region_feats.append(feat)
        
        # Fusion finale
        all_feats = [temporal_feat + spatial_feat] + region_feats
        return self.fusion(torch.cat(all_feats, 1))


class AdaptiveDetect(nn.Module):
    """
    Tête de détection adaptive multi-échelles
    Ajuste dynamiquement les anchors selon les statistiques du dataset
    """
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), 
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),
                Conv(c3, c3, 3),
                nn.Conv2d(c3, self.nc, 1)
            ) for x in ch
        )
        self.dfl = nn.Conv2d(self.reg_max, 1, 1, bias=False)
        
        # Adaptive anchor adjustment
        self.anchor_stats = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, 1, 1)) for _ in range(self.nl)
        ])

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            # Ajustement adaptatif des prédictions selon les statistiques
            x[i] = torch.cat((self.cv2[i](x[i]) * self.anchor_stats[i], 
                             self.cv3[i](x[i])), 1)
        return x


# Enregistrer les modules pour Ultralytics
__all__ = [
    'SpatialAttention', 'A2Module', 'RELAN', 'FlashAttention',
    'C2PSA', 'CrossScaleAttention', 'MicroExpressionAttention',
    'AdaptiveDetect', 'ResidualBlock'
]
