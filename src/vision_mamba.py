import torch
import torch.nn as nn
import torch.nn.functional as F

class CorrectedMambaBlock(nn.Module):
    """Functional SSM-style block with Residuals and Depthwise Sequence Convolutions."""
    def __init__(self, d_model, expand=2):
        super().__init__()
        self.d_inner = d_model * expand

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # Depthwise 1D Convolution over spatial sequence tokens
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner
        )
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        # Linear projection & split into value and gate paths
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        # Sequence-wise convolution
        x_inner = x_inner.transpose(1, 2)
        x_inner = self.conv1d(x_inner)
        x_inner = x_inner.transpose(1, 2)

        # Non-linear gating mechanism
        out = self.act(x_inner) * F.silu(z)
        out = self.out_proj(out)

        # Residual Connection
        return residual + out

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class VisionMamba(nn.Module):
    def __init__(self, in_channels=3, num_diseases=38, img_size=256, patch_size=16, embed_dim=192, depth=4, num_classes=None, **kwargs):
        super().__init__()
        num_diseases = num_classes if num_classes is not None else num_diseases
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList([
            CorrectedMambaBlock(d_model=embed_dim) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_diseases)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        # Global Average Pooling over all spatial patches
        x = x.mean(dim=1)
        return self.classifier(x)
    


