import torch
import torch.nn as nn
import lightning as L
from torch.utils.checkpoint import checkpoint
from xformers.ops import memory_efficient_attention

class GyroBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, -1).permute(2, 0, 1, 3, 4)
        # Memory-Efficient Attention is the key for 6GB GPUs
        attn = memory_efficient_attention(qkv[0], qkv[1], qkv[2]).reshape(B, N, C)
        return self.norm(x + attn)

class ScalableGyroNet(L.LightningModule):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embed = nn.Linear(1, embed_dim)
        self.blocks = nn.ModuleList([GyroBlock(embed_dim) for _ in range(2)])
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            # Gradient Checkpointing saves ~50% VRAM
            x = checkpoint(block, x, use_reentrant=False)
        return self.head(x.mean(dim=1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
