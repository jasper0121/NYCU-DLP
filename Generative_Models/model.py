import torch
import torch.nn as nn
from diffusers import UNet2DModel

class ConditionalDDPMModel(nn.Module):
    def __init__(self, image_size=64, in_channels=3, out_channels=3, cond_dim=24, embedding_dim=256):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim

        self.class_embedding = nn.Linear(cond_dim, embedding_dim)

        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512, 256, 128),
            down_block_types=[
                "DownBlock2D",      # 64→128
                "DownBlock2D",      # 128→256
                "AttnDownBlock2D",  # 256→512 + Attention
                "DownBlock2D",      # 512→256
                "DownBlock2D",      # 256→128
                "DownBlock2D",      # 128→64
            ],
            up_block_types=[
                "UpBlock2D",        # 128→256 (skip from down4)
                "UpBlock2D",        # 256→512 (skip from down3)
                "AttnUpBlock2D",    # 512→256 + Attention
                "UpBlock2D",        # 256→128 (skip from down2)
                "UpBlock2D",        # 128→64  (skip from down1)
                "UpBlock2D",        # 64→out  (skip from input)
            ],
            class_embed_type="identity",
            time_embedding_type="positional"
        )

    def forward(self, x, t, cond):
        class_embed = self.class_embedding(cond.float())
        return self.unet(x, t, class_embed).sample
