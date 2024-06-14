import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class Diffusion(nn.Module):

    def __init__(self):
        self.time_embeeding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)


    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent = (B, 4, H/8, W/8)
        # context = (B, Seq_len, Dim)
        # time (1, 320)

        