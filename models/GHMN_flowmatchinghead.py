import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.constant_(self.mlp[0].bias, 0)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.constant_(self.mlp[2].bias, 0)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0,
                                                 end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class FlowMatchingHead(nn.Module):  
    def __init__(self, model_args, data_args):
        super().__init__()
        self.model_args = model_args
        self.data_args = data_args

        self.num_sampling_steps = model_args["flow_T"]  
        self.max_t = 1.0
        self.time_emb_dim = model_args["time_emb_dim"]  
        
        self.flow_net = SimpleMLPAdaLN(
            in_channels=1,  
            model_channels=model_args["flow_hidden_size"],  
            out_channels=1, 
            z_channels=1,
            num_res_blocks=model_args["flow_res_blocks"] 
        )

    def sample_time_and_noised_x(self, x_0, batch_sample_idx):
        BNC, T, _ = x_0.shape  
        device = x_0.device
        
        unique_sample_ids = torch.unique(batch_sample_idx).squeeze()  
        num_unique = unique_sample_ids.numel()  
        
        sample_t = torch.rand((num_unique,), device=device)  
        sample_t = sample_t.unsqueeze(1)  
        
        t_emb = torch.nn.Embedding.from_pretrained(sample_t, freeze=True)  
        
        t = t_emb(batch_sample_idx.squeeze())  
        
        t = t.unsqueeze(2)  
        
        noise = torch.randn_like(x_0)
        x_t = (1 - t) * noise + t * x_0 
        return x_t, t.squeeze(), noise  

    def flow_loss(self, x_0, cond_latent, batch_sample_idx):
        x_t, t, _ = self.sample_time_and_noised_x(x_0, batch_sample_idx)
        v_pred = self.predict_flow(x_t, t, cond_latent)  
        mse_loss = F.mse_loss(v_pred, x_0)
        return x_t, mse_loss


    def predict_flow(self, x_t, t, cond_latent):
        t = t * 1000 
        BNC, T, _ = x_t.shape
        input_feat_flat = x_t.reshape(BNC * T, -1)  
        cond_latent_flat = cond_latent.reshape(BNC * T, -1)
        v_pred_flat = self.flow_net(input_feat_flat, t, cond_latent_flat)  
        v_pred = v_pred_flat.reshape(BNC, T, 1)  
        return v_pred


    def sample(self, cond_latent, num_sample=1, return_all_samples=False):
        BNC, T, _ = cond_latent.shape
        device = next(self.parameters()).device
        all_samples = []
        
        with torch.no_grad():
            for _ in range(num_sample):
                noise = torch.randn((BNC, T, 1), device=device)  
                x_t = noise
                dt = 1.0 / self.num_sampling_steps  

                for step in range(self.num_sampling_steps):
                    t = torch.full((BNC,), step / self.num_sampling_steps, device=device)
                    v_t = self.predict_flow(x_t, t, cond_latent)
                    x_t = x_t + (v_t - noise) * dt

                all_samples.append(x_t)

            if return_all_samples:
                return torch.stack(all_samples, dim=0)  
            else:
                mean_sample = torch.mean(torch.stack(all_samples), dim=0)  
                return mean_sample


class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.input_proj = nn.Linear(in_channels, model_channels)
        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        
        self.res_blocks = nn.ModuleList([
            ResBlock(model_channels) for _ in range(num_res_blocks)
        ])
        
        self.final_layer = FinalLayer(model_channels, out_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, x, t, c):
        x = self.input_proj(x) 
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(c) 
        y = t_emb + c_emb  
        for block in self.res_blocks:
            x = block(x, y)  
        return self.final_layer(x, y)