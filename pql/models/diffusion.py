import numpy as np
from collections.abc import Sequence
import math
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class DiffusionNet(nn.Module):
    def __init__(
        self,
        transition_dim,
        cond_dim,
        dim=256,
        num_blocks=3,
        act_fn=nn.Mish()
    ):
        super().__init__()

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.transition_dim = transition_dim
        self.action_dim = transition_dim - cond_dim

        embed_dim = dim

        self.mlp = nn.Sequential(
                        nn.Linear(embed_dim + transition_dim, 1024),
                        act_fn,
                        nn.Linear(1024, 512),
                        act_fn,
                        nn.Linear(512, 256),
                        act_fn,
                        nn.Linear(256, self.action_dim),
                    )
        
        # self.mlp = MLPResNet(in_dim=embed_dim + transition_dim, out_dim=self.action_dim, num_blocks=num_blocks, act=nn.Mish(), hidden_dim=dim)

    def forward(self, x, time, cond):
        '''
            x : [ batch x action ]
            cond: [batch x state]
            returns : [batch x 1]
        '''
        t = self.time_mlp(time)

        inp = torch.cat([t, cond, x], dim=-1)
        out = self.mlp(inp)

        return out


class MLPResNetBlock(nn.Module):
    """MLPResNet block."""
    def __init__(self, features, act, dropout_rate=None, use_layer_norm=False):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.features = features
        self.act = act

        self.dense1 = nn.Linear(features, features * 4)
        self.dense2 = nn.Linear(features * 4, features)
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(features)
        if self.dropout_rate is not None and self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        
    def forward(self, x):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0.0:
            x = self.dropout(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        x = self.dense1(x)
        x = self.act(x)
        x = self.dense2(x)

        if residual.shape != x.shape:
            residual = self.dense2(residual)

        return residual + x


class MLPResNet(nn.Module):
    def __init__(self, num_blocks, in_dim, out_dim, dropout_rate=0.1, use_layer_norm=True, hidden_dim=256, act=None):
        super().__init__()
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.act = act

        self.dense1 = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([MLPResNetBlock(hidden_dim, act, dropout_rate, use_layer_norm) for _ in range(num_blocks)])
        self.dense2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.dense1(x)
        for block in self.blocks:
            x = block(x)
        x = self.act(x)
        x = self.dense2(x)
        return x


from pql.models.pointnet import PointNetEncoderXYZ
from pql.models.visual import weight_init
class DiffusionPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, diffusion_iter, device="cuda"):
        super().__init__()
        if isinstance(state_dim, Sequence):
            state_dim = state_dim[0]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.diffusion_iter = diffusion_iter
        self.device = device
        self.point_encoder = PointNetEncoderXYZ(in_channels=3,
                                                out_channels=256,
                                                use_layernorm=True,
                                                final_norm='layernorm',
                                                use_projection=True)
        self.point_encoder.apply(weight_init)
        self.obs_encoder = nn.Identity()
        self.obs_encoder.apply(weight_init)

        # init network
        self.net = DiffusionNet(
            transition_dim=self.point_encoder.out_channels + action_dim + state_dim,
            cond_dim=self.point_encoder.out_channels + state_dim)

        # init noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.diffusion_iter,
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            prediction_type='epsilon'
        )

    def forward(self, img, state, pc, sample=True):
        return self.get_actions(img, state, pc, sample=sample)

    def get_actions(self, img, state, pc, sample=True):
        B = state.shape[0]
        point_feat = self.point_encoder(pc)
        obs_feat = self.obs_encoder(state)
        point_state_feat = torch.cat([point_feat, obs_feat], dim=-1)
        # init action from Guassian noise
        noisy_action = torch.randn(
            (B, self.action_dim), device=self.device)
        # init scheduler
        self.noise_scheduler.set_timesteps(self.diffusion_iter)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            timesteps = torch.ones(B, device=self.device) * k
            
            noise_pred = self.net(
                noisy_action,
                timesteps,
                point_state_feat
            )
            if sample:
                noise_pred = noise_pred.detach()

            # inverse diffusion step (remove noise)
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_action
            ).prev_sample
            if sample:
                noisy_action = noisy_action.detach()

        return noisy_action
        
    def get_loss(self, img, state, pc, action, noise=None, timesteps=None):
        B = action.shape[0]
        # sample noise to add to actions
        if noise is None:
            noise = torch.randn_like(action, device=self.device)
        
        # sample a diffusion iteration for each data point
        if timesteps is None:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=self.device
            ).long()

        # add noise at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action = self.noise_scheduler.add_noise(
            action, noise, timesteps)
        point_feat = self.point_encoder(pc)
        obs_feat = self.obs_encoder(state)
        point_state_feat = torch.cat([point_feat, obs_feat], dim=-1)

        # predict the noise residual
        noise_pred = self.net(
                noisy_action,
                timesteps,
                point_state_feat
            )
            
        # L2 loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss