import numpy as np
from collections.abc import Sequence
import math
import torch
import torch.nn as nn
import escnn
from escnn.nn import FieldType
from pql.models.emlp import EMLP, EMLPNew
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
    

class EquivariantDiffusionNet(nn.Module):
    def __init__(
        self,
        G, 
        input_fields, 
        output_fields, 
        in_dim, 
        out_dim,
        dim=256,
        act_fn=nn.Mish()
    ):
        super().__init__()

        self.time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        # generate the field types
        gspace = escnn.gspaces.no_base_space(G)
        in_field = [G.representations["irrep_0"] for _ in range(dim)] + [G.representations[rep] for rep in input_fields] + [G.representations[rep] for rep in output_fields]
        self.in_field_type = FieldType(gspace, in_field)
        assert self.in_field_type.size == dim + in_dim + out_dim, f"in_dim {in_dim} does not match the size of the input field type {self.in_field_type.size}"
        self.out_field_type = FieldType(gspace, [G.representations[rep] for rep in output_fields])
        assert self.out_field_type.size == out_dim, f"out_dim {out_dim} does not match the size of the output field type {self.out_field_type.size}"
        
        self.mlp = EMLPNew(in_type=self.in_field_type,
                        out_type=self.out_field_type,
                        hidden_layers=4,              # Input layer + 3 hidden layers + output/head layer
                        hidden_units=[1024, 512, 512, 256],
                        bias=True             # Use bias in the linear layers
                        )
        
    def forward(self, x, time, cond):
        '''
            x : [ batch x action ]
            cond: [batch x state]
            returns : [batch x 1]
        '''
        t = self.time_mlp(time)

        inp = torch.cat([t, cond, x], dim=-1)
        inp = self.in_field_type(inp)
        out = self.mlp(inp).tensor

        return out
    

class EquivariantDiffusionPolicy(nn.Module):
    def __init__(self, G, input_fields, output_fields, in_dim, out_dim, diffusion_iter, device="cuda"):
        super().__init__()
        self.action_dim = out_dim
        self.diffusion_iter = diffusion_iter
        self.device = device

        # init network
        self.net = EquivariantDiffusionNet(
            G, input_fields, output_fields, in_dim, out_dim,
        )

        # init noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.diffusion_iter,
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            prediction_type='epsilon'
        )

    def forward(self, x, sample=True):
        return self.get_actions(x, sample=True)

    def get_actions(self, state, sample=True):
        B = state.shape[0]
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
                state
            )
            # print(f"noise_pred: {noise_pred}")
            if sample:
                noise_pred = noise_pred.detach()

            # inverse diffusion step (remove noise)
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_action
            ).prev_sample
            # print(f"noisy_action: {noisy_action}")
            if sample:
                noisy_action = noisy_action.detach()

        return noisy_action
        
    def get_loss(self, state, action, noise=None, timesteps=None):
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

        # predict the noise residual
        noise_pred = self.net(
                noisy_action,
                timesteps,
                state
            )
            
        # L2 loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss