import numpy as np
from collections.abc import Sequence
import math
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


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
    
def zeromodule(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def fourier_encode(x: torch.Tensor, num_freqs: int, max_freq: float, include_input: bool=False) -> torch.Tensor:
    """x: [B, C] -> [B, C*(2F) (+C if include_input)]，对数间隔频率 + 幅度归一"""
    B, C = x.shape
    freqs = torch.exp(torch.linspace(0.0, math.log(max_freq + 1e-6), num_freqs, device=x.device, dtype=x.dtype))
    xb = x.unsqueeze(-1) * (2.0 * math.pi) * freqs          # [B,C,F]
    enc = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1) # [B,C,2F]
    enc = enc.reshape(B, C * (2 * num_freqs)) / math.sqrt(num_freqs)
    return torch.cat([x, enc], dim=-1) if include_input else enc

class FourierSE3Embedder(nn.Module):
    """
    输入:  T:[B,4,4]
    流程:  R->6D，与 t(平移) 拼接 -> (可选 Fourier) -> MLP -> [B,d_model]
    """
    def __init__(
        self,
        state_dim: int,
        d_model: int,
        hidden: int = 256,
        num_freqs: int = 6,
        max_freq: float = 12.0,
        dropout: float = 0.0,
        ln_before: bool = True,       # 在MLP前做LayerNorm
    ):
        super().__init__()
        self.num_freqs = num_freqs
        self.max_freq = max_freq

        in_dim_raw = state_dim
        in_dim = (in_dim_raw * (2*num_freqs) + in_dim_raw)

        self.norm_in = nn.LayerNorm(in_dim) if ln_before else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

        # 小初始化，避免一开始过强
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        T: [B,44] -> [B,d_model]
        """
        x = fourier_encode(x, self.num_freqs, self.max_freq, include_input=True)  # [B, 9+18*num_freqs]

        x = self.norm_in(x)
        out = self.mlp(x)  # [B,d_model]
        return out

class DiffusionNet(nn.Module):
    def __init__(
        self,
        transition_dim,
        cond_dim,
        dim=256,
        num_blocks=3,
        act_fn=nn.Mish(),
        multi_head=False,
        split_obs=False
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
        self.multi_head = multi_head
        self.split_obs = split_obs
        if self.multi_head:
            self.action_dim = self.action_dim // 2
            self.transition_dim = self.transition_dim - self.action_dim
            self.mlp1 = nn.Sequential(
                            nn.Linear(embed_dim + self.transition_dim, 1024),
                            act_fn,
                            nn.Linear(1024, 512),
                            act_fn,
                            nn.Linear(512, 256),
                            act_fn,
                            zeromodule(nn.Linear(256, self.action_dim)),
                        )
            self.mlp2 = nn.Sequential(
                        nn.Linear(embed_dim + self.transition_dim, 1024),
                        act_fn,
                        nn.Linear(1024, 512),
                        act_fn,
                        nn.Linear(512, 256),
                        act_fn,
                        zeromodule(nn.Linear(256, self.action_dim)),
            )
        else:
            self.mlp = nn.Sequential(
                        nn.Linear(embed_dim + self.transition_dim, 1024),
                        act_fn,
                        nn.Linear(1024, 512),
                        act_fn,
                        nn.Linear(512, 256),
                        act_fn,
                        nn.Linear(256, self.action_dim),
                    )
        # self.mlp = MLPResNet(in_dim=embed_dim + transition_dim, out_dim=self.action_dim, num_blocks=num_blocks, act=nn.Mish(), hidden_dim=dim)

    def forward(self, x, time, cond, cond2=None):
        '''
            x : [ batch x action ]
            cond: [batch x state]
            cond2: [batch x state]
            returns : [batch x 1]
        '''
        t = self.time_mlp(time)

        if self.multi_head:
            inp1 = torch.cat([t, cond, x[:, :22]], dim=-1)
            if self.split_obs:
                inp2 = torch.cat([t, cond2, x[:, 22:]], dim=-1)
            else:
                inp2 = torch.cat([t, cond, x[:, 22:]], dim=-1)
            out1 = self.mlp1(inp1)
            out2 = self.mlp2(inp2)
            return out1, out2
        else:
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


from pql.models.pointnet import PointNetEncoderXYZ, MultiStagePointNetEncoder
from pql.models.visual import weight_init
class DiffusionPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, diffusion_iter, 
                 point_encoder_type="xyz", 
                 point_encoder_out_channels=256,
                 state_encoder_out_channels=256,
                 multi_head=False,
                 split_obs=False,
                 normalizer=None,
                 use_fourier=False,
                 device="cuda"):
        super().__init__()
        if isinstance(state_dim, Sequence):
            state_dim = state_dim[0]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.diffusion_iter = diffusion_iter
        self.device = device
        self.multi_head = multi_head
        self.split_obs = split_obs
        self.normalizer = normalizer
        if point_encoder_type == "xyz":
            self.point_encoder_func = PointNetEncoderXYZ
        elif point_encoder_type == "multistage":
            self.point_encoder_func = MultiStagePointNetEncoder
        else:
            raise ValueError(f"Invalid point encoder type: {point_encoder_type}")
        # self.obs_encoder = nn.Identity()
        self.point_encoder = self.point_encoder_func(out_channels=point_encoder_out_channels)
        self.point_encoder.apply(weight_init)
        if self.split_obs:
            state_dim = state_dim // 2
            # self.point_encoder1 = self.point_encoder_func(out_channels=point_encoder_out_channels)
            # self.point_encoder1.apply(weight_init)
            # self.point_encoder2 = self.point_encoder_func(out_channels=point_encoder_out_channels)
            # self.point_encoder2.apply(weight_init)
            if use_fourier:
                self.obs_encoder1 = FourierSE3Embedder(state_dim=state_dim, d_model=state_encoder_out_channels, num_freqs=6, max_freq=12.0)
                self.obs_encoder1.apply(weight_init)
                self.obs_encoder2 = FourierSE3Embedder(state_dim=state_dim, d_model=state_encoder_out_channels, num_freqs=6, max_freq=12.0)
                self.obs_encoder2.apply(weight_init)
            else:
                self.obs_encoder1 = nn.Sequential(nn.Linear(state_dim, state_encoder_out_channels), 
                                                nn.ReLU(inplace=True), 
                                                nn.Linear(state_encoder_out_channels, state_encoder_out_channels), 
                                                nn.LayerNorm(state_encoder_out_channels))
                self.obs_encoder1.apply(weight_init)
                self.obs_encoder2 = nn.Sequential(nn.Linear(state_dim, state_encoder_out_channels), 
                                            nn.ReLU(inplace=True), 
                                            nn.Linear(state_encoder_out_channels, state_encoder_out_channels), 
                                            nn.LayerNorm(state_encoder_out_channels))
                self.obs_encoder2.apply(weight_init)
        else:
            # self.point_encoder = self.point_encoder_func(out_channels=point_encoder_out_channels)
            # self.point_encoder.apply(weight_init)
            if use_fourier:
                self.obs_encoder = FourierSE3Embedder(state_dim=state_dim, d_model=state_encoder_out_channels, num_freqs=6, max_freq=12.0)
                self.obs_encoder.apply(weight_init)
            else:
                self.obs_encoder = nn.Sequential(nn.Linear(state_dim, state_encoder_out_channels), 
                                                nn.ReLU(inplace=True), 
                                                nn.Linear(state_encoder_out_channels, state_encoder_out_channels), 
                                                nn.LayerNorm(state_encoder_out_channels))
                self.obs_encoder.apply(weight_init)

        # init network
        self.net = DiffusionNet(
            transition_dim=point_encoder_out_channels + action_dim + state_encoder_out_channels,
            cond_dim=point_encoder_out_channels + state_encoder_out_channels,
            multi_head=multi_head
        )
        # self.net = DiffusionNet(
        #     transition_dim=action_dim + state_encoder_out_channels,
        #     cond_dim=state_encoder_out_channels,
        #     multi_head=multi_head
        # )

        # init noise scheduler
        # self.noise_scheduler = DDPMScheduler(
        #     num_train_timesteps=self.diffusion_iter,
        #     beta_schedule='squaredcos_cap_v2',
        #     # clip output to [-1,1] to improve stability
        #     clip_sample=True,
        #     prediction_type='epsilon'
        # )
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=15,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )

    def forward(self, img, state, pc, sample=True):
        return self.get_actions(img, state, pc, sample=sample)

    def get_actions(self, img, state, pc, sample=True):
        B = state.shape[0]
        point_feat = self.point_encoder(pc)
        if self.split_obs:
            # point_feat1 = self.point_encoder1(pc[:, :, :3])
            # point_feat2 = self.point_encoder2(pc[:, :, 3:])
            obs_feat1 = self.obs_encoder1(state[:, :22])
            obs_feat2 = self.obs_encoder2(state[:, 22:])
            point_state_feat = torch.cat([point_feat, obs_feat1], dim=-1)
            point_state_feat2 = torch.cat([point_feat, obs_feat2], dim=-1)
        else:
            obs_feat = self.obs_encoder(state)
            point_state_feat = torch.cat([point_feat, obs_feat], dim=-1)
        # point_state_feat = self.obs_encoder(state)
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
                point_state_feat,
                point_state_feat2 if self.split_obs else None
            )
            if self.multi_head:
                noise_pred1, noise_pred2 = noise_pred
                noise_pred = torch.cat([noise_pred1, noise_pred2], dim=-1)

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

        if self.normalizer is not None:
            noisy_action = self.normalizer['action'].unnormalize(noisy_action)

        return noisy_action
        
    def get_loss(self, img, state, pc, action, noise=None, timesteps=None):
        B = action.shape[0]
        if self.normalizer is not None:
            action = self.normalizer['action'].normalize(action)

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
        if self.split_obs:
            obs_feat1 = self.obs_encoder1(state[:, :22])
            obs_feat2 = self.obs_encoder2(state[:, 22:])
            point_state_feat = torch.cat([point_feat, obs_feat1], dim=-1)
            point_state_feat2 = torch.cat([point_feat, obs_feat2], dim=-1)
        else:
            obs_feat = self.obs_encoder(state)
            point_state_feat = torch.cat([point_feat, obs_feat], dim=-1)
        # point_state_feat = self.obs_encoder(state)
        # predict the noise residual
        noise_pred = self.net(
                noisy_action,
                timesteps,
                point_state_feat,
                point_state_feat2 if self.split_obs else None
            )
        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = action

        if self.multi_head:
            noise_pred1, noise_pred2 = noise_pred
            loss1 = nn.functional.mse_loss(noise_pred1, target[:, :22], reduction='none').mean(dim=1)  # [B]
            loss2 = nn.functional.mse_loss(noise_pred2, target[:, 22:], reduction='none').mean(dim=1)  # [B]
            loss = ((loss1 + loss2) / 2).mean()  # scalar 
            return loss, loss1.mean(), loss2.mean()
        else:
            loss = nn.functional.mse_loss(noise_pred, target)
            return loss
