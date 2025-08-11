import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34
from torchvision import transforms
from torch.distributions import Independent, Normal


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        # cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        # cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, print(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        self.out_channels = out_channels
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            # cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class MultiStagePointNetEncoder(nn.Module):
    def __init__(self, h_dim=128, out_channels=128, num_layers=4, **kwargs):
        super().__init__()

        self.h_dim = h_dim
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=False)

        self.conv_in = nn.Conv1d(3, h_dim, kernel_size=1)
        self.layers, self.global_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))
        self.conv_out = nn.Conv1d(h_dim * self.num_layers, out_channels, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2) # [B, N, 3] --> [B, 3, N]
        y = self.act(self.conv_in(x))
        feat_list = []
        for i in range(self.num_layers):
            y = self.act(self.layers[i](y))
            y_global = y.max(-1, keepdim=True).values
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            y = self.act(self.global_layers[i](y))
            feat_list.append(y)
        x = torch.cat(feat_list, dim=1)
        x = self.conv_out(x)

        x_global = x.max(-1).values

        return x_global


class DINOEncoder(nn.Module):
    def __init__(self, width=128, height=128, num_cams=2):
        super(DINOEncoder, self).__init__()
        self.num_cams = num_cams
        self.model = torch.hub.load('facebookresearch/dinov2', "dinov2_vits14")
        for param in self.model.parameters():
            param.requires_grad = False
        self.repr_dim = 384
        self.aug = RandomShiftsAug(pad=4)

    def forward_single(self, x, aug=False):
        if aug:
            x = self.aug(x)
        x = self.model(x)
        return x

    def forward(self, obs, aug=False):
        assert obs.shape[1] == self.num_cams
        x = self.forward_single(obs[:, 0], aug=aug)
        for i in range(1, obs.shape[1]):
            x = torch.cat([x, self.forward_single(obs[:, i], aug=aug)], dim=-1)
        return x

class ResEncoder(nn.Module):
    def __init__(self, width=128, height=128, num_cams=2):
        super(ResEncoder, self).__init__()
        self.num_cams = num_cams
        self.model = resnet18(pretrained=True)
        # self.transform = transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224)
        #     ])

        # for param in self.model.parameters():
        #     param.requires_grad = False

        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.repr_dim = 1024
        self.image_channel = 3
        x = torch.randn([32] + [self.image_channel, width, height])
        with torch.no_grad():
            out_shape = self.forward_conv(x).shape
        self.out_dim = out_shape[1]
        self.fc = nn.Linear(self.out_dim, self.repr_dim)
        self.ln = nn.LayerNorm(self.repr_dim)
        #
        # # Initialization
        # nn.init.orthogonal_(self.fc.weight.data)
        # self.fc.bias.data.fill_(0.0)

        self.aug = RandomShiftsAug(pad=4)

    @torch.no_grad()
    def forward_conv(self, obs, flatten=True, aug=False):
        if aug:
            obs = self.aug(obs)
        # obs = obs / 255.0 - 0.5
        time_step = obs.shape[1] // self.image_channel
        obs = obs.reshape(obs.shape[0], time_step, self.image_channel, obs.shape[-2], obs.shape[-1])
        obs = obs.reshape(obs.shape[0] * time_step, self.image_channel, obs.shape[-2], obs.shape[-1])

        for name, module in self.model._modules.items():
            obs = module(obs)
            if name == 'layer2':
                break

        conv = obs.view(obs.size(0) // time_step, time_step, obs.size(1), obs.size(2), obs.size(3))
        conv_current = conv[:, 1:, :, :, :]
        conv_prev = conv_current - conv[:, :time_step - 1, :, :, :].detach()
        conv = torch.cat([conv_current, conv_prev], axis=1)
        conv = conv.view(conv.size(0), conv.size(1) * conv.size(2), conv.size(3), conv.size(4))
        if flatten:
            conv = conv.view(conv.size(0), -1)

        return conv

    def forward_single(self, obs, aug=False):
        conv = self.forward_conv(obs, aug=aug)
        out = self.fc(conv)
        out = self.ln(out)
        return out
    
    def forward(self, obs, aug=False):
        assert obs.shape[1] == self.num_cams
        x = self.forward_single(obs[:, 0], aug=aug)
        for i in range(1, obs.shape[1]):
            x = torch.cat([x, self.forward_single(obs[:, i], aug=aug)], dim=-1)
        return x
    

class DiagGaussianMLPVPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, repr_dim=1024, feature_dim=1024, hidden_dim=512,
                 init_log_std=0., num_cams=2, width=128, height=128, encoder_type='resnet'):
        super().__init__()
        if encoder_type == 'resnet':
            self.encoder = ResEncoder(width=width, height=height, num_cams=num_cams)
        elif encoder_type == 'dino':
            self.encoder = DINOEncoder(width=width, height=height, num_cams=num_cams)
        elif encoder_type is None:
            self.encoder = None
        else:
            raise ValueError(f"Invalid encoder type: {encoder_type}")
        if self.encoder is not None:
            self.trunk = nn.Sequential(nn.Linear(self.encoder.repr_dim * num_cams, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.ReLU(inplace=True))
            self.trunk.apply(weight_init)
            input_dim = feature_dim
        else:
            input_dim = 0
            
        self.point_encoder = PointNetEncoderXYZ(in_channels=3,
                                                out_channels=64,
                                                use_layernorm=True,
                                                final_norm='layernorm',
                                                use_projection=True)
        input_dim += obs_dim + self.point_encoder.out_channels
        self.policy = nn.Sequential(nn.Linear(input_dim, hidden_dim),  # feature_dim + 
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, act_dim))

        self.point_encoder.apply(weight_init)
        self.policy.apply(weight_init)

        self.logstd = nn.Parameter(torch.full((act_dim,), init_log_std))

    def forward(self, img, state, pc=None, sample=True, aug=False):
        return self.get_actions(img, state, pc=pc, sample=sample, aug=aug)[0]

    def get_actions(self, img, state, pc=None, sample=True, aug=False):
        if self.encoder is not None:
            x = self.encoder(img, aug=aug)
            h = self.trunk(x)
            h = torch.cat([h, state], dim=-1)
        else:
            h = state
        if pc is not None:
            pc = self.point_encoder(pc)
            h = torch.cat([h, pc], dim=-1)
        mean = self.policy(h)
        log_std = self.logstd.expand_as(mean)
        std = torch.exp(log_std)
        action_dist = Independent(Normal(loc=mean, scale=std), 1)
        if sample:
            actions = action_dist.rsample()
        else:
            actions = mean
        return actions, action_dist

    def get_actions_logprob_entropy(self, img, state, pc=None, sample=True, aug=False):
        actions, action_dist = self.get_actions(img, state, pc=pc, sample=sample, aug=aug)
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return actions, action_dist, log_prob, entropy

    def logprob_entropy(self, img, state, actions, pc=None, aug=False):
        _, action_dist = self.get_actions(img, state, pc=pc, aug=aug)
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return actions, action_dist, log_prob, entropy