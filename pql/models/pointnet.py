import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
from termcolor import cprint
from typing import Optional, Dict, Tuple, Union, List, Type

def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out

def maxpool(x, dim=-1, keepdim=False):
    out = x.max(dim=dim, keepdim=keepdim).values
    return out

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


class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=True,
                 final_norm: str='layernorm',
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


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules

__all__ = [
    "MSG_CFG", "SharedMLP",
    "PointnetSAModuleMSG_Torch", "PointNet2MSGEncoderTorch", "PointNet2ClsMSGTorch"
]

# --- same defaults as your P3D file ---
MSG_CFG = {
    "NPOINTS": [512, 256, 128, 64],
    "RADIUS":  [[0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [0.16, 0.32]],
    "NSAMPLE": [[16, 32],     [16, 32],     [16, 32],     [16, 32]],
    # "MLPS": [
    #     [[32, 64],   [32, 64]],
    #     [[64, 128],  [64, 128]],
    #     [[128, 256], [128, 256]],
    #     [[256, 256], [256, 512]],
    # ],
    "MLPS": [
        [[32, 64],   [32, 64]],
        [[64, 128],  [64, 128]],
        [[128, 256], [128, 256]],
        [[256, 128], [256, 128]],  # instead of 256,256 and 256,512
    ],
}

# ---------------- Utilities ----------------

@torch.no_grad()
def voxel_downsample(xyz: torch.Tensor, voxel_size: Optional[float]) -> torch.Tensor:
    """
    xyz: [B,N,3] float
    Return: downsampled xyz per batch (ragged kept by masking & concat)
    For simplicity, we return the same shape per-batch by selecting unique voxel centers
    independently and concatenating (you will usually call this before SA stages).
    """
    if (voxel_size is None) or (voxel_size <= 0):
        return xyz
    B, N, _ = xyz.shape
    outs = []
    for b in range(B):
        pts = xyz[b]  # [N,3]
        grid = torch.floor(pts / voxel_size).to(torch.int64)    # voxel hash
        # pack 3D grid coords into a single int key (no collisions for typical ranges)
        key = (grid[:, 0] << 42) + (grid[:, 1] << 21) + grid[:, 2]
        uniq, inv = torch.unique(key, return_inverse=True)
        # choose first index in each voxel (could also take mean by segment mean)
        # here we pick representative points to maintain input identity
        # build index of first occurrence
        first_idx = torch.zeros_like(uniq, dtype=torch.long)
        # scatter first occurrence
        seen = {}
        # A vectorized way to get first occurrence:
        # sort by inv, then pick boundaries
        order = torch.argsort(inv)
        inv_sorted = inv[order]
        boundary = torch.ones_like(inv_sorted, dtype=torch.bool)
        boundary[1:] = inv_sorted[1:] != inv_sorted[:-1]
        first_occ = order[boundary]
        ds = pts[first_occ]  # [M,3]
        outs.append(ds.unsqueeze(0))
    # Ragged batches are OK for SA, but we need tensors. We’ll just return a list-concat and
    # let SA do FPS to a fixed npoint. To keep API simple, we return a padded tensor to max M.
    maxM = max(x.size(1) for x in outs)
    out = []
    for ds in outs:
        M = ds.size(1)
        if M < maxM:
            pad = ds[:, :1].expand(1, maxM - M, 3)  # repeat a point
            out.append(torch.cat([ds, pad], dim=1))
        else:
            out.append(ds[:, :maxM])
    return torch.cat(out, dim=0)


@torch.no_grad()
def batched_fps(xyz: torch.Tensor, K: int) -> torch.Tensor:
    """
    Farthest Point Sampling in pure torch.
    xyz: [B,N,3]
    return new_xyz: [B,K,3]
    Complexity O(B*N*K). Works well for K up to ~1-2k.
    """
    B, N, _ = xyz.shape
    device = xyz.device
    # Initialize indices and distances
    centroids = torch.zeros(B, K, dtype=torch.long, device=device)
    distances = torch.full((B, N), float('inf'), device=device)
    # pick initial farthest randomly (or the first)
    farthest = torch.randint(0, N, (B,), device=device)

    batch_indices = torch.arange(B, device=device)
    for i in range(K):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].view(B, 1, 3)  # [B,1,3]
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)        # [B,N]
        distances = torch.minimum(distances, dist)
        farthest = torch.max(distances, dim=-1).indices
    return xyz.gather(1, centroids.unsqueeze(-1).expand(B, K, 3))


def _chunked_cdist2(a: torch.Tensor, b: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    """
    Compute squared distances between a=[B,Q,3] and b=[B,N,3] in chunks over N to limit memory.
    Returns: [B,Q,N] (squared distances).
    NOTE: Use prudent chunk sizes; on very large N you may want 8192 or 4096.
    """
    B, Q, _ = a.shape
    _, N, _ = b.shape
    device = a.device
    out = []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        # [B,Q,3] vs [B,Ne,3] -> [B,Q,Ne]
        sub = torch.sum((a.unsqueeze(2) - b[:, s:e].unsqueeze(1)) ** 2, dim=-1)
        out.append(sub)
    return torch.cat(out, dim=-1)


@torch.no_grad()
def ball_query_torch(new_xyz: torch.Tensor,
                     xyz: torch.Tensor,
                     radius: float,
                     nsample: int,
                     cdist_chunk_size: int = 16384) -> torch.Tensor:
    """
    new_xyz: [B, Q, 3]  (centers)
    xyz:     [B, N, 3]  (all points)
    Return: idx [B, Q, nsample] with -1 padding where neighbors < nsample.
    Memory-safe via chunked cdist.
    """
    B, Q, _ = new_xyz.shape
    _, N, _ = xyz.shape

    d2 = _chunked_cdist2(new_xyz, xyz, chunk=cdist_chunk_size)  # [B,Q,N]
    within = d2 <= (radius * radius)                            # [B,Q,N]
    # Put large distance for out-of-radius, then take topk smallest distances
    masked_d2 = d2.masked_fill(~within, float('inf'))           # [B,Q,N]
    # Get up to nsample nearest within radius (topk on negative for smallest)
    # Use torch.topk on -masked_d2 to get smallest distances at front.
    # If fewer than nsample exist, topk still returns nsample but many will be inf.
    neg = -masked_d2
    vals, idx = torch.topk(neg, k=nsample, dim=-1)              # [B,Q,nsample]
    # Replace inf (no neighbor) with -1 index
    no_neighbor = torch.isinf(-vals)                            # places where masked_d2 == inf
    idx = idx.masked_fill(no_neighbor, -1)
    return idx

# ---------------- Core blocks ----------------

class SharedMLP(nn.Module):
    def __init__(self, channels: List[int], bn: bool = True):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PointnetSAModuleMSG_Torch(nn.Module):
    def __init__(
        self,
        npoint: Optional[int],
        radii: List[float],
        nsamples: List[int],
        mlps: List[List[int]],
        use_xyz: bool = True,
        bn: bool = True,
        voxel_size: Optional[float] = None,
        cdist_chunk_size: int = 16384,
    ):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint, self.radii, self.nsamples = npoint, radii, nsamples
        self.use_xyz, self.bn = use_xyz, bn
        self.voxel_size = voxel_size
        self.cdist_chunk_size = cdist_chunk_size
        self.mlps = nn.ModuleList([SharedMLP(mlp, bn=bn) for mlp in mlps])

    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor]):
        # —— xyz 可能会被体素下采样修改形状，所以 B,N,_ 要在修改后重取 —— 
        if self.voxel_size is not None and self.voxel_size > 0:
            xyz = voxel_downsample(xyz, self.voxel_size)

        # 这里重新获取形状，千万不要用前面缓存的 N
        B, N, _ = xyz.shape

        # FPS
        if self.npoint is None:
            new_xyz = xyz
        else:
            new_xyz = batched_fps(xyz, self.npoint)
        npoint = new_xyz.shape[1]

        outs = []
        for radius, nsample, mlp in zip(self.radii, self.nsamples, self.mlps):
            idx = ball_query_torch(new_xyz, xyz, radius, nsample, self.cdist_chunk_size)  # [B,npoint,nsample]

            # ---- gather xyz ----
            idx_xyz = idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, 3)
            # 这里用 -1 保持其他维度不变，且在 voxel 后 N 已经是最新的
            xyz_exp = xyz.unsqueeze(1).expand(-1, npoint, -1, -1)  # [B, npoint, N, 3]
            grouped_xyz = torch.gather(xyz_exp, 2, idx_xyz)        # [B, npoint, nsample, 3]
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)

            invalid = (idx < 0).unsqueeze(-1)
            grouped_xyz = grouped_xyz.masked_fill(invalid, 0.0)

            # ---- gather features（如果有）----
            if features is not None:
                feat_NC = features.transpose(1, 2)                 # [B, N, Cin]
                Cin = feat_NC.shape[-1]
                feat_exp = feat_NC.unsqueeze(1).expand(-1, npoint, -1, -1)  # [B, npoint, N, Cin]
                idx_feat = idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, Cin)
                grouped_feat = torch.gather(feat_exp, 2, idx_feat)  # [B, npoint, nsample, Cin]
                grouped_feat = grouped_feat.masked_fill(invalid, 0.0)
                grouped = torch.cat([grouped_feat, grouped_xyz], dim=-1) if self.use_xyz else grouped_feat
            else:
                if not self.use_xyz:
                    raise ValueError("features=None and use_xyz=False -> empty groups")
                grouped = grouped_xyz

            grouped = grouped.permute(0, 3, 1, 2).contiguous()     # [B, C_in(+3), npoint, nsample]
            out = mlp(grouped).max(dim=-1).values                  # [B, C_out, npoint]
            outs.append(out)

        new_features = torch.cat(outs, dim=1)
        return new_xyz, new_features


# --------- Encoder (lazy-build like your original) ---------

class PointNet2MSGEncoderTorch(nn.Module):
    """
    Input: pts [B,N,3] or [B,N,3+Cin]
    Output tokens: [B, N_out, D_out]
    """
    def __init__(self, cfg: dict = MSG_CFG, use_xyz: bool = True, bn: bool = True,
                 voxel_size: Optional[float] = None, cdist_chunk_size: int = 16384):
        super().__init__()
        self.cfg, self.use_xyz, self.bn = cfg, use_xyz, bn
        self.voxel_size = voxel_size
        self.cdist_chunk_size = cdist_chunk_size
        self.SA = nn.ModuleList()
        self.out_channels: Optional[int] = None
        self._built = False
        self._cin_built: Optional[int] = None

    @staticmethod
    def _split_xyz_feat(pts: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        assert pts.size(-1) >= 3, "points last dim must be >= 3 (xyz)"
        if pts.size(-1) == 3:
            return pts, None, 0
        xyz, feat = pts[..., :3], pts[..., 3:]
        return xyz, feat.transpose(1, 2).contiguous(), feat.size(-1)

    def _build(self, cin: int, device: torch.device):
        sa_list = []
        in_channel = cin
        for npoint, radii, nsamp, mlp_specs in zip(
            self.cfg["NPOINTS"], self.cfg["RADIUS"], self.cfg["NSAMPLE"], self.cfg["MLPS"]
        ):
            mlps = []
            for spec in mlp_specs:
                c0 = in_channel + (3 if self.use_xyz else 0)
                mlps.append([c0] + spec)
            sa = PointnetSAModuleMSG_Torch(
                npoint=npoint, radii=radii, nsamples=nsamp, mlps=mlps,
                use_xyz=self.use_xyz, bn=self.bn,
                voxel_size=self.voxel_size,
                cdist_chunk_size=self.cdist_chunk_size
            ).to(device)
            sa_list.append(sa)
            in_channel = sum(m[-1] for m in mlps)

        self.SA = nn.ModuleList(sa_list)
        self.out_channels = in_channel
        self._cin_built = cin
        self._built = True
        self.to(device)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        if pts.dtype != torch.float32:
            pts = pts.float()
        xyz, feat, cin = self._split_xyz_feat(pts)
        if (not self._built) or (cin != self._cin_built):
            self._build(cin, xyz.device)

        cur_xyz, cur_feat = xyz, feat
        for sa in self.SA:
            cur_xyz, cur_feat = sa(cur_xyz, cur_feat)   # cur_feat: [B,C,npoint]
        return cur_feat.transpose(1, 2).contiguous()     # [B, npoint, C]


class PointNet2ClsMSGTorch(nn.Module):
    def __init__(self, cfg: dict = MSG_CFG, out_dim: int = 256,
                 use_xyz: bool = True, bn: bool = True,
                 voxel_size: Optional[float] = None, cdist_chunk_size: int = 16384):
        super().__init__()
        self.encoder = PointNet2MSGEncoderTorch(
            cfg=cfg, use_xyz=use_xyz, bn=bn,
            voxel_size=voxel_size, cdist_chunk_size=cdist_chunk_size
        )
        self.head: nn.Module = nn.Identity()
        self._head_built = False
        self.out_dim = out_dim

    def _build_head(self, in_dim: int, device: torch.device):
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, self.out_dim)
        ).to(device)
        self._head_built = True

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(pts)          # [B, npoint, C]
        g = tokens.max(dim=1).values        # [B, C]
        if not self._head_built:
            self._build_head(in_dim=g.size(-1), device=g.device)
        return self.head(g)


if __name__ == "__main__":
    enc = PointNet2MSGEncoderTorch(
        cfg=MSG_CFG, use_xyz=True, voxel_size=0.015, cdist_chunk_size=8192
    ).cuda()

    pts = torch.randn(2, 1024, 3, device="cuda", dtype=torch.float32)
    tokens = enc(pts)  # [B, npoint_last, C_out]
    print(tokens.shape)