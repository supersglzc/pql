import torch
import torch.nn as nn
import math
from pql.models.pointnet import PointNet2MSGEncoderTorch, MSG_CFG
from torch.distributions import Independent, Normal


class DiagGaussianTransformerPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, d_model=256, depth=4, nhead=8):
        super().__init__()
        self.d_model = d_model

        # Point encoder: project each point to D
        self.point_encoder = PointNet2MSGEncoderTorch(
            cfg=MSG_CFG, use_xyz=True, voxel_size=0.015, cdist_chunk_size=8192
        )
        
        # State encoder → 1 query token
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

        # Cross-Attention: state queries point cloud
        self.cross_attn = nn.Sequential(*[nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True) for _ in range(depth)])

        # # Optional: add N blocks of self-attn + FF
        # self.transformer_blocks = nn.Sequential(*[
        #     nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        #     for _ in range(depth)
        # ])

        # Output MLP
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, act_dim)
        )

        self.logstd = nn.Parameter(torch.zeros(act_dim))

    def forward(self, img, state, pc=None, sample=True, aug=False):
        return self.get_actions(img, state, pc=pc, sample=sample)[0]

    def get_actions(self, img, state, pc, sample=True):
        """
        state: [B, D_obs]
        pc:    [B, N, 3]
        """
        B, N, _ = pc.shape

        # 1. Encode point cloud tokens
        pc_tokens = self.point_encoder(pc)               # [B, N, D]
        
        # 2. Encode state query
        q = self.obs_encoder(state).unsqueeze(1)      # [B, 1, D]

        # 3. Cross-Attention: Q = state, KV = point tokens
        for attn in self.cross_attn:
            q, _ = attn(query=q, key=pc_tokens, value=pc_tokens)  # [B, 1, D]

        # # 4. Self-attention refinement
        # refined = self.transformer_blocks(attn_out)  # [B, 1, D]

        # 5. Output action distribution
        feat = q.squeeze(1)                    # [B, D]
        mean = self.policy_head(feat)
        std = torch.exp(self.logstd.expand_as(mean))

        dist = Independent(Normal(mean, std), 1)
        if sample:
            actions = dist.rsample()
        else:
            actions = mean
        return actions, dist

    def get_actions_logprob_entropy(self, img, state, pc, sample=True):
        actions, action_dist = self.get_actions(img, state, pc, sample=sample)
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return actions, action_dist, log_prob, entropy

    def logprob_entropy(self, img, state, actions, pc):
        _, action_dist = self.get_actions(img, state, pc)
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return actions, action_dist, log_prob, entropy
    

if __name__ == "__main__":
    model = DiagGaussianTransformerPolicy(obs_dim=44, act_dim=22)
    state = torch.randn(2, 44)
    pc = torch.randn(2, 1024, 3)
    actions, dist = model.get_actions(None, state, pc)
    print(actions.shape)
