from dataclasses import dataclass
from typing import Any

import torch
from omegaconf.dictconfig import DictConfig
from torch.nn.utils import clip_grad_norm_

from pql.utils.common import load_class_from_path
from pql.models import model_name_to_path
from pql.utils.common import Tracker
from pql.utils.torch_util import RunningMeanStd


@dataclass
class ActorCriticBase:
    env: Any
    cfg: DictConfig

    def __post_init__(self):
        self.obs = None
        self.obs_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.shape[0]
        act_class = load_class_from_path(self.cfg.algo.act_class,
                                         model_name_to_path[self.cfg.algo.act_class])
        cri_class = load_class_from_path(self.cfg.algo.cri_class,
                                         model_name_to_path[self.cfg.algo.cri_class])
        self.actor = act_class(self.obs_dim, self.action_dim).to(self.cfg.device)
        self.critic = cri_class(self.obs_dim, self.action_dim).to(self.cfg.device)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), self.cfg.algo.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), self.cfg.algo.critic_lr)

        self.return_tracker = Tracker(self.cfg.algo.tracker_len)
        self.step_tracker = Tracker(self.cfg.algo.tracker_len)
        self.current_returns = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.cfg.device)
        self.current_lengths = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.cfg.device)

        self.device = torch.device(self.cfg.device)

        if self.cfg.algo.obs_norm:
            self.obs_rms = RunningMeanStd(shape=self.obs_dim, device=self.device)
        else:
            self.obs_rms = None

    def reset_agent(self):
        self.obs = self.env.reset()

    def update_tracker(self, reward, done):
        self.current_returns += reward
        self.current_lengths += 1
        env_done_indices = torch.where(done)[0]
        self.return_tracker.update(self.current_returns[env_done_indices])
        self.step_tracker.update(self.current_lengths[env_done_indices])
        self.current_returns[env_done_indices] = 0
        self.current_lengths[env_done_indices] = 0

    def optimizer_update(self, optimizer, objective):
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if self.cfg.algo.max_grad_norm is not None:
            grad_norm = clip_grad_norm_(parameters=optimizer.param_groups[0]["params"],
                                        max_norm=self.cfg.algo.max_grad_norm)
        else:
            grad_norm = None
        optimizer.step()
        return grad_norm
