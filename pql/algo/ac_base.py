from dataclasses import dataclass
from typing import Any

import torch
from omegaconf.dictconfig import DictConfig
from torch.nn.utils import clip_grad_norm_

from pql.utils.common import load_class_from_path
from pql.models import model_name_to_path
from pql.utils.common import Tracker
from pql.utils.torch_util import RunningMeanStd
from bidex.utils.symmetry import load_symmetric_system


@dataclass
class ActorCriticBase:
    env: Any
    cfg: DictConfig
    obs_dim: int = None
    action_dim: int = None

    def __post_init__(self):
        self.obs = None
        if self.obs_dim is None:
            self.obs_dim = self.env.observation_space.shape
        if self.action_dim is None:
            self.action_dim = self.env.action_space.shape[0]
        act_class = load_class_from_path(self.cfg.algo.act_class,
                                         model_name_to_path[self.cfg.algo.act_class])
        cri_class = load_class_from_path(self.cfg.algo.cri_class,
                                         model_name_to_path[self.cfg.algo.cri_class])
        if not self.cfg.algo.multi_agent:
            if "Equivariant" in self.cfg.algo.act_class:
                self.G = load_symmetric_system(cfg=self.cfg.task.symmetry)
                self.actor = act_class(self.G, self.cfg.task.symmetry.actor_input_fields, self.cfg.task.symmetry.actor_output_fields, self.obs_dim, self.action_dim).to(self.cfg.device)
            else:
                self.actor = act_class(self.obs_dim, self.action_dim).to(self.cfg.device)
            if "Equivariant" in self.cfg.algo.cri_class and not self.cfg.algo.multi_agent:
                self.G = load_symmetric_system(cfg=self.cfg.task.symmetry)
                self.critic = cri_class(self.G, self.cfg.task.symmetry.critic_input_fields, self.cfg.task.symmetry.critic_output_fields, self.obs_dim, self.action_dim).to(self.cfg.device)
            else:
                self.critic = cri_class(self.obs_dim, self.action_dim).to(self.cfg.device)
            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), self.cfg.algo.actor_lr)
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), self.cfg.algo.critic_lr)

        self.current_returns = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.cfg.device)
        self.current_lengths = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.cfg.device)
        self.detailed_returns = None
        self.return_tracker = Tracker(self.cfg.algo.tracker_len)
        self.success_tracker = Tracker(self.cfg.algo.tracker_len)
        self.detailed_tracker = None
        self.step_tracker = Tracker(self.cfg.algo.tracker_len)

        info_track_keys = self.cfg.info_track_keys
        if info_track_keys is not None:
            info_track_keys = [info_track_keys] if isinstance(info_track_keys, str) else info_track_keys
            self.info_trackers = {key: Tracker(self.cfg.algo.tracker_len) for key in info_track_keys}
            self.info_track_step = {key: self.cfg.info_track_step[idx] for idx, key in enumerate(info_track_keys)}
            self.traj_info_values = {key: torch.zeros(self.cfg.num_envs, dtype=torch.float32, device='cpu') for key in info_track_keys}

        self.device = torch.device(self.cfg.device)

        if self.cfg.algo.obs_norm:
            self.obs_rms = RunningMeanStd(shape=self.obs_dim, device=self.device)
        else:
            self.obs_rms = None

    def reset_agent(self):
        self.obs, extras = self.env.reset()
        self.dones = torch.zeros(self.cfg.num_envs).to(self.device)
        self.current_returns = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.cfg.device)
        self.current_lengths = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.cfg.device)
        if self.detailed_returns is not None:
            for rew_name in self.detailed_returns.keys():
                self.detailed_returns[rew_name] = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.cfg.device)
        return self.obs, extras

    def update_tracker(self, reward, done, info):
        self.current_returns += reward
        self.current_lengths += 1
        env_done_indices = torch.where(done)[0]
        if len(env_done_indices) != 0:
            self.return_tracker.update(self.current_returns[env_done_indices])
            self.step_tracker.update(self.current_lengths[env_done_indices])
            self.success_tracker.update(info['success'][env_done_indices])
            self.current_returns[env_done_indices] = 0
            self.current_lengths[env_done_indices] = 0
        if self.cfg.info_track_keys is not None:
            env_done_indices = env_done_indices.cpu()
            for key in self.cfg.info_track_keys:
                if key not in info:
                    continue
                if self.info_track_step[key] == 'last':
                    info_val = info[key]
                    self.info_trackers[key].update(info_val[env_done_indices].cpu())
                elif self.info_track_step[key] == 'all-episode':
                    self.traj_info_values[key] += info[key].cpu()
                    self.info_trackers[key].update(self.traj_info_values[key][env_done_indices])
                    self.traj_info_values[key][env_done_indices] = 0
                elif self.info_track_step[key] == 'all-step':
                    self.info_trackers[key].update(info[key].cpu())

        # reward logger
        if self.detailed_returns is None:
            self.detailed_returns = {}
            self.detailed_tracker = {}
            for rew_name in info['detailed_reward'].keys():
                self.detailed_returns[rew_name] = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.cfg.device)
                self.detailed_tracker[rew_name] = Tracker(self.cfg.algo.tracker_len)
        for rew_name in info['detailed_reward'].keys():
            self.detailed_returns[rew_name] += info['detailed_reward'][rew_name]
            if len(env_done_indices) != 0:
                self.detailed_tracker[rew_name].update(self.detailed_returns[rew_name][env_done_indices])
                self.detailed_returns[rew_name][env_done_indices] = 0

    def add_info_tracker_log(self, log_info):
        if self.cfg.info_track_keys is not None:
            for key in self.cfg.info_track_keys:
                log_info[key] = self.info_trackers[key].mean()

    def optimizer_update(self, optimizer, objective, retain_graph=False):
        optimizer.zero_grad(set_to_none=True)
        objective.backward(retain_graph=retain_graph)
        if self.cfg.algo.max_grad_norm is not None:
            grad_norm = clip_grad_norm_(parameters=optimizer.param_groups[0]["params"],
                                        max_norm=self.cfg.algo.max_grad_norm)
        else:
            grad_norm = None
        optimizer.step()
        return grad_norm
