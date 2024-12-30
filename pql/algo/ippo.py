from dataclasses import dataclass

import numpy as np
import torch
from copy import deepcopy

from pql.algo.ac_base import ActorCriticBase
from pql.utils.torch_util import RunningMeanStd
from pql.utils.common import handle_timeout, aggregate_traj_info
from pql.utils.common import load_class_from_path
from pql.utils.common import parse_multi_rew
from pql.models import model_name_to_path
from bidex.utils.symmetry import load_symmetric_system, SymmetryManager, slice_tensor

@dataclass
class AgentIPPO(ActorCriticBase):
    def __post_init__(self):
        super().__post_init__()
        self.obs_dim = list(self.cfg.task.multi.single_agent_obs_dim)
        self.action_dim = int(self.cfg.task.multi.single_agent_action_dim)
        act_class = load_class_from_path(self.cfg.algo.act_class,
                                         model_name_to_path[self.cfg.algo.act_class])
        cri_class = load_class_from_path(self.cfg.algo.cri_class,
                                         model_name_to_path[self.cfg.algo.cri_class])
        if "Equivariant" in self.cfg.algo.act_class:
            self.G = load_symmetric_system(cfg=self.cfg.task.symmetry)
            self.actor = act_class(self.G, self.cfg.task.symmetry.actor_input_fields[0], self.cfg.task.symmetry.actor_output_fields[0], self.obs_dim[0], self.action_dim).to(self.cfg.device)
            self.actor_left = act_class(self.G, self.cfg.task.symmetry.actor_input_fields[1], self.cfg.task.symmetry.actor_output_fields[1], self.obs_dim[1], self.action_dim).to(self.cfg.device)
        else:
            self.actor = act_class(self.obs_dim[0], self.action_dim).to(self.cfg.device)
            self.actor_left = act_class(self.obs_dim[1], self.action_dim).to(self.cfg.device)
        if "Equivariant" in self.cfg.algo.cri_class:
            self.G = load_symmetric_system(cfg=self.cfg.task.symmetry)
            self.critic = cri_class(self.G, self.cfg.task.symmetry.critic_input_fields[0], self.cfg.task.symmetry.critic_output_fields[0], self.obs_dim[0], 1).to(self.cfg.device)
            self.critic_left = cri_class(self.G, self.cfg.task.symmetry.critic_input_fields[1], self.cfg.task.symmetry.critic_output_fields[1], self.obs_dim[1], 1).to(self.cfg.device)
        else:
            self.critic = cri_class(self.obs_dim[0], self.action_dim).to(self.cfg.device)
            self.critic_left = cri_class(self.obs_dim[1], self.action_dim).to(self.cfg.device)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), self.cfg.algo.actor_lr)
        self.actor_optimizer_left = torch.optim.AdamW(self.actor_left.parameters(), self.cfg.algo.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), self.cfg.algo.critic_lr)
        self.critic_optimizer_left = torch.optim.AdamW(self.critic_left.parameters(), self.cfg.algo.critic_lr)

        self.timeout_info = None

        if self.cfg.algo.value_norm:
            self.value_rms = RunningMeanStd(shape=(1), device=self.device)
            self.value_rms_left = RunningMeanStd(shape=(1), device=self.device)

        self.symmetry_manager = SymmetryManager(self.cfg)

    def get_actions(self, obs, actor, critic, value_rms):
        if self.cfg.algo.obs_norm:
            obs = self.obs_rms.normalize(obs)
        actions, action_dist, logprobs, entropy = actor.get_actions_logprob_entropy(obs)
        value = critic(obs)
        if self.cfg.algo.value_norm:
            value_rms.update(value)
            value = value_rms.unnormalize(value)
        return actions, logprobs, value.flatten()

    @torch.no_grad()
    def explore_env(self, env, timesteps: int, random: bool = False) -> list:
        obs_dim = (self.obs_dim[0],) if isinstance(self.obs_dim[0], int) else self.obs_dim[0]
        obs_dim_left = (self.obs_dim[1],) if isinstance(self.obs_dim[1], int) else self.obs_dim[1]
        traj_obs = torch.zeros((timesteps, self.cfg.num_envs) + (*obs_dim,), device=self.device)
        traj_obs_left = torch.zeros((timesteps, self.cfg.num_envs) + (*obs_dim_left,), device=self.device)
        traj_actions = torch.zeros((timesteps, self.cfg.num_envs) + (self.action_dim,), device=self.device)
        traj_actions_left = torch.zeros((timesteps, self.cfg.num_envs) + (self.action_dim,), device=self.device)
        traj_logprobs = torch.zeros((timesteps, self.cfg.num_envs), device=self.device)
        traj_logprobs_left = torch.zeros((timesteps, self.cfg.num_envs), device=self.device)
        traj_rewards = torch.zeros((timesteps, self.cfg.num_envs), device=self.device)
        traj_rewards_left = torch.zeros((timesteps, self.cfg.num_envs), device=self.device)
        traj_dones = torch.zeros((timesteps, self.cfg.num_envs), device=self.device)
        traj_values = torch.zeros((timesteps, self.cfg.num_envs), device=self.device)
        traj_values_left = torch.zeros((timesteps, self.cfg.num_envs), device=self.device)
        infos = []

        ob = self.obs
        dones = self.dones
        for step in range(timesteps):
            if self.cfg.algo.obs_norm:
                self.obs_rms.update(ob)
            cur_symmetry_tracker = env.unwrapped.symmetry_tracker
            ob_right, ob_left = self.symmetry_manager.get_multi_agent_obs(ob, cur_symmetry_tracker)
            traj_obs[step] = deepcopy(ob_right)
            traj_obs_left[step] = deepcopy(ob_left)
            traj_dones[step] = dones

            action_right, logprob_right, val_right = self.get_actions(ob_right, self.actor, self.critic, self.value_rms)
            action_left, logprob_left, val_left = self.get_actions(ob_left, self.actor_left, self.critic_left, self.value_rms_left)
            action = self.symmetry_manager.get_execute_action(action_right, action_left, cur_symmetry_tracker)
            next_ob, reward, done, info = env.step(action)

            reward_right, reward_left = self.symmetry_manager.get_multi_agent_rew(info['detailed_reward'], cur_symmetry_tracker)
            self.update_tracker(reward_right, done, info)
                
            traj_actions[step] = action_right
            traj_actions_left[step] = action_left
            traj_logprobs[step] = logprob_right
            traj_logprobs_left[step] = logprob_left
            traj_rewards[step] = reward_right
            traj_rewards_left[step] = reward_left
            traj_values[step] = val_right
            traj_values_left[step] = val_left
            infos.append(deepcopy(info))
            ob = next_ob
            dones = done

        if self.cfg.algo.handle_timeout:
            if 'TimeLimit.truncated' in infos[0].keys():
                self.timeout_info = aggregate_traj_info(infos, 'TimeLimit.truncated')
            elif 'time_outs' in infos[0].keys():
                self.timeout_info = aggregate_traj_info(infos, 'time_outs')
                
        self.obs = ob
        self.dones = dones
        
        ob_right, ob_left = self.symmetry_manager.get_multi_agent_obs(ob, env.unwrapped.symmetry_tracker)
        data = self.compute_adv((traj_obs, traj_actions, traj_logprobs, traj_rewards,
                                 traj_dones, traj_values, ob_right, dones), gae=self.cfg.algo.use_gae, timeout=self.timeout_info, critic=self.critic, value_rms=self.value_rms)
        data_left = self.compute_adv((traj_obs_left, traj_actions_left, traj_logprobs_left, traj_rewards_left,
                                 traj_dones, traj_values_left, ob_left, dones), gae=self.cfg.algo.use_gae, timeout=self.timeout_info, critic=self.critic_left, value_rms=self.value_rms_left)

        return [data, data_left], timesteps * self.cfg.num_envs

    def compute_adv(self, buffer, gae=True, timeout=None, critic=None, value_rms=None):
        with torch.no_grad():
            obs, actions, logprobs, rewards, dones, values, next_obs, next_done = buffer
            obs_dim = (obs.shape[-1],)
            timesteps = obs.shape[0]
            if self.cfg.algo.obs_norm:
                next_obs = self.obs_rms.normalize(next_obs)
            next_value = critic(next_obs)
            if self.cfg.algo.value_norm:
                value_rms.update(next_value)
                next_value = value_rms.unnormalize(next_value)
            next_value = next_value.reshape(1, -1)
            if gae:
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(timesteps)):
                    if t == timesteps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]

                    if timeout is not None:
                        nextnonterminal2 = torch.logical_xor(nextnonterminal, timeout[t])
                    else:
                        nextnonterminal2 = nextnonterminal

                    delta = rewards[t] + self.cfg.algo.gamma * nextvalues * nextnonterminal2 - values[t]
                    lastgaelam = delta + self.cfg.algo.gamma * self.cfg.algo.lambda_gae_adv * nextnonterminal * lastgaelam
                    advantages[t] = deepcopy(lastgaelam)
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(self.device)
                for t in reversed(range(timesteps)):
                    if t == timesteps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + self.cfg.algo.gamma * nextnonterminal * next_return
                advantages = returns - values

        b_obs = obs.reshape((-1,) + (*obs_dim,))
        b_actions = actions.reshape((-1,) + (self.action_dim,))
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)

        # normalize rewards and values
        if self.cfg.algo.value_norm:
            value_rms.update(returns.reshape(-1))
            b_returns = value_rms.normalize(returns.reshape(-1))
            value_rms.update(values.reshape(-1))
            b_values = value_rms.normalize(values.reshape(-1))
        else:
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

        return (b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values)

    def update_net(self, data):
        b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values = data[0]
        b_obs_left, b_actions_left, b_logprobs_left, b_advantages_left, b_returns_left, b_values_left = data[1]
        buffer_size = b_obs.size()[0]
        assert buffer_size >= self.cfg.algo.batch_size

        b_inds = np.arange(buffer_size)
        critic_loss_list = list()
        actor_loss_list = list()
        critic_loss_list_left = list()
        actor_loss_list_left = list()
        for _ in range(self.cfg.algo.update_times):
            np.random.shuffle(b_inds)
            for start in range(0, buffer_size, self.cfg.algo.batch_size):
                end = start + self.cfg.algo.batch_size
                mb_inds = b_inds[start:end]

                if self.cfg.algo.obs_norm:
                    obs = self.obs_rms.normalize(b_obs[mb_inds])
                else:
                    obs = b_obs[mb_inds]
                    obs_left = b_obs_left[mb_inds]
                _, action_dist, newlogprob, entropy = self.actor.logprob_entropy(obs, b_actions[mb_inds])
                _, action_dist_left, newlogprob_left, entropy_left = self.actor_left.logprob_entropy(obs_left, b_actions_left[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                logratio_left = newlogprob_left - b_logprobs_left[mb_inds]
                ratio = logratio.exp()
                ratio_left = logratio_left.exp()
                mb_advantages = b_advantages[mb_inds]
                mb_advantages_left = b_advantages_left[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                mb_advantages_left = (mb_advantages_left - mb_advantages_left.mean()) / (mb_advantages_left.std() + 1e-8)

                actor_loss1 = -mb_advantages * ratio
                actor_loss1_left = -mb_advantages_left * ratio_left
                actor_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.algo.ratio_clip, 1 + self.cfg.algo.ratio_clip)
                actor_loss2_left = -mb_advantages_left * torch.clamp(ratio_left, 1 - self.cfg.algo.ratio_clip, 1 + self.cfg.algo.ratio_clip)
                actor_loss = torch.max(actor_loss1, actor_loss2).mean()
                actor_loss_left = torch.max(actor_loss1_left, actor_loss2_left).mean()

                newvalue = self.critic(obs)
                newvalue = newvalue.view(-1)
                newvalue_left = self.critic_left(obs_left)
                newvalue_left = newvalue_left.view(-1)
                if self.cfg.algo.value_clip:
                    critic_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    critic_loss_unclipped_left = (newvalue_left - b_returns_left[mb_inds]) ** 2
                    critic_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.cfg.algo.ratio_clip,
                        self.cfg.algo.ratio_clip,
                    )
                    critic_clipped_left = b_values_left[mb_inds] + torch.clamp(
                        newvalue_left - b_values_left[mb_inds],
                        -self.cfg.algo.ratio_clip,
                        self.cfg.algo.ratio_clip,
                    )
                    critic_loss_clipped = (critic_clipped - b_returns[mb_inds]) ** 2
                    critic_loss_clipped_left = (critic_clipped_left - b_returns_left[mb_inds]) ** 2
                    critic_loss = 0.5 * torch.max(critic_loss_unclipped, critic_loss_clipped).mean()
                    critic_loss_left = 0.5 * torch.max(critic_loss_unclipped_left, critic_loss_clipped_left).mean()
                else:
                    critic_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    critic_loss_left = 0.5 * ((newvalue_left - b_returns_left[mb_inds]) ** 2).mean()
                actor_loss = actor_loss - self.cfg.algo.lambda_entropy
                actor_loss_left = actor_loss_left - self.cfg.algo.lambda_entropy
                self.optimizer_update(self.actor_optimizer, actor_loss)
                self.optimizer_update(self.critic_optimizer, critic_loss)
                self.optimizer_update(self.actor_optimizer_left, actor_loss_left)
                self.optimizer_update(self.critic_optimizer_left, critic_loss_left)
                critic_loss_list.append(critic_loss.item())
                actor_loss_list.append(actor_loss.item())
                critic_loss_list_left.append(critic_loss_left.item())
                actor_loss_list_left.append(actor_loss_left.item())

        log_info = {
            "train/critic_loss": np.mean(critic_loss_list),
            "train/actor_loss": np.mean(actor_loss_list),
            "train/critic_loss_left": np.mean(critic_loss_list_left),
            "train/actor_loss_left": np.mean(actor_loss_list_left),
            "train/return": self.return_tracker.mean(),
            "train/episode_length": self.step_tracker.mean(),
            "train/success_rate": self.success_tracker.mean(),
        }
        self.add_info_tracker_log(log_info)
        return log_info
