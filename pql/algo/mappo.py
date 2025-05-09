from dataclasses import dataclass

import numpy as np
import torch
from copy import deepcopy

from pql.algo.ac_base import ActorCriticBase
from pql.utils.torch_util import RunningMeanStd, slice_tensor
from pql.utils.common import handle_timeout, aggregate_traj_info
from pql.utils.common import load_class_from_path
from pql.utils.common import parse_multi_rew
from pql.models import model_name_to_path


@dataclass
class AgentMAPPO(ActorCriticBase):
    def __post_init__(self):
        super().__post_init__()
        self.timeout_info = None
        self.actor_left = self.actor
        cri_class = load_class_from_path(self.cfg.algo.cri_class,
                                         model_name_to_path[self.cfg.algo.cri_class])
        self.critic = cri_class(self.cfg.task.multi.shared_obs_dim, self.action_dim).to(self.cfg.device)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), self.cfg.algo.critic_lr)

        if self.cfg.algo.value_norm:
            self.value_rms = RunningMeanStd(shape=(1), device=self.device)

    def get_actions(self, obs, shared_obs):
        if self.cfg.algo.obs_norm:
            obs = self.obs_rms.normalize(obs)
        actions, action_dist, logprobs, entropy = self.actor.get_actions_logprob_entropy(obs)
        value = self.critic(shared_obs)
        if self.cfg.algo.value_norm:
            self.value_rms.update(value)
            value = self.value_rms.unnormalize(value)
        return actions, logprobs, value.flatten()

    @torch.no_grad()
    def explore_env(self, env, timesteps: int, random: bool = False) -> list:
        obs_dim = (self.obs_dim,) if isinstance(self.obs_dim, int) else self.obs_dim
        traj_obs = torch.zeros((timesteps, self.cfg.num_envs*2) + (*obs_dim,), device=self.device)
        traj_shared_obs = torch.zeros((timesteps, self.cfg.num_envs*2) + (self.cfg.task.multi.shared_obs_dim,), device=self.device)
        traj_actions = torch.zeros((timesteps, self.cfg.num_envs*2) + (self.action_dim,), device=self.device)
        traj_logprobs = torch.zeros((timesteps, self.cfg.num_envs*2), device=self.device)
        traj_rewards = torch.zeros((timesteps, self.cfg.num_envs*2), device=self.device)
        traj_dones = torch.zeros((timesteps, self.cfg.num_envs*2), device=self.device)
        traj_values = torch.zeros((timesteps, self.cfg.num_envs*2), device=self.device)
        infos = []

        ob = self.obs
        dones = self.dones
        for step in range(timesteps):
            if self.cfg.algo.obs_norm:
                self.obs_rms.update(ob)
            shared_obs = ob.repeat(2, 1)
            traj_shared_obs[step] = deepcopy(shared_obs)
            ob_right = slice_tensor(ob, self.cfg.task.multi.single_agent_obs_idx[0])
            ob_left = slice_tensor(ob, self.cfg.task.multi.single_agent_obs_idx[1])
            ob = torch.cat([ob_right, ob_left], dim=0)
            traj_obs[step] = deepcopy(ob)
            traj_dones[step] = dones.repeat(2)

            action, logprob, val = self.get_actions(ob, shared_obs)
            execute_action = torch.cat([action[:self.cfg.num_envs], action[self.cfg.num_envs:]], dim=-1)
            next_ob, reward, done, info = env.step(execute_action)
            self.update_tracker(reward, done, info)
                
            traj_actions[step] = action
            traj_logprobs[step] = logprob
            traj_rewards[step] = reward.repeat(2)
            traj_values[step] = val
            infos.append(deepcopy(info))
            ob = next_ob
            dones = done

        if self.cfg.algo.handle_timeout:
            if 'TimeLimit.truncated' in infos[0].keys():
                self.timeout_info = aggregate_traj_info(infos, 'TimeLimit.truncated').repeat(1, 2)
            elif 'time_outs' in infos[0].keys():
                self.timeout_info = aggregate_traj_info(infos, 'time_outs').repeat(1, 2)
            else:
                raise NotImplementedError
                
        self.obs = ob
        self.dones = dones
        
        data = self.compute_adv((traj_obs, traj_shared_obs, traj_actions, traj_logprobs, traj_rewards,
                                 traj_dones, traj_values, ob.repeat(2, 1), dones.repeat(2)), gae=self.cfg.algo.use_gae, timeout=self.timeout_info)

        return data, timesteps * self.cfg.num_envs

    def compute_adv(self, buffer, gae=True, timeout=None):
        with torch.no_grad():
            obs, shared_obs, actions, logprobs, rewards, dones, values, next_obs, next_done = buffer
            timesteps = obs.shape[0]
            if self.cfg.algo.obs_norm:
                next_obs = self.obs_rms.normalize(next_obs)
            next_value = self.critic(next_obs)
            if self.cfg.algo.value_norm:
                self.value_rms.update(next_value)
                next_value = self.value_rms.unnormalize(next_value)
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

        obs_dim = (self.obs_dim,) if isinstance(self.obs_dim, int) else self.obs_dim
        b_obs = obs.reshape((-1,) + (*obs_dim,))
        b_shared_obs = shared_obs.reshape((-1,) + (self.cfg.task.multi.shared_obs_dim,))
        b_actions = actions.reshape((-1,) + (self.action_dim,))
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)

        # normalize rewards and values
        if self.cfg.algo.value_norm:
            self.value_rms.update(returns.reshape(-1))
            b_returns = self.value_rms.normalize(returns.reshape(-1))
            self.value_rms.update(values.reshape(-1))
            b_values = self.value_rms.normalize(values.reshape(-1))
        else:
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

        return (b_obs, b_shared_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values)

    def update_net(self, data):
        b_obs, b_shared_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values = data
        buffer_size = b_obs.size()[0]
        assert buffer_size >= self.cfg.algo.batch_size

        b_inds = np.arange(buffer_size)
        critic_loss_list = list()
        actor_loss_list = list()
        for _ in range(self.cfg.algo.update_times):
            np.random.shuffle(b_inds)
            for start in range(0, buffer_size, self.cfg.algo.batch_size):
                end = start + self.cfg.algo.batch_size
                mb_inds = b_inds[start:end]

                if self.cfg.algo.obs_norm:
                    obs = self.obs_rms.normalize(b_obs[mb_inds])
                else:
                    obs = b_obs[mb_inds]
                    shared_obs = b_shared_obs[mb_inds]
                _, action_dist, newlogprob, entropy = self.actor.logprob_entropy(obs, b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                actor_loss1 = -mb_advantages * ratio
                actor_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.algo.ratio_clip, 1 + self.cfg.algo.ratio_clip)
                actor_loss = torch.max(actor_loss1, actor_loss2).mean()

                newvalue = self.critic(shared_obs)
                newvalue = newvalue.view(-1)
                if self.cfg.algo.value_clip:
                    critic_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    critic_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.cfg.algo.ratio_clip,
                        self.cfg.algo.ratio_clip,
                    )
                    critic_loss_clipped = (critic_clipped - b_returns[mb_inds]) ** 2
                    critic_loss = 0.5 * torch.max(critic_loss_unclipped, critic_loss_clipped).mean()
                else:
                    critic_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                actor_loss = actor_loss - self.cfg.algo.lambda_entropy
                self.optimizer_update(self.actor_optimizer, actor_loss)
                self.optimizer_update(self.critic_optimizer, critic_loss)
                critic_loss_list.append(critic_loss.item())
                actor_loss_list.append(actor_loss.item())

        log_info = {
            "train/critic_loss": np.mean(critic_loss_list),
            "train/actor_loss": np.mean(actor_loss_list),
            "train/return": self.return_tracker.mean(),
            "train/episode_length": self.step_tracker.mean(),
            "train/success_rate": self.success_tracker.mean(),
        }
        self.add_info_tracker_log(log_info)
        return log_info
