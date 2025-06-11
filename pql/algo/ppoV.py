from dataclasses import dataclass

import numpy as np
import torch
from copy import deepcopy

from pql.algo.ac_base import ActorCriticBase
from pql.utils.torch_util import RunningMeanStd
from pql.utils.common import handle_timeout, aggregate_traj_info
from pql.utils.common import load_class_from_path
from pql.models import model_name_to_path
from pql.models.visual import ResEncoder
from torch.nn.utils import clip_grad_norm_

@dataclass
class AgentPPOV(ActorCriticBase):
    def __post_init__(self):
        super().__post_init__()
        self.timeout_info = None
        self.encoder = ResEncoder(num_cams=self.cfg.task.cam.num_cams).to(self.cfg.device)
        self.encoder_optimizer = torch.optim.AdamW(self.encoder.parameters(), self.cfg.algo.actor_lr)
        act_class = load_class_from_path(self.cfg.algo.act_class,
                                         model_name_to_path[self.cfg.algo.act_class])
        self.actor = act_class(self.env.policy_space.shape[-1], self.action_dim, repr_dim=self.encoder.repr_dim, num_cams=self.cfg.task.cam.num_cams).to(self.cfg.device)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), self.cfg.algo.actor_lr)
        if self.cfg.algo.value_norm:
            self.value_rms = RunningMeanStd(shape=(1), device=self.device)

    def get_actions(self, obs):
        if self.cfg.algo.obs_norm:
            obs = self.obs_rms.normalize(obs)
        obs_img = self.encoder(obs['vision'])
        actions, action_dist, logprobs, entropy = self.actor.get_actions_logprob_entropy(obs_img, obs['policy'], pc=obs['point_cloud'])
        value = self.critic(obs['critic'])
        if self.cfg.algo.value_norm:
            self.value_rms.update(value)
            value = self.value_rms.unnormalize(value)
        return actions, logprobs, value.flatten()

    @torch.no_grad()
    def explore_env(self, env, timesteps: int, random: bool = False) -> list:
        img_shape = list(env.vision_space.shape[1:])
        policy_dim = env.policy_space.shape[-1]
        obs_dim = (self.obs_dim,) if isinstance(self.obs_dim, int) else self.obs_dim
        traj_obs = torch.zeros((timesteps, self.cfg.num_envs) + (*obs_dim,), device=self.device)
        traj_obs_img = torch.zeros((timesteps, self.cfg.num_envs) + (*img_shape,), device='cuda', dtype=torch.uint8)
        traj_obs_pc = torch.zeros((timesteps, self.cfg.num_envs, 1024, 3), device='cuda')
        traj_obs_policy = torch.zeros((timesteps, self.cfg.num_envs, policy_dim), device=self.device)
        traj_actions = torch.zeros((timesteps, self.cfg.num_envs) + (self.action_dim,), device=self.device)
        traj_logprobs = torch.zeros((timesteps, self.cfg.num_envs), device=self.device)
        traj_rewards = torch.zeros((timesteps, self.cfg.num_envs), device=self.device)
        traj_dones = torch.zeros((timesteps, self.cfg.num_envs), device=self.device)
        traj_values = torch.zeros((timesteps, self.cfg.num_envs), device=self.device)
        infos = []

        ob = self.obs
        dones = self.dones
        for step in range(timesteps):
            if self.cfg.algo.obs_norm:
                self.obs_rms.update(ob)
            traj_obs[step] = deepcopy(ob['critic'])
            traj_obs_img[step] = deepcopy(ob['vision']).to('cpu')
            traj_obs_pc[step] = deepcopy(ob['point_cloud']).to('cpu')
            traj_obs_policy[step] = deepcopy(ob['policy'])
            traj_dones[step] = dones

            action, logprob, val = self.get_actions(ob)
            next_ob, reward, done, info = env.step(action)
            self.update_tracker(reward, done, info)
                
            traj_actions[step] = action
            traj_logprobs[step] = logprob
            traj_rewards[step] = reward
            traj_values[step] = val
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
        
        data = self.compute_adv((traj_obs, traj_actions, traj_logprobs, traj_rewards,
                                 traj_dones, traj_values, ob['critic'], dones), gae=self.cfg.algo.use_gae, timeout=self.timeout_info, obs_img=traj_obs_img, obs_policy=traj_obs_policy, obs_pc=traj_obs_pc)

        return data, timesteps * self.cfg.num_envs

    def compute_adv(self, buffer, gae=True, timeout=None, obs_img=None, obs_policy=None, obs_pc=None):
        with torch.no_grad():
            obs, actions, logprobs, rewards, dones, values, next_obs, next_done = buffer
            img_shape = obs_img.shape[2:]
            pc_shape = obs_pc.shape[2:]
            policy_dim = obs_policy.shape[-1]
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
        b_obs_img = obs_img.reshape((-1,) + (*img_shape,))
        b_obs_pc = obs_pc.reshape((-1,) + (*pc_shape,))
        b_obs_policy = obs_policy.reshape((-1, policy_dim))
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

        return (b_obs, b_obs_img, b_obs_pc, b_obs_policy, b_actions, b_logprobs, b_advantages, b_returns, b_values)

    def update_net(self, data):
        b_obs, b_obs_img, b_obs_pc, b_obs_policy, b_actions, b_logprobs, b_advantages, b_returns, b_values = data
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

                obs = b_obs[mb_inds]
                obs_img = b_obs_img[mb_inds].float().to(self.device)
                obs_pc = b_obs_pc[mb_inds].to(self.device)

                obs_img = self.encoder(obs_img)
                _, action_dist, newlogprob, entropy = self.actor.logprob_entropy(obs_img, b_obs_policy[mb_inds], b_actions[mb_inds], pc=obs_pc, aug=False)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                actor_loss1 = -mb_advantages * ratio
                actor_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.algo.ratio_clip, 1 + self.cfg.algo.ratio_clip)
                actor_loss = torch.max(actor_loss1, actor_loss2).mean()

                newvalue = self.critic(obs)
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

                actor_loss = actor_loss - self.cfg.algo.lambda_entropy * entropy.mean()
                # self.optimizer_update(self.actor_optimizer, actor_loss)
                self.encoder_optimizer.zero_grad(set_to_none=True)
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                if self.cfg.algo.max_grad_norm is not None:
                    grad_norm = clip_grad_norm_(parameters=self.actor_optimizer.param_groups[0]["params"],
                                                max_norm=self.cfg.algo.max_grad_norm)
                    grad_norm_encoder = clip_grad_norm_(parameters=self.encoder_optimizer.param_groups[0]["params"],
                                                max_norm=self.cfg.algo.max_grad_norm)
                else:
                    grad_norm = None
                    grad_norm_encoder = None
                self.actor_optimizer.step()
                self.encoder_optimizer.step()
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
