from dataclasses import dataclass

import numpy as np
import torch
from copy import deepcopy

from pql.algo.ac_base import ActorCriticBase
from pql.utils.torch_util import RunningMeanStd
from pql.utils.common import handle_timeout, aggregate_traj_info
from pql.utils.common import load_class_from_path
from pql.utils.common import Tracker
from pql.models import model_name_to_path
from bidex.utils.symmetry import load_symmetric_system, SymmetryManager, slice_tensor
from pql.utils.schedule_util import LinearSchedule
@dataclass
class AgentEQSD2(ActorCriticBase):
    def __post_init__(self):
        super().__post_init__()
        self.algo_multi_cfg = self.cfg.task.multi.EQSD2
        self.obs_dim = list(self.cfg.task.multi.EQSD2.single_agent_obs_dim)
        self.action_dim = int(self.cfg.task.multi.EQSD2.single_agent_action_dim)
        act_class = load_class_from_path(self.cfg.algo.act_class,
                                         model_name_to_path[self.cfg.algo.act_class])
        cri_class = load_class_from_path(self.cfg.algo.cri_class,
                                         model_name_to_path[self.cfg.algo.cri_class])
        self.G = load_symmetric_system(cfg=self.cfg.task.symmetry)
        self.actor = act_class(self.G, self.cfg.task.symmetry.EQSD2.actor_input_fields[0], self.cfg.task.symmetry.EQSD2.actor_output_fields[0], self.obs_dim[0], self.action_dim).to(self.cfg.device)
        self.actor_left = act_class(self.G, self.cfg.task.symmetry.EQSD2.actor_input_fields[1], self.cfg.task.symmetry.EQSD2.actor_output_fields[1], self.obs_dim[1], self.action_dim).to(self.cfg.device)
        self.actor_team = act_class(self.G, self.cfg.task.symmetry.EQSD2.actor_input_fields[2], self.cfg.task.symmetry.EQSD2.actor_output_fields[2], self.obs_dim[2], self.action_dim*2).to(self.cfg.device)
        
        self.critic = cri_class(self.G, self.cfg.task.symmetry.EQSD2.critic_input_fields[0], self.cfg.task.symmetry.EQSD2.critic_output_fields[0], self.obs_dim[0], 1).to(self.cfg.device)
        self.critic_left = cri_class(self.G, self.cfg.task.symmetry.EQSD2.critic_input_fields[1], self.cfg.task.symmetry.EQSD2.critic_output_fields[1], self.obs_dim[1], 1).to(self.cfg.device)
        self.critic_team = cri_class(self.G, self.cfg.task.symmetry.EQSD2.critic_input_fields[2], self.cfg.task.symmetry.EQSD2.critic_output_fields[2], self.obs_dim[3], 1).to(self.cfg.device)
        
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), self.cfg.algo.actor_lr)
        self.actor_optimizer_left = torch.optim.AdamW(self.actor_left.parameters(), self.cfg.algo.actor_lr)
        self.actor_optimizer_team = torch.optim.AdamW(self.actor_team.parameters(), self.cfg.algo.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), self.cfg.algo.critic_lr)
        self.critic_optimizer_left = torch.optim.AdamW(self.critic_left.parameters(), self.cfg.algo.critic_lr)
        self.critic_optimizer_team = torch.optim.AdamW(self.critic_team.parameters(), self.cfg.algo.critic_lr)
        if self.cfg.task.multi.same_policy:
            self.actor_left = self.actor
            self.critic_left = self.critic
            self.actor_optimizer_left = self.actor_optimizer
            self.critic_optimizer_left = self.critic_optimizer

        self.timeout_info = None

        if self.cfg.algo.value_norm:
            self.value_rms = RunningMeanStd(shape=(1), device=self.device)
            self.value_rms_left = RunningMeanStd(shape=(1), device=self.device)
            self.value_rms_team = RunningMeanStd(shape=(1), device=self.device)
        self.symmetry_manager = SymmetryManager(self.cfg.task.multi.EQSD2, self.cfg.task.symmetry.symmetric_envs)

        self.kl_scheduler = LinearSchedule(start_val=self.cfg.algo.kl_max,
                                                  end_val=self.cfg.algo.kl_min,
                                                  total_iters=self.cfg.algo.kl_decay_timesteps // self.cfg.num_envs
                                                  )

    def reset_agent(self):
        self.obs, extras = self.env.reset()
        self.dones = torch.zeros(self.cfg.num_envs).to(self.device)
        self.current_returns = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.cfg.device)
        self.current_lengths = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.cfg.device)
        self.detailed_returns = None
        self.detailed_tracker = None
        self.detailed_returns_team = None
        self.detailed_tracker_team = None
        return self.obs, extras
    
    def update_tracker(self, reward, done, info):
        self.current_returns[:self.cfg.num_envs//2] += reward
        self.current_lengths[:self.cfg.num_envs//2] += 1
        env_done_indices = torch.where(done)[0]
        first_half_indices = torch.arange(self.cfg.num_envs//2).to(self.device)
        global_env_done_indices = first_half_indices[env_done_indices]
        if len(env_done_indices) != 0:
            self.return_tracker.update(self.current_returns[global_env_done_indices])
            self.step_tracker.update(self.current_lengths[global_env_done_indices])
            self.success_tracker.update(info['success'][global_env_done_indices])
            self.current_returns[global_env_done_indices] = 0
            self.current_lengths[global_env_done_indices] = 0
        # reward logger
        if self.detailed_returns is None:
            self.detailed_returns = {}
            self.detailed_tracker = {}
            for rew_name in info['detailed_reward'].keys():
                self.detailed_returns[rew_name] = torch.zeros(self.cfg.num_envs//2, dtype=torch.float32, device=self.cfg.device)
                self.detailed_tracker[rew_name] = Tracker(self.cfg.algo.tracker_len)
        for rew_name in info['detailed_reward'].keys():
            self.detailed_returns[rew_name] += info['detailed_reward'][rew_name][:self.cfg.num_envs//2]
            if len(env_done_indices) != 0:
                self.detailed_tracker[rew_name].update(self.detailed_returns[rew_name][env_done_indices])
                self.detailed_returns[rew_name][env_done_indices] = 0
    
    def update_tracker_team(self, reward, done, info):
        self.current_returns[self.cfg.num_envs//2:] += reward
        self.current_lengths[self.cfg.num_envs//2:] += 1
        env_done_indices = torch.where(done)[0]
        second_half_indices = torch.arange(self.cfg.num_envs//2, self.cfg.num_envs).to(self.device)
        global_env_done_indices = second_half_indices[env_done_indices]
        if len(env_done_indices) != 0:
            self.return_tracker.update(self.current_returns[global_env_done_indices])
            self.step_tracker.update(self.current_lengths[global_env_done_indices])
            self.success_tracker.update(info['success'][global_env_done_indices])
            self.current_returns[global_env_done_indices] = 0
            self.current_lengths[global_env_done_indices] = 0
        # reward logger
        if self.detailed_returns_team is None:
            self.detailed_returns_team = {}
            self.detailed_tracker_team = {}
            for rew_name in info['detailed_reward'].keys():
                self.detailed_returns_team[rew_name] = torch.zeros(self.cfg.num_envs//2, dtype=torch.float32, device=self.cfg.device)
                self.detailed_tracker_team[rew_name] = Tracker(self.cfg.algo.tracker_len)
        for rew_name in info['detailed_reward'].keys():
            self.detailed_returns_team[rew_name] += info['detailed_reward'][rew_name][self.cfg.num_envs//2:]
            if len(env_done_indices) != 0:
                self.detailed_tracker_team[rew_name].update(self.detailed_returns_team[rew_name][env_done_indices])
                self.detailed_returns_team[rew_name][env_done_indices] = 0
    
    def get_actions(self, obs, actor, critic=None, value_rms=None):
        if self.cfg.algo.obs_norm:
            obs = self.obs_rms.normalize(obs)
        actions, action_dist, logprobs, entropy = actor.get_actions_logprob_entropy(obs)
        if critic is not None:
            value = critic(obs)
            if self.cfg.algo.value_norm:
                value_rms.update(value)
                value = value_rms.unnormalize(value)
            return actions, logprobs, value.flatten()
        else:
            return actions, logprobs, None
    
    def get_values(self, obs, critic, value_rms):
        value = critic(obs)
        if self.cfg.algo.value_norm:
            value_rms.update(value)
            value = value_rms.unnormalize(value)
        return value.flatten()

    @torch.no_grad()
    def explore_env(self, env, timesteps: int, random: bool = False) -> list:
        obs_dim = (self.obs_dim[0],) if isinstance(self.obs_dim[0], int) else self.obs_dim[0]
        obs_dim_left = (self.obs_dim[1],) if isinstance(self.obs_dim[1], int) else self.obs_dim[1]
        obs_dim_team = (self.obs_dim[2],) if isinstance(self.obs_dim[2], int) else self.obs_dim[2]
        obs_dim_team_critic= (self.obs_dim[3],) if isinstance(self.obs_dim[3], int) else self.obs_dim[3]
        traj_obs = torch.zeros((timesteps, self.cfg.num_envs//2) + (*obs_dim,), device=self.device)
        traj_obs_left = torch.zeros((timesteps, self.cfg.num_envs//2) + (*obs_dim_left,), device=self.device)
        traj_obs_ind_side = torch.zeros((timesteps, self.cfg.num_envs//2) + (*obs_dim_team,), device=self.device)
        traj_obs_team = torch.zeros((timesteps, self.cfg.num_envs//2) + (*obs_dim_team,), device=self.device)
        traj_obs_ind_side_critic = torch.zeros((timesteps, self.cfg.num_envs//2) + (*obs_dim_team_critic,), device=self.device)
        traj_obs_team_critic = torch.zeros((timesteps, self.cfg.num_envs//2) + (*obs_dim_team_critic,), device=self.device)

        traj_actions = torch.zeros((timesteps, self.cfg.num_envs//2) + (self.action_dim,), device=self.device)
        traj_actions_left = torch.zeros((timesteps, self.cfg.num_envs//2) + (self.action_dim,), device=self.device)
        traj_actions_ind_side = torch.zeros((timesteps, self.cfg.num_envs//2) + (self.action_dim*2,), device=self.device)
        traj_actions_team = torch.zeros((timesteps, self.cfg.num_envs//2) + (self.action_dim*2,), device=self.device)

        traj_logprobs = torch.zeros((timesteps, self.cfg.num_envs//2), device=self.device)
        traj_logprobs_left = torch.zeros((timesteps, self.cfg.num_envs//2), device=self.device)
        traj_logprobs_ind_side = torch.zeros((timesteps, self.cfg.num_envs//2), device=self.device)
        traj_logprobs_team = torch.zeros((timesteps, self.cfg.num_envs//2), device=self.device)

        traj_rewards = torch.zeros((timesteps, self.cfg.num_envs//2), device=self.device)
        traj_rewards_left = torch.zeros((timesteps, self.cfg.num_envs//2), device=self.device)
        traj_rewards_ind_side = torch.zeros((timesteps, self.cfg.num_envs//2), device=self.device)
        traj_rewards_team = torch.zeros((timesteps, self.cfg.num_envs//2), device=self.device)

        traj_dones = torch.zeros((timesteps, self.cfg.num_envs//2), device=self.device)
        traj_dones_team = torch.zeros((timesteps, self.cfg.num_envs//2), device=self.device)

        traj_values = torch.zeros((timesteps, self.cfg.num_envs//2), device=self.device)
        traj_values_left = torch.zeros((timesteps, self.cfg.num_envs//2), device=self.device)
        traj_values_ind_side = torch.zeros((timesteps, self.cfg.num_envs//2), device=self.device)
        traj_values_team = torch.zeros((timesteps, self.cfg.num_envs//2), device=self.device)
        infos = []

        ob = self.obs
        dones = self.dones
        for step in range(timesteps):
            self.kl_scheduler.step()
            if self.cfg.algo.obs_norm:
                self.obs_rms.update(ob)
            cur_symmetry_tracker = env.unwrapped.symmetry_tracker
            ob_right, ob_left, ob_team, ob_team_critic = self.symmetry_manager.get_multi_agent_obs(ob, cur_symmetry_tracker)
            traj_obs[step] = deepcopy(ob_right[:self.cfg.num_envs//2])
            traj_obs_left[step] = deepcopy(ob_left[:self.cfg.num_envs//2])
            traj_dones[step] = dones[:self.cfg.num_envs//2]
            traj_obs_team[step] = deepcopy(ob_team[self.cfg.num_envs//2:])
            traj_obs_team_critic[step] = deepcopy(ob_team_critic[self.cfg.num_envs//2:])
            traj_obs_ind_side[step] = deepcopy(ob_team[:self.cfg.num_envs//2])
            traj_obs_ind_side_critic[step] = deepcopy(ob_team_critic[:self.cfg.num_envs//2])
            traj_dones_team[step] = dones[self.cfg.num_envs//2:]

            with torch.no_grad():
                action_right, logprob_right, val_right = self.get_actions(ob_right[:self.cfg.num_envs//2], self.actor, self.critic, self.value_rms)
                action_left, logprob_left, val_left = self.get_actions(ob_left[:self.cfg.num_envs//2], self.actor_left, self.critic_left, self.value_rms_left)
                action_team, logprob_team, _ = self.get_actions(ob_team[self.cfg.num_envs//2:], self.actor_team)
                val_team = self.get_values(ob_team_critic[self.cfg.num_envs//2:], self.critic_team, self.value_rms_team)
                action_ind = self.symmetry_manager.get_execute_action(action_right, action_left, cur_symmetry_tracker[:self.cfg.num_envs//2] if cur_symmetry_tracker is not None else None)
                
                # get logprob for another side
                logprob_ind_side = self.actor_team.logprob(ob_team[:self.cfg.num_envs//2], action_ind)
                val_ind_side = self.get_values(ob_team_critic[:self.cfg.num_envs//2], self.critic_team, self.value_rms_team)

            action = torch.cat([action_ind, action_team], dim=0)
            next_ob, reward, done, info = env.step(action)

            reward_right, reward_left, reward_team = self.symmetry_manager.get_multi_agent_rew(info['detailed_reward'], cur_symmetry_tracker)
            self.update_tracker(reward_right[:self.cfg.num_envs//2] + reward_left[:self.cfg.num_envs//2], done[:self.cfg.num_envs//2], info)
            self.update_tracker_team(reward_team[self.cfg.num_envs//2:], done[self.cfg.num_envs//2:], info)
                
            traj_actions[step] = action_right
            traj_actions_left[step] = action_left
            traj_actions_ind_side[step] = action_ind
            traj_actions_team[step] = action_team

            traj_logprobs[step] = logprob_right
            traj_logprobs_left[step] = logprob_left
            traj_logprobs_ind_side[step] = logprob_ind_side
            traj_logprobs_team[step] = logprob_team

            traj_rewards[step] = reward_right[:self.cfg.num_envs//2]
            traj_rewards_left[step] = reward_left[:self.cfg.num_envs//2]
            traj_rewards_ind_side[step] = reward_team[:self.cfg.num_envs//2]
            traj_rewards_team[step] = reward_team[self.cfg.num_envs//2:]

            traj_values[step] = val_right
            traj_values_left[step] = val_left
            traj_values_ind_side[step] = val_ind_side
            traj_values_team[step] = val_team
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
        
        ob_right, ob_left, ob_team, ob_team_critic = self.symmetry_manager.get_multi_agent_obs(ob, env.unwrapped.symmetry_tracker)
        data = self.compute_adv((traj_obs, traj_actions, traj_logprobs, traj_rewards,
                                 traj_dones, traj_values, ob_right[:self.cfg.num_envs//2], dones[:self.cfg.num_envs//2]), gae=self.cfg.algo.use_gae, timeout=self.timeout_info[:, :self.cfg.num_envs//2], critic=self.critic, value_rms=self.value_rms)
        data_left = self.compute_adv((traj_obs_left, traj_actions_left, traj_logprobs_left, traj_rewards_left,
                                 traj_dones, traj_values_left, ob_left[:self.cfg.num_envs//2], dones[:self.cfg.num_envs//2]), gae=self.cfg.algo.use_gae, timeout=self.timeout_info[:, :self.cfg.num_envs//2], critic=self.critic_left, value_rms=self.value_rms_left)
        data_team = self.compute_adv((traj_obs_team, traj_actions_team, traj_logprobs_team, traj_rewards_team,
                                 traj_dones_team, traj_values_team, ob_team_critic[self.cfg.num_envs//2:], dones[self.cfg.num_envs//2:]), gae=self.cfg.algo.use_gae, timeout=self.timeout_info[:, self.cfg.num_envs//2:], critic=self.critic_team, value_rms=self.value_rms_team)
        data_obs_team_critic = traj_obs_team_critic.reshape(-1, *obs_dim_team_critic)
        data_ind_side = self.compute_adv((traj_obs_ind_side, traj_actions_ind_side, traj_logprobs_ind_side, traj_rewards_ind_side,
                                 traj_dones_team, traj_values_ind_side, ob_team_critic[:self.cfg.num_envs//2], dones[:self.cfg.num_envs//2]), gae=self.cfg.algo.use_gae, timeout=self.timeout_info[:, :self.cfg.num_envs//2], critic=self.critic_team, value_rms=self.value_rms_team)
        data_obs_ind_side_critic = traj_obs_ind_side_critic.reshape(-1, *obs_dim_team_critic)
        return [data, data_left, data_team, data_ind_side, data_obs_team_critic, data_obs_ind_side_critic], timesteps * self.cfg.num_envs

    def compute_adv(self, buffer, gae=True, timeout=None, critic=None, value_rms=None):
        with torch.no_grad():
            obs, actions, logprobs, rewards, dones, values, next_obs, next_done = buffer
            obs_dim = (obs.shape[-1],)
            action_dim = actions.shape[-1] if actions is not None else None
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
        b_actions = actions.reshape((-1,) + (action_dim,)) if actions is not None else None
        b_logprobs = logprobs.reshape(-1) if logprobs is not None else None
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
    
    def compute_actor_loss(self, policy, obs, old_actions, old_logprobs, advantages, 
                           obs2=None, old_actions2=None, old_logprobs2=None, new_logprobs2=None, advantages2=None):
        _, _, newlogprob, entropy = policy.logprob_entropy(obs, old_actions)
        logratio = newlogprob - old_logprobs
        ratio = logratio.exp()
        mb_advantages = advantages
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
        actor_loss1 = -mb_advantages * ratio
        actor_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.algo.ratio_clip, 1 + self.cfg.algo.ratio_clip)
        actor_loss = torch.max(actor_loss1, actor_loss2).mean()

        if obs2 is not None:
            _, _, newlogprob_off, _ = policy.logprob_entropy(obs2, old_actions2)
            logratio_off = newlogprob_off - new_logprobs2
            miu = old_logprobs2 - new_logprobs2
            ratio_off = logratio_off.exp()
            mb_advantages_team_side = advantages2
            mb_advantages_team_side = (mb_advantages_team_side - mb_advantages_team_side.mean()) / (mb_advantages_team_side.std() + 1e-8)
            actor_loss1_team_side = -mb_advantages_team_side * ratio_off
            actor_loss2_team_side = -mb_advantages_team_side * torch.clamp(ratio_off, miu * (1 - self.cfg.algo.ratio_clip), miu * (1 + self.cfg.algo.ratio_clip))
            actor_loss += torch.max(actor_loss1_team_side, actor_loss2_team_side).mean()

        return actor_loss, entropy
    
    def compute_actor_loss2(self, policy, obs, old_actions, old_logprobs, advantages, 
                           obs2=None, old_actions2=None, old_logprobs2=None, new_logprobs2=None, advantages2=None):
        _, _, newlogprob, entropy = policy.logprob_entropy(obs, old_actions)
        logratio = newlogprob - old_logprobs
        ratio = logratio.exp()
        miu = torch.ones_like(ratio)
        mb_advantages = advantages
        if obs2 is not None:
            _, _, newlogprob_off, _ = policy.logprob_entropy(obs2, old_actions2)
            logratio_off = newlogprob_off - new_logprobs2
            ratio_off = logratio_off.exp()
            print(ratio_off)
            miu_off = old_logprobs2 - new_logprobs2
            mb_advantages = torch.cat([advantages, advantages2], dim=0)
            miu = torch.cat([miu, miu_off], dim=0)
            ratio = torch.cat([ratio, ratio_off], dim=0)
        miu = torch.ones_like(ratio)
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
        actor_loss1 = -mb_advantages * ratio
        actor_loss2 = -mb_advantages * torch.clamp(ratio, miu * (1 - self.cfg.algo.ratio_clip), miu * (1 + self.cfg.algo.ratio_clip))
        actor_loss = torch.max(actor_loss1, actor_loss2).mean()

        return actor_loss, entropy
    
    def compute_critic_loss(self, critic, obs, returns, values):
        newvalue = critic(obs)
        newvalue = newvalue.view(-1)
        if self.cfg.algo.value_clip:
            critic_loss_unclipped = (newvalue - returns) ** 2
            critic_clipped = values + torch.clamp(
                newvalue - values,
                -self.cfg.algo.ratio_clip,
                self.cfg.algo.ratio_clip,
            )
            critic_loss_clipped = (critic_clipped - returns) ** 2
            critic_loss = 0.5 * torch.max(critic_loss_unclipped, critic_loss_clipped).mean()
        else:
            critic_loss = 0.5 * ((newvalue - returns) ** 2).mean()
        return critic_loss
    
    def compute_kl_divergence(self, policy1, policy2, policy3, s1, s2, s3, a1, a2, a3):
        # Get the distributions from the independent policies.
        logp_indep = (policy1.logprob(s1, a1) + policy2.logprob(s2, a2)).detach()
        logp_joint = policy3.logprob(s3, a3)
        return (logp_indep - logp_joint).mean()

    def update_net(self, data):
        b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values = data[0]
        b_obs_left, b_actions_left, b_logprobs_left, b_advantages_left, b_returns_left, b_values_left = data[1]
        b_obs_team, b_actions_team, b_logprobs_team, b_advantages_team, b_returns_team, b_values_team = data[2]

        b_obs_ind_side, b_actions_ind_side, b_logprobs_ind_side, b_advantages_ind_side, _, _ = data[3]
        b_obs_team_critic = data[4]
        b_obs_ind_side_critic = data[5]

        buffer_size = b_obs.size()[0]
        assert buffer_size >= self.cfg.algo.batch_size

        b_inds = np.arange(buffer_size)
        critic_loss_list = list()
        actor_loss_list = list()
        critic_loss_list_left = list()
        actor_loss_list_left = list()
        critic_loss_list_team = list()
        actor_loss_list_team = list()
        for _ in range(self.cfg.algo.update_times):
            np.random.shuffle(b_inds)
            for start in range(0, buffer_size, self.cfg.algo.batch_size):
                end = start + self.cfg.algo.batch_size
                mb_inds = b_inds[start:end]

                if self.cfg.algo.obs_norm:
                    obs = self.obs_rms.normalize(b_obs[mb_inds])
                    obs_left = self.obs_rms.normalize(b_obs_left[mb_inds])
                    obs_team = self.obs_rms.normalize(b_obs_team[mb_inds])
                else:
                    obs = b_obs[mb_inds]
                    obs_left = b_obs_left[mb_inds]
                    obs_team = b_obs_team[mb_inds]
                actor_loss, entropy = self.compute_actor_loss(self.actor, obs, b_actions[mb_inds], b_logprobs[mb_inds], b_advantages[mb_inds])
                actor_loss_left, entropy_left = self.compute_actor_loss(self.actor_left, obs_left, b_actions_left[mb_inds], b_logprobs_left[mb_inds], b_advantages_left[mb_inds])
                actor_loss_team, entropy_team = self.compute_actor_loss(self.actor_team, obs_team, b_actions_team[mb_inds], b_logprobs_team[mb_inds], b_advantages_team[mb_inds])
                # new_logprobs_ind_side = (self.actor.logprob(obs, b_actions[mb_inds]) + self.actor_left.logprob(obs_left, b_actions_left[mb_inds])).detach()
                # actor_loss_team, entropy_team = self.compute_actor_loss2(self.actor_team, obs_team, b_actions_team[mb_inds], b_logprobs_team[mb_inds], b_advantages_team[mb_inds],
                #                                                         b_obs_ind_side[mb_inds], b_actions_ind_side[mb_inds], b_logprobs_ind_side[mb_inds], new_logprobs_ind_side, b_advantages_ind_side[mb_inds])
                team_kl_loss = self.compute_kl_divergence(self.actor, self.actor_left, self.actor_team, 
                                                          obs, obs_left, b_obs_ind_side[mb_inds],
                                                          b_actions[mb_inds], b_actions_left[mb_inds], b_actions_ind_side[mb_inds])
                actor_loss_team += self.kl_scheduler.val() * team_kl_loss
                critic_loss = self.compute_critic_loss(self.critic, obs, b_returns[mb_inds], b_values[mb_inds])
                critic_loss_left = self.compute_critic_loss(self.critic_left, obs_left, b_returns_left[mb_inds], b_values_left[mb_inds])
                critic_loss_team = self.compute_critic_loss(self.critic_team, b_obs_team_critic[mb_inds], b_returns_team[mb_inds], b_values_team[mb_inds])

                actor_loss = actor_loss - self.cfg.algo.lambda_entropy * entropy.mean()
                actor_loss_left = actor_loss_left - self.cfg.algo.lambda_entropy * entropy_left.mean()
                actor_loss_team = actor_loss_team - self.cfg.algo.lambda_entropy * entropy_team.mean()
                self.optimizer_update(self.actor_optimizer, actor_loss)
                self.optimizer_update(self.critic_optimizer, critic_loss)
                self.optimizer_update(self.actor_optimizer_left, actor_loss_left)
                self.optimizer_update(self.critic_optimizer_left, critic_loss_left)
                self.optimizer_update(self.actor_optimizer_team, actor_loss_team)
                self.optimizer_update(self.critic_optimizer_team, critic_loss_team)
                critic_loss_list.append(critic_loss.item())
                actor_loss_list.append(actor_loss.item())
                critic_loss_list_left.append(critic_loss_left.item())
                actor_loss_list_left.append(actor_loss_left.item())
                critic_loss_list_team.append(critic_loss_team.item())
                actor_loss_list_team.append(actor_loss_team.item())

        log_info = {
            "train/critic_loss": np.mean(critic_loss_list),
            "train/actor_loss": np.mean(actor_loss_list),
            "train/critic_loss_left": np.mean(critic_loss_list_left),
            "train/actor_loss_left": np.mean(actor_loss_list_left),
            "train/critic_loss_team": np.mean(critic_loss_list_team),
            "train/actor_loss_team": np.mean(actor_loss_list_team),
            "train/return": self.return_tracker.mean(),
            "train/episode_length": self.step_tracker.mean(),
            "train/success_rate": self.success_tracker.mean(),
        }
        self.add_info_tracker_log(log_info)
        return log_info
