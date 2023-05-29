from dataclasses import dataclass
from typing import Any

from gym import spaces


@dataclass
class FlatObEnvWrapper:
    env: Any

    def __post_init__(self):
        self.observation_space = self.env.observation_space
        if isinstance(self.observation_space, spaces.Dict):
            self.observation_space = self.observation_space['obs']
        self.action_space = self.env.action_space
        self.max_episode_length = self.env.max_episode_length

    def reset(self):
        ob = self.env.reset()
        return ob['obs']

    def step(self, actions):
        next_obs, rewards, dones, info = self.env.step(actions)
        return next_obs['obs'], rewards, dones, info
