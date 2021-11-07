import gym
from abc import ABC, abstractmethod
from environments.lunar_lander_wrapper import LunarLanderContinous
from portfolio.fx_portfolio_wrapper import FXPortfolioWrapper

eps = 1e-8

def create_env_wrapper(config):
    env_name = config['env']
    if env_name == "LunarLanderContinuous-v2":
        return LunarLanderContinous(config)
    elif env_name == 'FX':
        return FXPortfolioWrapper

    return EnvWrapper(env_name)

class ABCEnvWrapper(ABC):
    def __init__(self, env_name, env):
        self.env_name = env_name
        self.env = env

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_random_action(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def set_random_seed(self, seed):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_action_space(self):
        pass

    def normalise_state(self, state):
        return state

    def normalise_reward(self, reward):
        return reward

class EnvWrapper:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(self.env_name)

    def reset(self):
        state = self.env.reset()
        return state

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def step(self, action):
        next_state, reward, terminal, _ = self.env.step(action.ravel())
        return next_state, reward, terminal

    def set_random_seed(self, seed):
        self.env.seed(seed)

    def render(self):
        frame = self.env.render(mode='rgb_array')
        return frame

    def close(self):
        self.env.close()

    def get_action_space(self):
        return self.env.action_space

    def normalise_state(self, state):
        return state

    def normalise_reward(self, reward):
        return reward


