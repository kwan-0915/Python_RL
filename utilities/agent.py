import os
import ray
import sys
import torch
from collections import deque
from utilities.utils import create_env_wrapper
from utilities.ou_noise import OUNoise
from utilities.logger import Logger

""""
Agent to interact with the env, for d3pg | d4pg
"""

class Agent(object):
    def __init__(self, config, policy, n_agent=0, agent_type='exploration', log_dir='', should_exploit=False, shared_actor=None):
        print(f"Initializing {agent_type}-agent-{n_agent}...")
        self.config = config
        self.n_agent = n_agent
        self.agent_type = agent_type
        self.max_steps = config['max_ep_length']
        self.num_episode_save = config['num_episode_save']
        self.local_episode = 0
        self.should_exploit = should_exploit
        self.shared_actor = shared_actor
        self.global_episode = ray.get(self.shared_actor.get_global_episode.remote()) if self.shared_actor is not None else None
        self.exp_buffer = deque()  # Initialise deque buffer to store experiences for N-step returns

        # Logging
        self.log_dir = log_dir
        log_path = f"{log_dir}/agent-{self.agent_type}-{n_agent}"
        self.logger = Logger(log_path)

        # Create environment
        self.env_wrapper = create_env_wrapper(config)
        self.ou_noise = OUNoise(dim=config["action_dim"], low=config["action_low"], high=config["action_high"])
        self.ou_noise.reset()

        self.actor = policy
        print("Agent [", self.agent_type, "]", n_agent, self.actor.device)

    def update_actor_learner(self):
        """Update local actor to the actor from learner. """
        if not ray.get(self.shared_actor.get_training_on.remote()) or self.should_exploit: return

        try:
            source = ray.get(self.shared_actor.get_queue.remote("learner_w_queue")).pop()
        except IndexError:
            return

        target = self.actor
        for target_param, source_param in zip(target.parameters(), source):
            w = torch.tensor(source_param).float()
            target_param.data.copy_(w)

        if source is not None: del source

    def _append_queue(self, next_state, done):
        state_0, action_0, reward_0 = self.exp_buffer.popleft()
        discounted_reward = reward_0
        gamma = self.config['discount_rate']
        for (_, _, r_i) in self.exp_buffer:
            discounted_reward += r_i * gamma
            gamma *= self.config['discount_rate']

        if self.agent_type == "exploration":
            try:
                self.shared_actor.append.remote("replay_queue", [state_0, action_0, discounted_reward, next_state, done, gamma])
            except KeyError:
                sys.exit(-1)

    def run(self):
        raise NotImplementedError("Run method must be implemented for different child")

    def save(self, checkpoint_name):
        process_dir = f"{self.log_dir}/agent_{self.agent_type}_{self.n_agent}"
        if not os.path.exists(process_dir): os.makedirs(process_dir)

        model_fn = f"{process_dir}/{checkpoint_name}.pt"
        torch.save(self.actor, model_fn)

    def save_plot(self):
        pass