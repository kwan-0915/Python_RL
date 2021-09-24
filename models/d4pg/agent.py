import torch
import ray
import sys
import time
from collections import deque
from utilities.ou_noise import OUNoise
from utilities.logger import Logger
from lunar_lander_wrapper import LunarLanderContinous

class Agent(object):
    def __init__(self, config, policy, n_agent=0, agent_type='exploration', log_dir='', should_exploit=False, shared_object_actor=None):
        print(f"Initializing agent {n_agent}...")
        self.config = config
        self.n_agent = n_agent
        self.agent_type = agent_type
        self.max_steps = config['max_ep_length']
        self.num_episode_save = config['num_episode_save']
        self.local_episode = 0
        self.should_exploit = should_exploit
        self.shared_object_actor = shared_object_actor
        self.global_episode = ray.get(self.shared_object_actor.get_global_episode.remote())
        self.exp_buffer = deque()  # Initialise deque buffer to store experiences for N-step returns

        # Logging
        self.log_dir = log_dir
        log_path = f"{log_dir}/agent-{n_agent}"
        self.logger = Logger(log_path)

        # Create environment
        self.env_wrapper = LunarLanderContinous(config)
        self.ou_noise = OUNoise(dim=config["action_dim"], low=config["action_low"], high=config["action_high"])
        self.ou_noise.reset()

        self.actor = policy
        print("Agent [", self.agent_type, "]", n_agent, self.actor.device)

    def update_actor_learner(self):
        """Update local actor to the actor from learner. """
        if not ray.get(self.shared_object_actor.get_training_on.remote()) or self.should_exploit: return

        try:
            source = ray.get(self.shared_object_actor.get_queue.remote("learner_w_queue")).pop()
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
                self.shared_object_actor.append.remote("replay_queue", [state_0, action_0, discounted_reward, next_state, done, gamma])
            except KeyError:
                sys.exit(-1)

    def run(self):
        best_reward = -float("inf")
        rewards = []

        while ray.get(self.shared_object_actor.get_training_on.remote()):
            episode_reward = 0
            num_steps = 0
            self.local_episode += 1
            self.global_episode += 1
            self.exp_buffer.clear()

            if self.local_episode % 100 == 0:
                print(f"Agent: {self.n_agent}  episode {self.local_episode}")

            ep_start_time = time.time()
            state = self.env_wrapper.reset()
            self.ou_noise.reset()
            done = False

            while not done:
                action = self.actor.get_action(state)
                if self.agent_type == "exploration":
                    action = self.ou_noise.get_action(action, num_steps)
                    action = action.squeeze(0)
                else:
                    action = action.detach().cpu().numpy().flatten()

                next_state, reward, done = self.env_wrapper.step(action)

                episode_reward += reward

                state = self.env_wrapper.normalise_state(state)
                reward = self.env_wrapper.normalise_reward(reward)

                self.exp_buffer.append((state, action, reward))

                # We need at least N steps in the experience buffer before we can compute Bellman
                # rewards and add an N-step experience to replay memory
                if len(self.exp_buffer) >= self.config['n_step_returns']:
                    self._append_queue(next_state, done)

                state = next_state

                if done or num_steps == self.max_steps:
                    # add rest of experiences remaining in buffer
                    while len(self.exp_buffer) != 0:
                        self._append_queue(next_state, done)

                    break

                num_steps += 1

            # Log metrics
            step = ray.get(self.shared_object_actor.get_update_step.remote())
            self.logger.scalar_summary("agent/reward", episode_reward, step)
            self.logger.scalar_summary("agent/episode_timing", time.time() - ep_start_time, step)

            # Saving agent
            reward_outperformed = episode_reward - best_reward > self.config["save_reward_threshold"]
            time_to_save = self.local_episode % self.num_episode_save == 0
            if self.n_agent == 0 and (time_to_save or reward_outperformed):
                if episode_reward > best_reward:
                    best_reward = episode_reward

            rewards.append(episode_reward)
            if self.agent_type == "exploration" and self.local_episode % self.config['update_agent_ep'] == 0:
                self.update_actor_learner()

        print("Agent [", self.agent_type, "]", {self.n_agent}, " done.")
