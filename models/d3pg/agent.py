import ray
import time
from utilities.agent import Agent

class D3PGAgent(Agent):

    def __init__(self, config, policy, n_agent=0, agent_type='exploration', log_dir='', should_exploit=False, shared_actor=None):
        super(D3PGAgent, self).__init__(config,
                                        policy,
                                        n_agent,
                                        agent_type,
                                        log_dir,
                                        should_exploit,
                                        shared_actor)

    def run(self):
        best_reward = -float("inf")
        rewards = []

        while ray.get(self.shared_actor.get_training_on.remote()):
            episode_reward = 0
            num_steps = 0
            self.local_episode += 1
            self.global_episode += 1
            self.exp_buffer.clear()

            if self.local_episode % 100 == 0:
                print("Agent [", self.agent_type, "]", {self.n_agent}, " episode : ", self.local_episode, "")

            ep_start_time = time.time()
            state = self.env_wrapper.reset()
            self.ou_noise.reset()
            done = False

            while not done:
                action = self.actor.get_action(state)
                if self.agent_type == "exploration":
                    action = self.ou_noise.get_action(action, num_steps)
                    action = action.squeeze()
                else:
                    action = action.detach().cpu().numpy().flatten()

                next_state, reward, done, _ = self.env_wrapper.step(action)
                num_steps += 1

                if num_steps == self.max_steps: done = True

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
            step = ray.get(self.shared_actor.get_update_step.remote())
            self.logger.scalar_summary(f"agent_{self.agent_type}/reward", episode_reward, step)
            self.logger.scalar_summary(f"agent_{self.agent_type}/episode_timing", time.time() - ep_start_time, step)

            # Saving agent
            reward_outperformed = episode_reward - best_reward > self.config["save_reward_threshold"]
            # time_to_save = self.local_episode % self.num_episode_save == 0
            if self.agent_type == 'exploitation' and (self.local_episode == 1 or reward_outperformed):
                if episode_reward > best_reward:
                    best_reward = episode_reward

                self.save(f"local_episode_{self.local_episode}_reward_{best_reward:4f}")

            rewards.append(episode_reward)
            if self.agent_type == "exploration" and self.local_episode % self.config['update_agent_ep'] == 0:
                self.update_actor_learner()

        print("Agent [", self.agent_type, "]", {self.n_agent}, " done.")
        self.shared_actor.set_child_threads.remote()

    def save_plot(self):
        state = self.env_wrapper.reset()
        total_reward = 0
        for step in range(self.max_steps):
            action = self.actor.get_action(state)
            action = action.cpu().detach().numpy().squeeze()
            next_state, reward, done, info = self.env_wrapper.step(action)
            total_reward += reward
            state = next_state
            if done: break

        print("total rewards: {}".format(total_reward))
        self.env_wrapper.plot()