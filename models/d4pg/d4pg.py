import ray
import sys
import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from models.d4pg.critic import Critic
from utilities.ou_noise import OUNoise
from utilities.l2_projection import l2_project
from utilities.logger import Logger

class D4PG(object):
    """Actor and Critic update routine. """

    def __init__(self, config, actor, target_actor, shared_actor, log_dir=''):
        hidden_dim = config['dense_size']
        state_dim = config['state_dim']
        action_dim = config['action_dim']
        critic_lr = config['critic_learning_rate']
        actor_lr = config['actor_learning_rate']
        self.v_min = config['v_min']
        self.v_max = config['v_max']
        self.num_atoms = config['num_atoms']
        self.device = config['device']
        self.max_steps = config['max_ep_length']
        self.num_train_steps = config['num_steps_train']
        self.batch_size = config['batch_size']
        self.tau = config['tau']
        self.gamma = config['discount_rate']
        self.n_step_return = config['n_step_returns']
        self.prioritized_replay = config['replay_memory_prioritized']
        self.shared_actor = shared_actor
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        # Logging
        self.log_dir = log_dir
        self.logger = Logger(f"{log_dir}/learner")

        # Noise process
        self.ou_noise = OUNoise(dim=config["action_dim"], low=config["action_low"], high=config["action_high"])

        # Base Actor and Critic
        self.actor = actor
        self.critic = Critic(state_dim, action_dim, hidden_dim, self.v_min, self.v_max, self.num_atoms, device=self.device)

        # Target Actor and Critic
        self.target_actor = target_actor
        self.target_critic = Critic(state_dim, action_dim, hidden_dim, self.v_min, self.v_max, self.num_atoms, device=self.device)

        # Copy Actor to Target Actor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        # Copy Critic to Target Critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Actor and Critic Optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.value_criterion = nn.BCELoss(reduction='none')

    def _update_step(self, batch):
        update_time = time.time()

        state, action, reward, next_state, done, gamma, weights, inds = batch

        # state_c = deepcopy(state)
        # action_c = deepcopy(np.asarray(action))
        # next_state_c = deepcopy(np.asarray(next_state))
        # reward_c = deepcopy(np.asarray(reward))
        # done_c = deepcopy(np.asarray(done))
        # weights_c = deepcopy(np.asarray(weights))
        # inds_c = deepcopy(np.asarray(inds).flatten())
        #
        # state = torch.from_numpy(state_c).float().cuda()
        # action = torch.from_numpy(action_c).float().cuda()
        # next_state = torch.from_numpy(next_state_c).float().cuda()
        # reward = torch.from_numpy(reward_c).float().cuda()
        # done = torch.from_numpy(done_c).float().cuda()

        state = torch.from_numpy(state).float().cuda()
        action = torch.from_numpy(action).float().cuda()
        next_state = torch.from_numpy(next_state).float().cuda()
        reward = torch.from_numpy(reward).float().cuda()
        done = torch.from_numpy(done).float().cuda()

        # ------- Update critic -------

        # Predict next actions with target policy network
        next_action = self.target_actor(next_state)

        # Predict Z distribution with target value network
        target_value = self.target_critic.get_probs(next_state, next_action.detach())

        # Get projected distribution
        target_z_projected = l2_project(next_distr_v=target_value, rewards_v=reward, dones_mask_t=done,
                                        gamma=self.gamma ** self.n_step_return, n_atoms=self.num_atoms, v_min=self.v_min,
                                        v_max=self.v_max, delta_z=self.delta_z)

        target_z_projected = torch.from_numpy(target_z_projected).float().cuda()

        critic_value = self.critic.get_probs(state, action)
        critic_value = critic_value.cuda()

        critic_loss = self.value_criterion(critic_value, target_z_projected)
        critic_loss = critic_loss.mean(axis=1)

        # Update priorities in buffer
        td_error = critic_loss.cpu().detach().numpy().flatten()
        priority_epsilon = 1e-4
        if self.prioritized_replay:
            weights_update = np.abs(td_error) + priority_epsilon
            self.shared_actor.append.remote("replay_priority_queue", (inds, weights_update))
            critic_loss = critic_loss * torch.tensor(weights).float().cuda()

        # Update step
        critic_loss = critic_loss.mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------- Update actor -----------
        actor_loss = self.critic.get_probs(state, self.actor(state))
        actor_loss = actor_loss * torch.from_numpy(self.critic.z_atoms).float().cuda()
        actor_loss = torch.sum(actor_loss, dim=1)
        actor_loss = -actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        # Send updated learner to the queue
        if ray.get(self.shared_actor.get_update_step.remote()) % 100 == 0:
            try:
                params = [p.data.cpu().detach().numpy() for p in self.actor.parameters()]
                self.shared_actor.append.remote("learner_w_queue", params)
            except KeyError:
                sys.exit(-1)

        # del state_c, action_c, reward_c, next_state_c, done_c, weights_c, inds_c

        # Logging
        step = ray.get(self.shared_actor.get_update_step.remote())
        self.logger.scalar_summary("learner/actor_loss", actor_loss.item(), step)
        self.logger.scalar_summary("learner/critic_loss", critic_loss.item(), step)
        self.logger.scalar_summary("learner/learner_update_timing", time.time() - update_time, step)

    def run(self):
        while ray.get(self.shared_actor.get_update_step.remote()) < self.num_train_steps:
            try:
                batch = ray.get(self.shared_actor.get_queue.remote("batch_queue")).pop()
            except IndexError:
                continue

            self._update_step(batch)
            self.shared_actor.set_update_step.remote()

            if ray.get(self.shared_actor.get_update_step.remote()) % 1000 == 0:
                print("Training step ", ray.get(self.shared_actor.get_update_step.remote()))

        self.shared_actor.set_training_on.remote(0)

        print("Exit learner.")
        self.shared_actor.set_child_threads.remote()
