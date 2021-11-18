import ray
import sys
import copy
import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from models.d3pg.critic import Critic
from utilities.ou_noise import OUNoise
from utilities.logger import Logger

class D3PG(object):
    """Actor and Critic update routine. """

    def __init__(self, config, actor, target_actor, shared_actor, log_dir=''):
        num_asset = config['num_asset'] + int(config['add_cash_asset'])  # get num of asset for first dim of state and action for replay buffer
        hidden_dim = config['dense_size']
        action_dim = num_asset * config["action_dim"]
        critic_lr = config['critic_learning_rate']
        actor_lr = config['actor_learning_rate']
        n_features = config['n_features']
        seq_len = config["state_dim"]
        self.num_train_steps = config['num_steps_train']
        self.device = config['device']
        self.max_steps = config['max_ep_length']
        self.frame_idx = 0
        self.batch_size = config['batch_size']
        self.gamma = config['discount_rate']
        self.tau = config['tau']
        self.shared_actor = shared_actor

        # Logging
        self.log_dir = log_dir
        self.logger = Logger(f"{log_dir}/learner")

        # Noise process
        self.ou_noise = OUNoise(dim=config["action_dim"], low=config["action_low"], high=config["action_high"])

        # Base Actor and Critic
        self.actor = actor
        self.critic = Critic(n_features, seq_len, action_dim, hidden_dim, device=self.device)
        
        # Target Actor and Critic
        self.target_actor = target_actor
        self.target_critic = copy.deepcopy(self.critic)
        
        # Actor and Critic Optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.value_criterion = nn.MSELoss(reduction='none')

    def _update_step(self, batch, min_value=-np.inf, max_value=np.inf):
        update_time = time.time()

        state, action, reward, next_state, done = batch

        # state_c = deepcopy(state)
        # action_c = deepcopy(action)
        # reward_c = deepcopy(reward)
        # next_state_c = deepcopy(next_state)
        # done_c = deepcopy(done)
        #
        # # Move to CUDA
        # state = torch.from_numpy(state_c).float().to(self.device)
        # action = torch.from_numpy(action_c).float().to(self.device)
        # reward = torch.from_numpy(reward_c).float().to(self.device)
        # next_state = torch.from_numpy(next_state_c).float().to(self.device)
        # done = torch.from_numpy(done_c).float().to(self.device)

        state = torch.from_numpy(state).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)

        # ------- Update critic -------
        next_action = self.target_actor(next_state)

        target_value = self.target_critic(next_state, next_action.detach())

        expected_value = reward + done * self.gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)

        critic_value = self.critic(state, action)
        critic_loss = self.value_criterion(critic_value, expected_value.detach())
        critic_loss = critic_loss.mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------- Update actor --------
        actor_loss = self.critic(state, self.actor(state))
        actor_loss = -actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
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

        # Logging
        step = ray.get(self.shared_actor.get_update_step.remote())
        self.logger.scalar_summary("learner/actor_loss", actor_loss.item(), step)
        self.logger.scalar_summary("learner/critic_loss", critic_loss.item(), step)
        self.logger.scalar_summary("learner/update_time", time.time() - update_time, step)

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
