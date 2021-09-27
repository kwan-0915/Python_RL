import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from .actor import Actor
from .critic import Critic
from utilities.replay_buffer import SimpleReplayBuffer
from tensorboardX import SummaryWriter

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, args, directory):
        self.max_action = max_action
        self.device = args.device
        self.discount = args.discount
        self.tau = args.tau
        self.policy_noise = args.policy_noise * self.max_action
        self.noise_clip = args.noise_clip * self.max_action
        self.policy_freq = args.policy_freq
        self.batch_size = args.batch_size
        self.total_it = 0
        self.directory = directory
        self.writer = SummaryWriter(self.directory)

        self.replay_buffer = SimpleReplayBuffer(args.capacity)

        self.actor = Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        self.total_it += 1

        # Sample replay buffer
        x, y, u, r, d = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(x).to(self.device)
        next_state = torch.FloatTensor(y).to(self.device)
        action = torch.FloatTensor(u).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)
        done = torch.FloatTensor(1 - d).to(self.device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self):
        torch.save(self.critic.state_dict(), self.directory + "critic.pth")
        torch.save(self.critic_optimizer.state_dict(), self.directory + "critic_optimizer.pth")

        torch.save(self.actor.state_dict(), self.directory + "actor.pth")
        torch.save(self.actor_optimizer.state_dict(), self.directory + "actor_optimizer.pth")

    def load(self):
        self.critic.load_state_dict(torch.load(self.directory + "critic.pth"))
        self.critic_optimizer.load_state_dict(torch.load(self.directory + "critic_optimizer.pth"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(self.directory + "actor.pth"))
        self.actor_optimizer.load_state_dict(torch.load(self.directory + "actor_optimizer.pth"))
        self.actor_target = copy.deepcopy(self.actor)