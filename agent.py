import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import torch.nn as nn


def discount_rewards(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 128
        self.tanh = torch.nn.Tanh()
        self.log_std = nn.Parameter(torch.zeros(self.action_space))

        self.actor = nn.Sequential(
            nn.Linear(state_space, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 256),
            nn.ReLU(),
            nn.Linear(256, action_space)
        ).float()

        self.critic = nn.Sequential(
            nn.Linear(state_space, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).float()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor-Critic forward
        """
        # Actor: ottieni la mean delle azioni
        action_mean = self.actor(x)
        sigma = torch.exp(self.log_std)
        distribution = Normal(action_mean, sigma)

        # Critic: ottieni il valore stimato dallo stato
        value = self.critic(x)

        return distribution, value      


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizerActor = torch.optim.Adam(policy.actor.parameters(), lr=1e-4)
        self.optimizerCritic = torch.optim.Adam(policy.critic.parameters(), lr=1e-4)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.values = []
        self.masks =[]


    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        masks = torch.Tensor(self.masks).to(self.train_device)
        values = torch.stack(self.values, dim=0).to(self.train_device).squeeze(-1)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done, self.values, self.masks = [], [], [], [], [], [], []

        next_state = states[-1].float().to(self.train_device)
        _, next_value = self.policy(next_state)
        returns = discount_rewards(next_value, rewards, masks)

        log_probs = action_log_probs
        returns = torch.stack(returns).detach()

        advantage = returns - values

        self.actor_loss = -(log_probs * advantage.detach()).mean()
        self.critic_loss = advantage.pow(2).mean()

        self.optimizerActor.zero_grad()
        self.optimizerCritic.zero_grad()
        self.actor_loss.backward()
        self.critic_loss.backward()
        self.optimizerActor.step()
        self.optimizerCritic.step()

        return        


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        dist, value = self.policy(x)

        if evaluation:  # Return mean
            return dist.mean.detach().cpu().numpy(), None  # azione deterministica


        else:   # Sample from the distribution
            action = dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = dist.log_prob(action).sum()

            return action, action_log_prob, value


    def store_outcome(self, state, next_state, action_log_prob, reward, done, value, mask):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
        self.values.append(value)
        self.masks.append(mask)