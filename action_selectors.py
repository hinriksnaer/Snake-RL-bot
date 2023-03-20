import random
import torch
import torch.nn.functional as F
import numpy as np


# define class for random action with episode start, episode end and episode decay
class RandomActionSelector:
    def __init__(self, n_actions, eps_start, eps_end, eps_decay, device):
        self.n_actions = n_actions
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.device = device

    def __call__(self, policy, state, total_steps):
        sample = random.random()
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * total_steps / self.eps_decay)
        if sample > epsilon:
            with torch.no_grad():
                return policy(torch.tensor(state, device=self.device)).argmax().view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)


# define class for probabilistic action
class ProbabilisticActionSelector:
    def __init__(self, n_actions, device):
        self.n_actions = n_actions
        self.device = device

    def __call__(self, policy, state, total_steps):
        with torch.no_grad():
            probs = F.softmax(policy(torch.tensor(state, device=self.device)), dim=1)
            action = probs.multinomial(num_samples=1)
            return action
