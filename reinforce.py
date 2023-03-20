import cv2
import os
import numpy as np
from snake import SnakeEnv
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
from moviepy.editor import ImageSequenceClip
from collections import deque
from itertools import count
from torch.utils.tensorboard import SummaryWriter


class PolicyEstimator():
    def __init__(self, env):
        self.n_inputs = env.observation_space.n
        self.n_outputs = env.action_space.n
        self.stored_log_probs = []
        self.rewards = []

        self.model = nn.Sequential(
            nn.Linear(self.n_inputs, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.n_outputs),
            nn.Softmax(dim=-1)
        )
    
    def predict(self, x):

        x = torch.tensor(x).float()
        x = self.model(x)

        return x

def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()

def select_action(state, policy):
    state = torch.tensor(state).float()
    probs = policy.predict(state)
    m = Categorical(probs)
    action = m.sample()
    policy.stored_log_probs.append(m.log_prob(action))
    return action.item()

def finish_episode(policy, optimizer, gamma=0.99, eps=1e-9):
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.stored_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    del policy.rewards[:]
    del policy.stored_log_probs[:]
    return policy_loss.item()


def main(env, policy, discount_factor=1.0, eps=1e-9):


    reward_history = deque(maxlen=100)

    optimizer = optim.Adam(policy.model.parameters(), lr=1e-4)
    summaryWriter = SummaryWriter()
    os.mkdir(os.path.join(summaryWriter.get_logdir(), 'results'))
    os.mkdir(os.path.join(summaryWriter.get_logdir(), 'model'))
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state, policy)
            state, reward, done, _, _ = env.step(action)
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        reward_history.append(ep_reward)

        loss = finish_episode(policy, optimizer, gamma=discount_factor, eps=eps)
        summaryWriter.add_scalar('reward', ep_reward, i_episode)
        summaryWriter.add_scalar('food', env.num_food, i_episode)
        summaryWriter.add_scalar('loss', loss, i_episode)
        print(f'Episode {i_episode}\tLast 100 rewards: {sum(reward_history)/len(reward_history):.2f}\tLoss: {loss:.2f}\tFood: {env.num_food}')

        if i_episode % 500 == 0:
            torch.save(policy.model.state_dict(), summaryWriter.get_logdir() + f'/model/snake_model_{i_episode}.pt')
            rgb_array = []
            state, _ = env.reset()
            counter = 0
            done = False
            while done == False and counter < 4000: # max iter
                action_probs = policy.predict(state).detach().numpy()
                action = action_probs.argmax()
                state, reward, done, _, _ = env.step(action)
                rgb_array.append(env.render())
                counter += 1
            #resized_imgs = [cv2.resize(x, (256, 256), interpolation=cv2.INTER_NEAREST) for x in rgb_array]
            with ImageSequenceClip(rgb_array, fps=24) as clip:
                clip.write_gif(os.path.join(summaryWriter.get_logdir(), f'results/{i_episode}.gif'), fps=24)


if __name__ == "__main__":
    checkpoint = torch.load('runs/Dec25_23-05-54_supercomputer/model/snake_model_168000.pt')
    env = SnakeEnv(grid_size=10, wall=True, state_type='numpy', max_steps=250)
    policy_estimator = PolicyEstimator(env)
    policy_estimator.model.load_state_dict(checkpoint)
    rewards = main(env, policy_estimator, discount_factor=0.9, eps=1e-9)