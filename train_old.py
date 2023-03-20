import torch
import torch.nn as nn
from collections import namedtuple, deque
from itertools import count
from snake import SnakeEnv
import numpy as np
import gym
from moviepy.editor import ImageSequenceClip
import random
from collections import deque
import cv2
import os
from action_selectors import ProbabilisticActionSelector, RandomActionSelector
from models import DQNConv
from torch.utils.tensorboard import SummaryWriter


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class FrameBuffer:
    def __init__(self, frame_h, frame_w, n_frames, device='cpu'):
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.device = device
        self.n_frames = n_frames
        self.buffer = torch.zeros((n_frames, frame_h, frame_w), dtype=torch.float32, device=self.device)
        self.reset()
        
    def reset(self):
        self.buffer.zero_()
            
    def push(self, frame):
        self.buffer[:-1] = self.buffer[1:].clone()
        self.buffer[-1] = frame
       
        
    def get(self):
        return self.buffer


class TensorboardWriter:
    def __init__(self):
        self.writer = SummaryWriter()
        self.counter = 1
        
    def write(self, loss, name='Loss/train'):
        self.writer.add_scalar(name, loss, int(self.counter))
        self.counter += 1

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)
                    
def optimize_model(policy_net, target_net, criterion, optimizer, memory, batch_size, discount, writer, device='cpu'):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * discount) + reward_batch

    # Compute Huber loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    writer.write(loss.item())
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def evaluate_policy(policy_net, env, device, max_iter, iterator, log_dir, frame_buffer = None):
    if frame_buffer is not None:
        frame_buffer.reset()
    rgb_array = []
    state, _ = env.reset()
    state = state_to_tensor(state, frame_buffer, device)
    rgb_array.append(env.render())
    done = False
    truncated = False
    counter = 0
    while (not done and not truncated) and counter < max_iter:
        
        with torch.no_grad():
            action = policy_net(torch.tensor(state, device=device)).argmax().item()
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = state_to_tensor(next_state, frame_buffer, device)
            state = next_state
            rgb_array.append(env.render())
        counter += 1
    with ImageSequenceClip(rgb_array, fps=24) as clip:
        resized_imgs = [cv2.resize(x, (256, 256), interpolation=cv2.INTER_NEAREST) for x in rgb_array]
        clip.write_gif(os.path.join(log_dir, f'results/{iterator}.gif'), fps=24)

def state_to_tensor(state, frame_buffer, device):
    state = torch.tensor(state, device=device, dtype=torch.float32)
    if frame_buffer is not None:
        frame_buffer.push(state)
        state = frame_buffer.get()

    state = state.unsqueeze(0)
    return state


def train():
    discount = 0.99
    batch_size = 128
    num_episodes = 10000000
    device = 'cuda'
    output_type = 'image'
    state_type = 'grayscale'
    resume = None#'state_dict/15400.pt'
    tau = 0.005
    eps_start = 0.9
    eps_end = 0.1
    eps_decay = 10000
    total_steps = 0
    memory_size = 100000
    learning_rate = 1e-3
    conv_layers = [32, 64, 64]
    fcn_layers = [512, 128]
    frame_buffer_size = 2
    image_width = 70
    image_height = 70

    #env = gym.make('CartPole-v1', render_mode='rgb_array') #SnakeEnv()
    env = SnakeEnv(grid_size=10, wall=True, state_type=state_type, width=image_width, height=image_height)
    memory = ReplayMemory(memory_size)
    if output_type == 'image':
        state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    #select_action = ProbabilisticActionSelector(n_actions=action_dim, device=device)
    select_action = RandomActionSelector(action_dim, eps_start, eps_end, eps_decay, device=device)

    model = 'DQN'
    if model == 'DQN':
        if output_type == 'image':
            model = DQNConv
    
    if output_type == 'image':
        policy_net = model(state_dim[0] * frame_buffer_size, state_dim[1:], action_dim, conv_layers, fcn_layers).to(device)
        if resume is not None:
            policy_net.load_state_dict(torch.load(resume))
        target_net = model(state_dim[0] * frame_buffer_size, state_dim[1:], action_dim, conv_layers, fcn_layers).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        frame_buffer = FrameBuffer(state_dim[1], state_dim[2], frame_buffer_size, device)
    

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss(reduction='sum')
    
    writer = TensorboardWriter()

    log_dir = writer.writer.get_logdir()
    os.mkdir(log_dir + '/state_dict')
    os.mkdir(log_dir + '/results')


    for i in range(num_episodes):
        if frame_buffer is not None:
            frame_buffer.reset()

        rewards = []
        state, _ = env.reset()
        state = state_to_tensor(state, frame_buffer, device)
 
        done = False

        for t in count():

            action = select_action(policy_net, state, total_steps)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            rewards += [reward]
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            if done:
                next_state = None
            else:
                next_state = state_to_tensor(next_state, frame_buffer, device)
            memory.push(*(state, action, next_state, reward))

            state = next_state

            optimize_model(policy_net, target_net, criterion, optimizer, memory, batch_size, discount, writer, device)
            total_steps += 1
            # Update the target network, copying all weights and biases in DQN
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                writer.write(t, 'Duration/train')
                writer.write(sum(rewards), 'Rewards/train')
                writer.write(env.num_food, 'Food/train')
                break
        print('Iteration:', i,'num apples:' , env.num_food,'reward:', round(sum(rewards),2), 'decay:', round(eps_end + (eps_start - eps_end) * np.exp(-1. * total_steps / eps_decay), 3))
        if i % 100 == 0:
            print('evaluating policy')
            if output_type == 'image':
                evaluate_policy(policy_net, env, device, 3000, i, log_dir, frame_buffer)
            else:
                evaluate_policy(policy_net, env, device, 3000, i, log_dir)
            env.render()
            torch.save(policy_net.state_dict(), os.path.join(log_dir,f'state_dict/{i}.pt'))

    print('done')

if __name__ == '__main__':
    train()
