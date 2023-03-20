import cv2
import os
import numpy as np
from snake import SnakeEnv
import torch
from torch import nn
from torch import optim
from moviepy.editor import ImageSequenceClip
from torch.utils.tensorboard import SummaryWriter


class PolicyEstimator():
    def __init__(self, env):
        self.n_inputs = env.observation_space.n
        self.n_outputs = env.action_space.n

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


def reinforce(env, estimator_policy, num_episodes=1000, batch_size=16, discount_factor=1.0):

    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    optimzer = optim.Adam(estimator_policy.model.parameters(), lr=5e-2)

    action_space = np.arange(env.action_space.n)

    summaryWriter = SummaryWriter()
    os.makedirs(summaryWriter.get_logdir() + '/results', exist_ok=True)
    os.makedirs(summaryWriter.get_logdir() + '/model', exist_ok=True)

    ep = 0
    while ep < num_episodes:
        s_0, _ = env.reset()
        states = []
        actions = []
        rewards = []
        done = False

        while done == False:
            action_probs = estimator_policy.predict(s_0).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)

            s_1, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            states.append(s_0)
            rewards.append(reward)
            actions.append(action)

            s_0 = s_1

            if done:
                batch_rewards.extend(discount_rewards(rewards, discount_factor))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                if batch_counter == batch_size:
                    optimzer.zero_grad()
                    state_tensor = torch.tensor(batch_states, dtype=torch.float)

                    reward_tensor = torch.tensor(batch_rewards, dtype=torch.float)
                    action_tensor = torch.tensor(batch_actions, dtype=torch.long)
                    reward_tensor = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-9)
                    logprob = torch.log(estimator_policy.model(state_tensor))
                    
                    #selected_logprobs = reward_tensor * logprob[range(len(action_tensor)), action_tensor]
                    
                    selected_logprobs = reward_tensor * torch.gather(logprob, 1, action_tensor.unsqueeze(0)).squeeze()
                    loss = -selected_logprobs.mean()

                    loss.backward()

                    optimzer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                    summaryWriter.add_scalar("Loss", loss, ep)
                summaryWriter.add_scalar("Reward", sum(rewards), ep)
                summaryWriter.add_scalar("Food", env.num_food, ep)



                avg_rewards = np.mean(total_rewards[-100:])
                ep += 1
                print("Episode: {}, Average of last 100: {:.2f}".format(ep, avg_rewards))
        if ep % 500 == 0:
            torch.save(estimator_policy.model.state_dict(), summaryWriter.get_logdir() + f'/model/snake_model_{ep}.pt')
            rgb_array = []
            state, _ = env.reset()
            counter = 0
            done = False
            while done == False and counter < 2000: # max iter
                action_probs = estimator_policy.predict(state).detach().numpy()
                action = action_probs.argmax()
                state, reward, done, _, _ = env.step(action)
                rgb_array.append(env.render())
                counter += 1
            resized_imgs = [cv2.resize(x, (256, 256), interpolation=cv2.INTER_NEAREST) for x in rgb_array]
            with ImageSequenceClip(resized_imgs, fps=24) as clip:
                clip.write_gif(os.path.join(summaryWriter.get_logdir(), f'results/{ep}.gif'), fps=24)
    return total_rewards

if __name__ == "__main__":
    env = SnakeEnv(grid_size=10, wall=True, state_type='numpy', max_steps=2000)
    policy_estimator = PolicyEstimator(env)
    rewards = reinforce(env, policy_estimator, batch_size=16, discount_factor=0.9, num_episodes=1000000)