import gym
from gym import spaces
import random
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
import cv2
import numpy as np
import math

class SnakeEnv(gym.Env):
    def rgb2grayscale(self, frame):
        grayscale = np.ones((frame.shape[0], frame.shape[1])) * 0
        grayscale[frame[..., 0] == 255] = 1
        grayscale[frame[..., 1] == 255] = -1
        return grayscale

    def gen_obs(self, state_type='numpy'):
        if state_type == 'numpy':
            if self.max_steps == np.inf or self.steps == 0:
                remaining_steps = 1
            else:
                remaining_steps =  1 - self.steps/self.max_steps

            snake_obs = np.zeros((self.grid_size, self.grid_size))
            snake_obs[tuple(np.array(self.snake).T)] = 1
            
            food_row, food_col = self.food
            food_col_ops = np.zeros(self.grid_size)
            food_col_ops[food_col] = 1
            food_row_ops = np.zeros(self.grid_size)
            food_row_ops[food_row] = 1

            head_row, head_col = self.snake[0]
            head_col_obs = np.zeros(self.grid_size)
            head_col_obs[head_col] = 1
            head_row_obs = np.zeros(self.grid_size)
            head_row_obs[head_row] = 1

            return snake_obs.flatten().tolist() + food_col_ops.tolist() + food_row_ops.tolist() + head_col_obs.tolist() + head_row_obs.tolist() + [remaining_steps]
        if state_type == 'grayscale':
            return self.rgb2grayscale(cv2.resize(self.gen_frame(), (self.height, self.width), interpolation=cv2.INTER_NEAREST))

    def gen_frame(self):
        obs = np.array(self.gen_obs())[:- 2*self.grid_size]
        snake_obs = obs[:self.grid_size ** 2].reshape(self.grid_size, self.grid_size)

        food_col_ops = obs[self.grid_size ** 2: self.grid_size ** 2 + self.grid_size]
        food_row_ops = obs[self.grid_size **2 + self.grid_size: self.grid_size ** 2 + self.grid_size * 2]

        food_obs = np.zeros((self.grid_size, self.grid_size))
        food_obs[np.where(food_row_ops == 1)[0][0], np.where(food_col_ops == 1)[0][0]] = 1

        zero_grid = np.zeros((self.grid_size, self.grid_size))
        image = np.stack([snake_obs, food_obs, zero_grid], axis=2)
        image *= 255
        return image

    def __init__(self, grid_size=10, wall=True, state_type='numpy', width=512, height=512, max_steps=np.inf):
        self.grid_size = grid_size
        self.wall = wall
        self.width = width
        self.height = height
        self.positions = [(row, col) for col in range(self.grid_size) for row in range(self.grid_size)]
        self.action_space = spaces.Discrete(4)
        self.state_type = state_type
        self.max_steps = max_steps
        self.steps = 0
        if self.state_type == 'numpy':
            self.observation_space = spaces.Discrete(
                self.grid_size ** 2 + 4 * self.grid_size + 1
            )
        elif self.state_type == 'grayscale':
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(1, self.height, self.width), dtype=np.uint8
            )

        self.num_food = 0

        self.last_action = 'LEFT'

        self.snake = [
            (self.grid_size//2, self.grid_size//2),
            (self.grid_size//2, self.grid_size//2 + 1),
            (self.grid_size//2, self.grid_size//2 + 2),
            (self.grid_size//2, self.grid_size//2 + 3),
        ]
        self.food = (self.snake[0][0], self.snake[0][1] - 3)
        self.last_dist = np.sqrt((self.food[0] - self.snake[0][0])**2 + (self.food[1] - self.snake[0][1])**2)

    def step(self, action):
        self.steps += 1
        action_name = {
            0: 'UP',
            1: 'RIGHT',
            2: 'DOWN',
            3: 'LEFT'
        }

        text_to_action = {
            'UP': (-1, 0),
            'RIGHT': (0, 1),
            'DOWN': (1, 0),
            'LEFT': (0, -1)
        }

        action_text = action_name[action]
        '''
        if action_text == 'LEFT':
            self.last_action -= 90
        elif action_text == 'RIGHT':
            self.last_action += 90
        '''
        if action_text == 'UP' and self.last_action == 'DOWN':
            action_text = 'DOWN'
        elif action_text == 'DOWN' and self.last_action == 'UP':
            action_text = 'UP'
        elif action_text == 'LEFT' and self.last_action == 'RIGHT':
            action_text = 'RIGHT'
        elif action_text == 'RIGHT' and self.last_action == 'LEFT':
            action_text = 'LEFT'

        self.last_action = action_text

        #action_row, action_col = round(math.sin(math.radians(self.last_action))), round(math.cos(math.radians(self.last_action)))
        action_row, action_col = text_to_action[action_text]
        head_row, head_col = self.snake[0]
        # move snake

        if not self.wall:
            new_head_col = head_col + action_col if head_col + action_col >= 0 else self.grid_size - 1
            new_head_row = head_row + action_row if head_row + action_row >= 0 else self.grid_size - 1
            new_head_col = new_head_col % self.grid_size
            new_head_row = new_head_row % self.grid_size
        else:
            new_head_col = head_col + action_col
            new_head_row = head_row + action_row

        if self.steps == self.max_steps:
            return self.gen_obs(self.state_type), -1000, True, False, {}

        # check if snake is dead
        if not self.wall:
            if (new_head_row, new_head_col) in self.snake:
                # show frames as gif render

                return self.gen_obs(self.state_type), -1000, True, False, {}
        if self.wall:
            if new_head_col < 0 or new_head_col >= self.grid_size or new_head_row < 0 or new_head_row >= self.grid_size or (new_head_row, new_head_col) in self.snake:
                # show frames as gif render

                return self.gen_obs(self.state_type), -1000, True, False, {}

        self.snake.insert(0, (new_head_row, new_head_col))
        
        if (new_head_row, new_head_col) == self.food:
            self.food = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            self.steps = 0
            if self.food in self.snake:
                if len(self.snake) == self.grid_size ** 2:
                    return self.gen_obs(self.state_type), 1000, True, False, {}
                self.food = random.choice(list(set(self.positions) - set(self.snake)))
            self.num_food += 1
            self.last_dist = np.sqrt((self.food[0] - new_head_row)**2 + (self.food[1] - new_head_col)**2) 
            return self.gen_obs(self.state_type), 100, False, False, {}
        else:
            self.snake.pop()
            # compute distance from head to food using pythagorean theorem
            dist = np.sqrt((self.food[0] - new_head_row)**2 + (self.food[1] - new_head_col)**2)
            dist_diff = self.last_dist - dist
            self.last_dist = dist

            #print('dist_diff', dist_diff, 'action', action_text)
            #print(np.array(self.gen_obs()[:self.grid_size**2]).reshape(self.grid_size, self.grid_size))
            return self.gen_obs(self.state_type), 0, False, False, {}

    def reset(self):
        self.snake = [
            (self.grid_size//2, self.grid_size//2),
            (self.grid_size//2, self.grid_size//2 + 1),
            (self.grid_size//2, self.grid_size//2 + 2),
            (self.grid_size//2, self.grid_size//2 + 3),
        ]
        self.food = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
        if self.food in self.snake:
            self.food = random.choice(list(set(self.positions) - set(self.snake)))
        self.num_food = 0
        self.steps = 0
        self.last_action = 'LEFT'
        self.last_dist = np.sqrt((self.food[0] - self.snake[0][0])**2 + (self.food[1] - self.snake[0][1])**2)
        return self.gen_obs(self.state_type), {}

    def render(self):

        image = self.gen_frame()
        return image

if __name__ == '__main__':
 
    env = SnakeEnv(grid_size=10, wall=True, width=10, height=10, state_type='grayscale')
    env.reset()
    done = False

    while not done:
        # take random action
        state, action, done, _, _ = env.step(random.randrange(3))
