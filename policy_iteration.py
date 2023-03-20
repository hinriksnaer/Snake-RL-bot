import numpy as np
import gym
    
class PolicyIteration(object):
    def __init__(self, env, discount_factor=0.9, theta=0.0001, n_states=16, n_actions=4):
        self.n_actions = n_actions
        self.n_states = n_states
        self.env = env
        self.discount_factor = discount_factor
        self.theta = theta
        self.V = np.zeros(self.n_states)
        self.policy = np.ones([self.n_states, self.n_actions]) / self.n_actions
        self.policy_stable = False
    
    def policy_evaluation(self):
        while True:
            delta = 0
            for s in range(self.n_states):
                v = 0
                for a, action_prob in enumerate(self.policy[s]):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        v += action_prob * prob * (reward + self.discount_factor * self.V[next_state])
                delta = max(delta, np.abs(v - self.V[s]))
                self.V[s] = v
            if delta < self.theta:
                break
    
    def policy_improvement(self):
        self.policy_stable = True
        for s in range(self.n_states):
            chosen_a = np.argmax(self.policy[s])
            action_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    action_values[a] += prob * (reward + self.discount_factor * self.V[next_state])
            best_a = np.argmax(action_values)
            if chosen_a != best_a:
                self.policy_stable = False
            self.policy[s] = np.eye(self.n_actions)[best_a]


if __name__ == '__main__':
    env = gym.make('CliffWalking-v0')
    env = BufferWrapper(env, 1)
    pi = PolicyIteration(env, n_states=48, n_actions=4)
    while not pi.policy_stable:
        pi.policy_evaluation()
        pi.policy_improvement()
    print(pi.policy) 