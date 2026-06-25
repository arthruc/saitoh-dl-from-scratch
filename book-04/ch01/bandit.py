import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



# Multi-arms Bandit
class Bandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0
        


# bandit = Bandit()
# ns = np.zeros(10) # number of played for each bandit
# Qs = np.zeros(10) # action value for each bandit

# for n in range(10):
#     action = np.random.randint(0,10) # which bandit to play
#     reward = bandit.play(action)

#     ns[action] += 1
#     Qs[action] += (reward - Qs[action]) / ns[action]

# print(Qs)
# print(ns)

class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.ns = np.zeros(action_size)
        self.Qs = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


def main():        
    # simple Q calculation
    np.random.seed(0)
    rewards = []
    Q = 0

    steps = 1000
    epsilon = 0.1
    agent = Agent(epsilon)
    bandit = Bandit()


    total_reward = 0
    total_rewards = []
    rates = []

    for step in tqdm(range(steps)):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)

        total_reward += reward

        total_rewards.append(total_reward)
        rates.append(total_reward / (step+1) ) #start with 0


    # total reward plot
    plt.ylabel("total rewards")
    plt.xlabel("steps")
    plt.plot(total_rewards)
    plt.show()

    # rate plot
    plt.ylabel("rate")
    plt.xlabel("steps")
    plt.plot(rates)
    plt.show()


if __name__ == "__main__":
    main()