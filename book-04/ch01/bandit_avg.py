import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from bandit import Bandit, Agent

runs = 200
steps = 1000
epsilon = 0.3
all_rates = np.zeros((200, 1000))

for run in tqdm(range(runs)):
    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action,reward)
        total_reward += reward
        rates.append(total_reward / (step+1))

    all_rates[run] = rates

avg_rates = np.average(all_rates, axis=0)

# plot avg rate
plt.ylabel("avg rate per run")
plt.xlabel('steps')
plt.plot(avg_rates)
plt.show()