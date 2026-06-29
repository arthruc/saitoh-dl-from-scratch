import numpy as np
from bandit import Bandit, Agent
from tqdm import tqdm
import matplotlib.pyplot as plt


class NonStatBandit(Bandit):

    def play(self, action):
        rate = self.rates[action] # rate is win rate
        self.rates += 0.1 * np.random.randn(self.arms) # randomly add noise to make it non-stationary
        self.rates = np.clip(self.rates, 0.0, 1.0)

        if rate > np.random.rand():
            return 1
        return 0
    

class AlphaAgent(Agent):
    def __init__(self, epsilon, alpha, action_size=10):
        super().__init__(epsilon, action_size)
        self.alpha = alpha

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action])  * self.alpha


# write a run function
def run_bandit(runs, steps, make_bandit, make_agent):
    all_rates = np.zeros((runs, steps))
    
    for run in tqdm(range(runs)):
        rates = []
        total_reward = 0
        agent = make_agent()
        bandit = make_bandit()

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)

            total_reward += reward
            rates.append(total_reward / (step+1))

        all_rates[run] = rates

    avg_rates = np.average(all_rates, axis=0)
    return avg_rates

def main():
    np.random.seed(0)
    runs = 200
    steps = 1000
    epsilon = 0.1
    alpha = 0.8

    bandit = Bandit()
    agent = Agent(epsilon)
    nonstat_bandit = NonStatBandit()
    alpha_agent = AlphaAgent(epsilon, alpha)

    avg_rates = run_bandit(runs, steps, make_bandit=lambda: Bandit(), make_agent=lambda:Agent(epsilon))
    nonstat_avg_rates = run_bandit(runs, steps, make_bandit=lambda:NonStatBandit(), make_agent=lambda:AlphaAgent(epsilon, alpha))

    plt.figure()

    plt.ylabel("Average rate")
    plt.xlabel("steps")

    plt.plot(avg_rates, label="static bandit")
    plt.plot(nonstat_avg_rates, label="non-static bandit with decay")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()





    






        
