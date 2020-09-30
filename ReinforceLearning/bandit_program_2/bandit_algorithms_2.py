import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# ------------------------------------------------------------------------------
# DEFINITION
# ------------------------------------------------------------------------------
# BernoulliArm()クラスの定義
class BernoulliArm():
    def __init__(self, p):
        self.p = p

    def draw(self):
        if self.p > random.random():
            return 1.0
        else:
            return 0.0


# EpsilonGreedy()クラスの定義
class EpsilonGreedy():
    def __init__(self, epsilon, counts, values):
        self.epsilon = epsilon  # Probability of searching
        self.counts = counts    # Number of times to pull the arm
        self.values = values    # The average value of the reward obtained from the drawn arm
        return

    # initialize counts and values
    def initialize(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        return

    def select_arm(self):
        if self.epsilon > random.random():
            # Search with probability ε.
            return np.random.randint(0, len(self.values))
        else:
            # Use with probability 1-ε.
            return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        # Update the number of times you selected an arm.
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return


# Test_algorithm() method to run simulation tests
def test_algorithm(algo, arms, num_sims, horizon):
    # Variable initialization
    chosen_arms = np.zeros(num_sims * horizon)
    rewards = np.zeros(num_sims * horizon)
    cumulative_rewards = np.zeros(num_sims * horizon)
    sim_nums = np.zeros(num_sims * horizon)
    times = np.zeros(num_sims * horizon)

    # Double for loop
    for sim in range(num_sims):
        # Count the number of simulations.
        sim = sim + 1

        # Initialize the algorithm settings.
        algo.initialize(len(arms))

        for t in range(horizon):
            # Count the number of rounds.
            t = t + 1

            # Substitute the current count (simulation count x round count) into index.
            index = (sim - 1) * horizon + t - 1

            # Substitute sim for the index th element of sim_nums.
            sim_nums[index] = sim

            # Substitute t for the index th element of times.
            times[index] = t

            # The arm selected by the select_arm() method is assigned to chosen_arm.
            chosen_arm = algo.select_arm()

            # The arm selected by the select_arm() method is assigned to chosen_arm.
            chosen_arms[index] = chosen_arm

            reward = arms[chosen_arm].draw()

            # Assign reward to the index th element of rewards.
            rewards[index] = reward

            if t == 1:
                # Assign reward to the index th element of cumulative_rewards.
                cumulative_rewards[index] = reward
            else:
                cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

            algo.update(chosen_arm, reward)

    return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # Probability of success with reward from arm
    theta = np.array([0.1, 0.1, 0.1, 0.1, 0.9])
    n_arms = len(theta)
    random.shuffle(theta)

    arms = map(lambda x: BernoulliArm(x), theta)
    arms = list(arms)

    for epsilon in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        algo = EpsilonGreedy(epsilon, [], [])
        algo.initialize(n_arms)

        # Store the simulation execution result in results.
        results = test_algorithm(algo, arms, 5000, 250)

        df = pd.DataFrame({"times": results[1], "rewards": results[3]})
        grouped = df["rewards"].groupby(df["times"])

        plt.plot(grouped.mean(), label="epsilon=" + str(epsilon))

    plt.legend(loc="best")
    plt.show()