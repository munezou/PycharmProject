import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ReinforceLearning.bandit_problem.utils import EpsilonGreedy, UCB1, ThompsonSampling
from ReinforceLearning.bandit_problem.utils import BernoulliArm


def test_algorithm(algo, arms, num_sims, horizon):
    chosen_arms = np.zeros(num_sims * horizon)
    rewards = np.zeros(num_sims * horizon)
    cumulative_rewards = np.zeros(num_sims * horizon)
    sim_nums = np.zeros(num_sims * horizon)
    times = np.zeros(num_sims * horizon)

    for sim in tqdm(range(num_sims)):
        sim = sim + 1
        algo.initialize(len(arms))

        for t in range(horizon):
            t = t + 1
            index = (sim - 1) * horizon + t - 1
            sim_nums[index] = sim
            times[index] = t
            chosen_arm = algo.select_arm()
            chosen_arms[index] = chosen_arm

            reward = arms[chosen_arm].draw()
            rewards[index] = reward

            if t == 1:
                cumulative_rewards[index] = reward
            else:
                cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

            algo.update(chosen_arm, reward)

    return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]


def run(algo, label):
    print(label)
    algo.initialize(n_arms)
    results = test_algorithm(algo, arms, num_sims=NUM_SIMS, horizon=HORIZON)

    df = pd.DataFrame({"times": results[1], "rewards": results[3]})
    grouped = df["rewards"].groupby(df["times"])
    plt.plot(grouped.mean(), label=label)
    plt.legend(loc="best")


if __name__ == '__main__':
    NUM_SIMS = 1000
    HORIZON = 200

    # 問題設定: 腕：7本のうち、あたりは1つ (0.8)とする。
    theta = np.array([0.1, 0.4, 0.1, 0.2, 0.8, 0.1, 0.1])
    n_arms = len(theta)
    # random.shuffle(theta)
    print(theta)

    arms = map(lambda x: BernoulliArm(x), theta)
    arms = list(arms)

    algos = {'Eps': EpsilonGreedy([], [], epsilon=0.01), 'UCB': UCB1([], []), 'TS': ThompsonSampling([], [])}
    for key, algo in algos.items():
        run(algo, label=key)

    plt.show()
