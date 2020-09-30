
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ReinforceLearning.bandit_program_3.model import BernoulliArm, random_select, EpsilonGreedy
from ReinforceLearning.bandit_program_3.model import UCB, ThompsonSampling, test_algorithm

import random


n_arms = 10
means = [0.054, 0.069, 0.080, 0.097, 0.112,
         0.119, 0.121, 0.144, 0.155, 0.174]

epsilon = 0.2  # パラメータ
sim_num = 500  # シミュレーション回数
time = 10000  # 試行回数

arms = pd.Series(map(lambda x: BernoulliArm(x), means))

algo_1 = random_select([], [])  # random
algo_2 = EpsilonGreedy(epsilon, [], [])  # epsilon-greedy
algo_3 = UCB([], [])  # UCB
algo_4 = ThompsonSampling([], [], [])  # ThompsonSampling
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
heights = []
random.seed(2017)
for algo in [algo_1, algo_2, algo_3, algo_4]:
    algo.initialize(n_arms)
    result = test_algorithm(algo, arms, sim_num, time)

    df_result = pd.DataFrame({"times": result[0], "chosen_arms": result[1]})
    df_result["best_arms"] = (df_result["chosen_arms"] == np.argmax(means)).astype(int)
    grouped = df_result["best_arms"].groupby(df_result["times"])

    ax1.plot(grouped.mean(), label=algo.__class__.__name__)
    heights.append(result[2][-1])

ax1.set_title("Compare 4model - Best Arm Rate")
ax1.set_xlabel("Time")
ax1.set_ylabel("Best Arm Rate")
ax1.legend(loc="upper left")

plt_label = ["Random", "Epsilon\nGreedy", "UCB", "Tompson \nSampling"]
plt_color = ["deep", "muted", "pastel", "bright"]
ax2.bar(range(1, 5), heights, color=sns.color_palette()[:4], align="center")
ax2.set_xticks(range(1, 5))
ax2.set_xticklabels(plt_label)
ax2.set_label("random_select")
ax2.set_ylabel("Cumulative Rewards")
ax2.set_title("Compare 4model - Cumulative Rewards")
plt.show()