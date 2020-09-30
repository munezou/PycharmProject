import numpy as np
import random


class BernoulliArm():

    def __init__(self, p):
        self.p = p

    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0


class random_select():
    """
    this method is to explore only.
    """

    def __init__(self, counts, values):
        """
        Declare variables
        :param counts: list of Selected times each arm
        :param values: List of value variables each arm
        """
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        """
        Prepare an array of the number of selectors and set its value to 0.
        :param n_arms: a number of selected counts
        :return: void
        """
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        """
        Select selectors at random.
        :return: the index of the selector.
        """
        return random.randint(0, len(self.values) - 1)

    def update(self, chosen_arm, reward):
        """
        Update the information on the selected arm.
        :param chosen_arm: index of the selected arm
        :param reward: reward of the selected arm
        :return:
        """
        # Increase the selected number of specified arm by one.
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1

        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]

        # Add the current reward to the past amount of value and average it out.
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward

        # Update values
        self.values[chosen_arm] = new_value


class EpsilonGreedy():
    """
    To explore at epsilon ratio.
    And the rest, to learn.
    """

    def __init__(self, epsilon, counts, values):
        """
        Declare variables
        :param epsilon: The ratio of exploring to teaching.
        :param counts: list of Selected times each arm
        :param values: List of value variables each arm
        """
        self.epsilon = epsilon
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        """
        Prepare an array of the number of selectors and set its value to 0.
        :param n_arms: a number of selected counts
        :return: void
        """
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        if random.random() > self.epsilon:
            """
            Teaching)
              Return the arm indices with the highest amount of value in the past.
            """
            return np.argmax(self.values)
        else:
            """
            search)
              Selects an arm at random and returns its index.
            """
            return random.randint(0, len(self.values) - 1)

    def update(self, chosen_arm, reward):
        """
        Update the information on the selected arm.
        :param chosen_arm: index of the selected arm
        :param reward: reward of the selected arm
        :return:
        """
        # Increase the selected number of specified arm by one.
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1

        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]

        # Add the current reward to the past amount of value and average it out.
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward

        # Update values
        self.values[chosen_arm] = new_value


class UCB():
    """
    UCB (Upper Confidence Bound) is an algorithm that selects the arm
    that has the largest upper confidence interval of the expected reward estimated
    by the previously observed reward at a given time.
    """
    def __init__(self, counts, values):
        """
        Declare variables
        :param counts: list of Selected times each arm
        :param values: List of value variables each arm
        """
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        """
        Prepare an array of the number of selectors and set its value to 0.
        :param n_arms: a number of selected counts
        :return: void
        """
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        """
        How to select
        :return:
        """
        # Set the number of arms of choice.
        n_arms = len(self.counts)

        # Processing when there is an arm that is never selected
        if min(self.counts) == 0:
            return np.argmin(self.counts)

        # Calculate the total number of selections.
        total_counts = sum(self.counts)

        """
        The ratio of the number of times each arm is selected 
        to the overall number of selections is used to correct for value.（Upper Confidence Bound）
        """
        bonus = np.sqrt((np.log(np.array(total_counts))) / (2 * np.array(self.counts)))

        # Calculating Value with UCB
        ucb_values = np.array(self.values) + bonus

        # Return the index of the arm with the highest value.
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        """
                Update the information on the selected arm.
                :param chosen_arm: index of the selected arm
                :param reward: reward of the selected arm
                :return:
                """
        # Increase the selected number of specified arm by one.
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1

        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]

        # Add the current reward to the past amount of value and average it out.
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward

        # Update values
        self.values[chosen_arm] = new_value


class ThompsonSampling():
    """
     There is a methodology called the probability matching method,
    which focuses on the "probability that the arm is the best arm".
     Thompson Sampling is a strategy that applies a Bayesian statistical framework
    to this probability matching method.
    """

    def __init__(self, counts_alpha, counts_beta, values):
        """
        Declare variables
        :param counts_alpha:
        :param counts_beta:
        :param values:
        """
        self.counts_alpha = counts_alpha
        self.counts_beta = counts_beta
        self.alpha = 1
        self.beta = 1
        self.values = values

    def initialize(self, n_arms):
        """

        :param n_arms:
        :return:
        """
        self.counts_alpha = np.zeros(n_arms)
        self.counts_beta = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        """

        :return:
        """
        theta = [
                    (
                        arm,
                        random.betavariate(self.counts_alpha[arm] + self.alpha,
                        self.counts_beta[arm] + self.beta)
                    ) for arm in range(len(self.counts_alpha))
                ]

        theta = sorted(theta, key=lambda x: x[1])

        return theta[-1][0]

    def update(self, chosen_arm, reward):
        if reward == 1:
            self.counts_alpha[chosen_arm] += 1
        else:
            self.counts_beta[chosen_arm] += 1

        n = float(self.counts_alpha[chosen_arm]) + self.counts_beta[chosen_arm]

        self.values[chosen_arm] = (n - 1) / n * \
                                  self.values[chosen_arm] + 1 / n * reward


def test_algorithm(algo, arms, num_sims, horizon):
    chosen_arms = np.zeros(num_sims * horizon)
    cumulative_rewards = np.zeros(num_sims * horizon)
    times = np.zeros(num_sims * horizon)
    for sim in range(num_sims):
        algo.initialize(len(arms))
        for t in range(horizon):
            index = sim * horizon + t
            times[index] = t + 1
            chosen_arm = algo.select_arm()
            chosen_arms[index] = chosen_arm
            reward = arms[chosen_arm].draw()
            if t == 0:
                cumulative_rewards[index] = reward
            else:
                cumulative_rewards[index] = cumulative_rewards[
                                                index - 1] + reward
            algo.update(chosen_arm, reward)
    return [times, chosen_arms, cumulative_rewards]
