import random
import numpy as np


class BernoulliArm:
    def __init__(self, p):
        self.p = p

    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0


class ActionSelector:
    def __init__(self, counts, values):
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        raise NotImplementedError

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value


class EpsilonGreedy(ActionSelector):
    def __init__(self, counts, values, epsilon):
        super(EpsilonGreedy, self).__init__(counts, values)
        self.epsilon = epsilon

    def select_arm(self):
        if random.random() > self.epsilon:
            return np.argmax(self.values)
        else:
            return random.randrange(len(self.values))


class UCB1(ActionSelector):
    def __init__(self, counts, values):
        super(UCB1, self).__init__(counts, values)

    def select_arm(self):
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm

        total_counts = sum(self.counts)
        bonus = np.sqrt((2 * np.log(np.array(total_counts))) / np.array(self.counts))
        ucb_values = np.array(self.values) + bonus
        return np.argmax(ucb_values)


class ThompsonSampling(ActionSelector):
    def __init__(self, counts, values):
        super(ThompsonSampling, self).__init__(counts, values)
        self.alpha = None
        self.beta = None

    def initialize(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self):
        return self.beta_sampling(self.alpha, self.beta)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value

        if new_value > 0:
            self.alpha[chosen_arm] += 1
        else:
            self.beta[chosen_arm] += 1

    @staticmethod
    def beta_sampling(alpha, beta):
        samples = [np.random.beta(alpha[i] + 1, beta[i] + 1) for i in range(len(alpha))]
        return np.argmax(samples)


def categorical_draw(probs):
    z = random.random()
    cum_prob = 0.0
    for i in range(len(probs)):
        prob = probs[i]
        cum_prob += prob
        if cum_prob > z:
            return i

    return len(probs) - 1


class PolicyGradient(ActionSelector):
    def __init__(self, counts, values, n_arms):
        super(PolicyGradient, self).__init__(counts, values)
        self.n_arms = n_arms

        self.total_values = [0 for _ in range(self.n_arms)]
        self.total_counts = 1
        self.episodes = 0

        # soft max parameter
        self.theta = np.ones(self.n_arms) + np.random.normal(0, 0.01, self.n_arms)

        # policy gradient
        self.grad_reward = 0
        self.mean_grad_reward = 0

        # hyper parameter
        self.beta = 1.0
        self.eta = 0.10

    def initialize(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

        # M : episodes T : total step
        # (1/M)∑(1/T)∑[∇log(π(a))* rewards]
        self.total_counts = np.sum(self.counts)
        if self.total_counts == 0:
            self.total_counts = 1
        self.mean_grad_reward = self.grad_reward / self.total_counts
        self.episodes += 1

        # update theta
        if self.episodes % 10 == 0:
            self.theta = self.reinforce(self.theta)
            th = self.soft_max(self.theta)
            print(f"EPISODE {self.episodes} : Update Policy Probability {th}")
        # print(f"(1/T)∑∇logπ(a)R = {self.mean_grad_reward}\n"))

    def select_arm(self):
        # t = sum(self.counts) + 1
        # beta= 1 / np.log(t + 0.0000001)
        beta = 1.0
        logits = beta * self.theta
        probs = self.soft_max(logits)
        return categorical_draw(probs)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        self.grad_reward += self.grad_ln_pi().dot(self.values)  # new_value

    def reinforce(self, theta):
        # REINFORCE
        nabla_J = self.mean_grad_reward / self.episodes
        new_theta = theta + self.eta * nabla_J
        self.mean_grad_reward += self.mean_grad_reward
        return new_theta

    @staticmethod
    def soft_max(logits):
        return np.nan_to_num([np.exp(var) / np.nansum(np.exp(logits)) for var in logits])

    def grad_soft_max(self):
        pi = self.soft_max(self.theta)
        dpi = [i * (1 - i) if i == j else -i * j for i in pi for j in pi]
        grad_pi = np.array(dpi).reshape(len(pi), -1)
        return grad_pi

    def grad_ln_pi(self):
        pi = self.soft_max(self.theta)
        dlog_pi = [(1 - i) if i == j else -j for i in pi for j in pi]
        dlog_pi = np.array(dlog_pi).reshape(len(pi), -1)
        return dlog_pi
