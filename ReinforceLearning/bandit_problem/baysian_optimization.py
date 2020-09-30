import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)
noise = 0.


def f(x, noise=noise):
    return -np.sin(3 * x) - x ** 2 + 0.7 * x + noise * np.random.randn(*x.shape)


def f(x, noise=noise):
    return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1) + noise * np.random.randn(*x.shape)


bounds = np.array([[-2.0, 10.0]])
X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)

X_init = np.array([[-0.9], [1.1]])
y_init = f(X_init)
X_sample = np.arange(-2., 10., 2).reshape(-1, 1)
y_sample = f(X_sample).flatten()

Y = f(X, 0)


def init_plot():
    plt.plot(X, Y, 'y--', lw=2, label='Noise-free objective')
    plt.plot(X, f(X), 'bx', lw=1, alpha=0.1, label='Noisy samples')
    plt.plot(X_init, y_init, 'kx', mew=3, label='Initial samples')
    plt.legend()


def ucb(X, X_sample, y_sample, gpr):
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    n_sample = X_sample.shape[0]
    mu_sample_opt = np.max(mu_sample)
    z = mu_sample_opt + np.sqrt(np.log(n_sample) / n_sample) * sigma
    # sc = (z - np.mean(z)) / np.std(z)
    return z


def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    """
    Computes the EI at points X based on existing samples X_sample and Y_sample
    using a Gaussian process surrogate model.

    Args: X: Points at which EI shall be computed (m x d).
    X_sample: Sample locations (n x d).
    Y_sample: Sample values (n x 1).
    gpr: A GaussianProcess Regressor fitted to samples.
    xi: Exploitation-exploration trade-off parameter.
    Returns: Expected improvements at points X.
    """
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)
    sigma = sigma.reshape(-1, X_sample.shape[1])

    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def plot_result(X, X_init, y_init, gpr, itr):
    y_mean, y_cov = gpr.predict(X, return_std=True)
    upr = y_mean - np.sqrt(y_cov.reshape(-1, 1))
    lwr = y_mean + np.sqrt(y_cov.reshape(-1, 1))

    plt.clf()
    plt.subplot(211)
    plt.title("Bayesian Optimization # {}".format(i + 1))
    plt.plot(X_init, y_init, "ro", label="Initial samples")
    plt.plot(X, y_mean, label="Surrogate model")
    plt.plot(X, f(X), label="Objective")
    plt.xlim([-2.0, 10.0])
    plt.ylim([-0.5, 1.5])
    plt.fill_between(X.ravel(), upr.ravel(), lwr.ravel(), alpha=0.5)
    plt.legend()
    plt.subplot(212)
    # plt.plot(X, ucb(X, X_init, y_init, gpr), label="UCB")
    plt.plot(X, expected_improvement(X, X_init, y_init, gpr), label="EI")
    plt.xlim([-2.0, 10.0])
    plt.legend()
    plt.savefig("./BayesOptEI{}.png".format(itr))


def next_X(acquisition, X_sample, Y_sample, gpr):
    dim = X_sample.shape[1]

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)

    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(25, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < 1:
            min_val = res.fun[0]
            min_x = res.x
    X_next = min_x.reshape(-1, 1)
    return X_next


X_init = np.array([[-0.9], [1.1]])
y_init = f(X_init)
kernel = ConstantKernel(1.0) + RBF(length_scale=2.0, length_scale_bounds=(0, 10))
gpr = GaussianProcessRegressor(kernel=kernel, random_state=42).fit(X_init, y_init)

forest = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)

X_init = np.linspace(-2, 10, 10).reshape(-1, 1)
y_init = f(X_init)

for i in range(2):
    forest.fit(X_init[i:10], y_init[i:10])
    y_mean = forest.predict(X)
    plt.plot(X_init, y_init, "ro", label="Initial samples")
    plt.plot(X, y_mean, label="Surrogate model")
    plt.plot(X, f(X), label="Objective")

# plot_result(X, X_init, y_init, gpr)

ims = []
for i in range(10):
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=42).fit(X_init, y_init)
    plot_result(X, X_init, y_init, gpr, i)
    X_next = next_X(expected_improvement, X_init, y_init, gpr)
    X_init = np.r_[X_init, X_next]
    y_init = np.r_[y_init, f(X_next)]


def plot_bo(f, bo):
    x = np.linspace(-2, 10, 10000)
    mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)

    plt.figure(figsize=(16, 9))
    plt.plot(x, f(x))
    plt.plot(x, mean)
    plt.fill_between(x, mean + sigma, mean - sigma, alpha=0.1)
    plt.scatter(bo.space.params.flatten(), bo.space.target, c="red", s=50, zorder=10)
    plt.show()


def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    """
     Proposes the next sampling point by optimizing the acquisition function.
    Args: acquisition: Acquisition function.
    X_sample: Sample locations (n x d).
    Y_sample: Sample values (n x 1).
    gpr: A GaussianProcessRegressor fitted to samples.
    Returns: Location of the acquisition function maximum.
    """
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)

    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x.reshape(-1, 1)


def plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next=None, show_legend=False):
    mu, std = gpr.predict(X, return_std=True)
    plt.fill_between(X.ravel(),
                     mu.ravel() + 1.96 * std,
                     mu.ravel() - 1.96 * std,
                     alpha=0.1)
    plt.plot(X, Y, 'y--', lw=1, label='Noise-free objective')
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()


def plot_acquisition(X, Y, X_next, show_legend=False):
    plt.plot(X, Y, 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    if show_legend:
        plt.legend()


def plot_convergence(X_sample, Y_sample, n_init=2):
    plt.figure(figsize=(12, 3))

    x = X_sample[n_init:].ravel()
    y = Y_sample[n_init:].ravel()
    r = range(1, len(x) + 1)

    x_neighbor_dist = [np.abs(a - b) for a, b in zip(x, x[1:])]
    y_max_watermark = np.maximum.accumulate(y)

    plt.subplot(1, 2, 1)
    plt.plot(r[1:], x_neighbor_dist, 'bo-')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Distance between consecutive x\'s')

    plt.subplot(1, 2, 2)
    plt.plot(r, y_max_watermark, 'ro-')
    plt.xlabel('Iteration')
    plt.ylabel('Best Y')
    plt.title('Value of best selected sample')


# Initialize samples
X_sample = X_init
Y_sample = y_init

# Number of iterations
n_iter = 20

plt.figure(figsize=(12, n_iter * 3))
plt.subplots_adjust(hspace=0.4)

for i in range(n_iter):
    # Update Gaussian process with existing samples
    gpr.fit(X_sample, Y_sample)

    # Obtain next sampling point from the acquisition function (expected_improvement)
    X_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds)

    # Obtain next noisy sample from the objective function
    Y_next = f(X_next, noise)

    # Plot samples, surrogate function, noise-free objective and next sampling location
    if (i + 1) % 5 == 0:
        plt.subplot(n_iter, 2, 2 * i + 1)
        plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next, show_legend=i == 0)
        plt.title(f'Iteration {i + 1}')

        plt.subplot(n_iter, 2, 2 * i + 2)
        plot_acquisition(X, expected_improvement(X, X_sample, Y_sample, gpr), X_next, show_legend=i == 0)

    # Add sample to previous samples
    X_sample = np.vstack((X_sample, X_next))
    Y_sample = np.vstack((Y_sample, Y_next))

# plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
#          % (kernel, gpr.kernel_,
#             gpr.log_marginal_likelihood(gpr.kernel_.theta)))
# plt.tight_layout()
# plt.show()


bo = BayesianOptimization(
    f=f,
    pbounds={"x": (-2, 10)},
    verbose=0,
    random_state=987234,
)

# bo.maximize(n_iter=10, acq="ucb", kappa=0.1)
# plot_bo(f, bo)
