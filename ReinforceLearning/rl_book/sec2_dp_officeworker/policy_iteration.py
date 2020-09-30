"""
overview:
    We solve the three-state,
    two-action Markov decision process by using policy iterative methods.

args:
    The setting values for each parameter are specified in the code.

output:
    For each iterative step, it outputs the three-state value function value and policy probability.

usage-example:
    python3 policy_iteraton.py
"""
# Importing various modules
import numpy as np
import copy

# Configuring MDP
p = [0.8, 0.5, 1.0]

# Setting the discount rate
gamma = 0.95

# Setting the Expected Reward Value
"""
--------------------------------------------
r[s, s', a]
s: current state
s': next state
a: action

prameter s, s':
0: home
1: office
2: bar

prament a:
0: move
1: stay
--------------------------------------------
"""
r = np.zeros((3, 3, 2))
r[0, 1, 0] = 1.0
r[0, 2, 0] = 2.0
r[0, 0, 1] = 0.0
r[1, 0, 0] = 1.0
r[1, 2, 0] = 2.0
r[1, 1, 1] = 1.0
r[2, 0, 0] = 1.0
r[2, 1, 0] = 0.0
r[2, 2, 1] = -1.0

# Initialization of the value function
v = [0, 0, 0]
v_prev = copy.copy(v)

# Initialization of the action value function
q = np.zeros((3, 2))

# Initialization of policy distribution
pi = [0.5, 0.5, 0.5]


# Definition of a policy evaluation function
def policy_estimator(pi, p, r, gamma):
    # Initialize
    R = [0, 0, 0]
    P = np.zeros((3, 3))
    A = np.zeros((3, 3))

    for i in range(3):

        # Calculation of the state transition matrix
        P[i, i] = 1 - pi[i]
        P[i, (i + 1) % 3] = p[i] * pi[i]
        P[i, (i + 2) % 3] = (1 - p[i]) * pi[i]

        # Calculating the Reward Vector
        R[i] = pi[i] * (p[i] * r[i, (i + 1) % 3, 0] + (1 - p[i]) * r[i, (i + 2) % 3, 0]) \
               + (1 - pi[i]) * r[i, i, 1]

    # Solving Bellman's equation by matrix calculation
    A = np.eye(3) - gamma * P
    B = np.linalg.inv(A)
    v_sol = np.dot(B, R)

    return v_sol


# Policy Iterative Calculation
for step in range(100):

    # policy evaluation step
    v = policy_estimator(pi, p, r, gamma)

    # If the value function v does not improve the value v_prep of the previous step, it ends.
    if np.min(v - v_prev) <= 0:
        break

    # Display the value function and measures of the current step.
    print('step:', step, ' value:', v, ' policy:', pi)

    # policy improvement steps
    for i in range(3):

        # Calculate the action value function.
        q[i, 0] = p[i] * (r[i, (i + 1) % 3, 0] + gamma * v[(i + 1) % 3]) + \
                  (1 - p[i]) * (r[i, (i + 2) % 3, 0] + gamma * v[(i + 2) % 3])
        q[i, 1] = r[i, i, 1] + gamma * v[i]

        # Improving measures greedy under the behavioral value function.
        if q[i, 0] > q[i, 1]:
            pi[i] = 1
        elif q[i, 0] == q[i, 1]:
            pi[i] = 0.5
        else:
            pi[i] = 0

    # Record the value function of the current step.
    v_prev = copy.copy(v)
