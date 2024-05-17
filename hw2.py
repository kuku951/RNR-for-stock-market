import numpy as np
import math

# Parameters
initial_probs = np.array([0.5, 0.2, 0.3])
transition_probs = np.array([
    [0.6, 0.2, 0.2],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])
emission_probs = np.array([
    [0.7, 0.1, 0.3],
    [0.1, 0.6, 0.4],
    [0.2, 0.3, 0.3]
])

# Map observations
obs_map = {'up': 0, 'unchanged': 1, 'down': 2}
obs = [obs_map[o] for o in ['up', 'up', 'unchanged', 'down', 'unchanged', 'down', 'up']]

# Forward algorithm
def forward_algorithm(obs):
    T = len(obs)
    N = initial_probs.shape[0]
    alpha = np.zeros((T, N))
    alpha[0] = initial_probs * emission_probs[:, obs[0]]
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t - 1] * transition_probs[:, j] * emission_probs[j, obs[t]])
    return alpha

alpha = forward_algorithm(obs)

numbers=[-0.0152,-0.0885]
incremented_numbers = [num + 1 for num in numbers]
product = math.prod(incremented_numbers)
geometric_mean=product ** (1/len(numbers))
print("nn",geometric_mean)

print("Alpha values:")
print(alpha)
print("\nProbability of observation sequence:", np.sum(alpha[-1]))