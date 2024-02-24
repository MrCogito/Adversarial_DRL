import numpy as np

# Function to load observations from CSV files
def load_obs_from_csv(filename):
    return np.loadtxt(filename, delimiter=",").reshape((84, 84, 4))  # Reshape to original observation shape

# Load observations
sb3_obs = load_obs_from_csv("SB3_observation_values.csv")
pettingzoo_obs = load_obs_from_csv("PettingZoo_observation_values.csv")

# Compute absolute differences
differences = np.abs(sb3_obs - pettingzoo_obs)

# Summarize differences
max_difference = np.max(differences)
mean_difference = np.mean(differences)
elements_different = np.sum(differences > 0)  # Count elements with any difference

print(f"Max difference: {max_difference}")
print(f"Mean difference: {mean_difference}")
print(f"Total elements with difference: {elements_different} out of {differences.size}")
