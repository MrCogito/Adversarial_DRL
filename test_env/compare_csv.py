import numpy as np
import matplotlib.pyplot as plt

# Function to reshape the loaded flat array back to its original shape
def reshape_observation(flat_observation, shape=(84, 84)):
    return flat_observation.reshape(shape)

# Load the observation values from the CSV files
# Replace 'PettingZoo_observation_values.csv' and 'SB3_observation_values.csv' with your actual file paths
flat_image1 = np.loadtxt('PettingZoo_observation_values.csv', delimiter=',')
flat_image2 = np.loadtxt('SB3_observation_values.csv', delimiter=',')

# Reshape the flat observations back to 84x84
image1 = reshape_observation(flat_image1)
image2 = reshape_observation(flat_image2)

# Print the shapes of the reshaped images
print(f"Shape of Image 1 after reshaping: {image1.shape}")
print(f"Shape of Image 2 after reshaping: {image2.shape}")

# Ensure both images have the same shape
assert image1.shape == image2.shape == (84, 84), "Images must have the same shape."

# Save the first image
plt.imsave('image1.png', image1, cmap='gray')
print("First image saved as 'image1.png'.")

# Save the second image
plt.imsave('image2.png', image2, cmap='gray')
print("Second image saved as 'image2.png'.")

# Calculate the absolute difference between the two images
diff_image = np.abs(image1 - image2)

# Save the difference image
plt.imsave('difference_image.png', diff_image, cmap='gray')
print("Difference image saved as 'difference_image.png'.")
