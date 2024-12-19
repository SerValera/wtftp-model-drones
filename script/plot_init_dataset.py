import matplotlib.pyplot as plt
import numpy as np

# Read the data from the text file
data = np.loadtxt('/home/vs/wtftp-model/data_set_for_test/test_4/test/dataset_spline_E.txt')

# Extract the x, y, z coordinates
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Create the plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory in 3D
ax.plot(x, y, z, label='Trajectory', color='b')

# Get the range for each axis
x_range = np.max(x) - np.min(x)
y_range = np.max(y) - np.min(y)
z_range = np.max(z) - np.min(z)

# Find the max range to make the axes equal
max_range = max(x_range, y_range, z_range)

# Set the limits for each axis to ensure they are equal
mid_x = (np.max(x) + np.min(x)) / 2
mid_y = (np.max(y) + np.min(y)) / 2
mid_z = (np.max(z) + np.min(z)) / 2

ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Trajectory')

# Show the legend
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
