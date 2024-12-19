import matplotlib.pyplot as plt

# Data: MDE values for each step
steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mde_values = [
    0.03468410, 0.03508964, 0.03631274, 0.03850091, 0.04175429,
    0.04602943, 0.05119750, 0.05715946, 0.06378765, 0.07100121
]
# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(steps, mde_values, marker='o', color='b', linestyle='-', linewidth=2, markersize=6)

# Add labels and title
plt.title("MDE (Mean Distance Error) vs Step", fontsize=14)
plt.xlabel("Prediction Step", fontsize=12)
plt.ylabel("MDE (unscaled)", fontsize=12)

# Show grid for better readability
plt.xticks(range(1, 11))  # Ensures x-axis has integer steps from 1 to 10
plt.grid(True)

# Display the plot
plt.show()
