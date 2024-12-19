import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the folder paths containing the CSV files
folders = [
    "/home/vs/wtftp-model/log/24-12-19_no_att_1_len_1",
    "/home/vs/wtftp-model/log/24-12-19_no_att_2_len_1",
    "/home/vs/wtftp-model/log/24-12-19_no_att_3_len_1",
    "/home/vs/wtftp-model/log/24-12-19_att_1_len_1",
    "/home/vs/wtftp-model/log/24-12-19_att_2_len_1",
    "/home/vs/wtftp-model/log/24-12-19_att_3_len_1",
]

# Define labels for the models
model_labels = ["No attention, level=1", "No attention, level=2", "No attention, level=3", "Wavelet attention, level=1", "Wavelet attention, level=2", "Wavelet attention, level=3"]

colors = ["darkred", "salmon", "crimson", "navy", "royalblue", "skyblue"]
markers = ["x","x","x","o","o","o"]
# Initialize a list to store DataFrames
dataframes = []

# Read the CSV files from each folder
for folder in folders:
    csv_file = os.path.join(folder, "output_data.csv")
    if os.path.exists(csv_file):
        dataframes.append(pd.read_csv(csv_file))
    else:
        print(f"Warning: File not found {csv_file}")

# Check if all dataframes are loaded
if len(dataframes) < len(folders):
    print("Some files are missing. Please check the folder paths and files.")

# Create the first plot: ave_train_loss
plt.figure(figsize=(10, 6))

for df, label, color, marker in zip(dataframes, model_labels, colors[:len(dataframes)], markers[:len(dataframes)]):
    plt.plot(df.index, df["ave_train_loss"], marker=marker, markersize=4, color=color, label=f"{label}")

plt.title("Average Training Loss of Models")
plt.xlabel("Epoch")
plt.ylabel("Average Training Loss")
plt.legend()
plt.grid(True)
# plt.show()

# Create the second plot: mean_rmse_position and mean_rmse_velocity
plt.figure(figsize=(10, 6))
for df, label, color, marker in zip(dataframes, model_labels, colors[:len(dataframes)], markers[:len(dataframes)]):
    plt.plot(df.index, df["mean_rmse_position"], marker=marker, markersize=4, color=color, label=f"{label}")

plt.title("Mean RMSE of Position")
plt.xlabel("Epoch")
plt.ylabel("Mean RMSE")
plt.legend()
plt.grid(True)
# plt.show()


plt.figure(figsize=(10, 6))
for df, label, color, marker in zip(dataframes, model_labels, colors[:len(dataframes)], markers[:len(dataframes)]):
    plt.plot(df.index, df["mean_rmse_velocity"], marker=marker, markersize=4, color=color, label=f"{label}")

plt.title("Mean RMSE of Velocity")
plt.xlabel("Epoch")
plt.ylabel("Mean RMSE")
plt.legend()
plt.grid(True)
# plt.show()




# ERROR COMPARE
# Define the folder paths containing the CSV files
folders = [
    "/home/vs/wtftp-model/log/24-12-19_att_1_len_1",
    "/home/vs/wtftp-model/log/24-12-19_att_2_len_1",
    "/home/vs/wtftp-model/log/24-12-19_att_3_len_1",
]

# Define labels for the models
model_labels = ["Wavelet attention, level=1", "Wavelet attention, level=2", "Wavelet attention, level=3"]

colors = ["navy", "royalblue", "skyblue"]
markers = ["o","o","o"]


dataframes_err = []
for folder in folders:
    csv_file = os.path.join(folder, "spline.csv")
    if os.path.exists(csv_file):
        dataframes_err.append(pd.read_csv(csv_file))
    else:
        print(f"Warning: File not found {csv_file}")

# Check if all dataframes are loaded
if len(dataframes_err) < len(folders):
    print("Some files are missing. Please check the folder paths and files.")

print(dataframes_err)
koef = 1.0
alphas = [koef, koef, koef]

plt.figure(figsize=(10, 6))
for df, label, color, marker, alpha in zip(dataframes_err, model_labels, colors[:len(dataframes_err)], markers[:len(dataframes_err)], alphas[:len(dataframes_err)]):
    plt.plot(df.index, df["pose_errors"], marker=marker, markersize=4, label=f"{label}", alpha=alpha)

plt.title("Pose Errors (Euclidean Distance) between Target and Predicted.")
plt.xlabel("Epoch")
plt.ylabel("Pose error, m")
plt.legend()
plt.grid(True)
plt.show()
