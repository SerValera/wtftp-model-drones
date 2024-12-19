import re
import matplotlib.pyplot as plt
import csv
import numpy as np

# Function to parse the log file
def parse_log_file(log_file_path):
    # Lists to store the extracted data
    ave_train_loss = []
    ave_temporal_loss = []
    ave_freq_loss = []
    rmse_values = {"lon": [], "lat": [], "alt": [], "spdx": [], "spdy": [], "spdz": []}
    mae_values = {"lon": [], "lat": [], "alt": [], "spdx": [], "spdy": [], "spdz": []}
    mse_values = []  # To store aveMSE(scaled) values
    mae_scaled_values = []  # To store aveMAE(scaled) values
    
    with open(log_file_path, 'r') as file:
        epoch_data = {}
        for line in file:
            # Match the epoch loss lines
            if "ave_train_loss" in line:
                # Extract ave_train_loss, ave_temporal_loss, ave_freq_loss
                match = re.search(r'ave_train_loss: (\S+), ave_temporal_loss: (\S+), ave_freq_loss: (\S+)', line)
                if match:
                    ave_train_loss.append(float(match.group(1)))
                    ave_temporal_loss.append(float(match.group(2)))
                    ave_freq_loss.append(float(match.group(3)))
            
            # Match RMSE and MAE values in the evaluation stage
            elif "Evaluation-Stage:" in line:
                # Extract RMSE values
                match_rmse = re.search(r'in each attr\(RMSE, unscaled\): {([^}]+)}', file.readline())
                if match_rmse:
                    rmse_data = match_rmse.group(1)
                    # Clean up commas and convert to floats
                    rmse_vals = {param: float(value.strip(',')) for param, value in re.findall(r"'(\w+)': (\S+)", rmse_data)}
                    for param in rmse_values:
                        rmse_values[param].append(rmse_vals.get(param, None))
                
                # Extract MAE values
                match_mae = re.search(r'in each attr\(MAE, unscaled\): {([^}]+)}', file.readline())
                if match_mae:
                    mae_data = match_mae.group(1)
                    mae_vals = {param: float(value.strip(',')) for param, value in re.findall(r"'(\w+)': (\S+)", mae_data)}
                    for param in mae_values:
                        mae_values[param].append(mae_vals.get(param, None))

                # Extract aveMSE(scaled) value
                match_mse = re.search(r'aveMSE\(scaled\): (\S+)', line)
                if match_mse:
                    mse_values.append(float(match_mse.group(1)))

                # Extract aveMAE(scaled) value
                match_mae_scaled = re.search(r'aveMAE\(scaled\): (\S+)', line)
                if match_mae_scaled:
                    mae_scaled_values.append(float(match_mae_scaled.group(1)))

    return ave_train_loss, ave_temporal_loss, ave_freq_loss, rmse_values, mae_values, mse_values, mae_scaled_values

# Function to plot the results
def plot_results(ave_train_loss, ave_temporal_loss, ave_freq_loss, rmse_values, mae_values, mse_values, mae_scaled_values):
    epochs = list(range(1, len(ave_train_loss) + 1))

    # Plot loss values
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, ave_train_loss, label='Train Loss', marker='o')
    # plt.plot(epochs, ave_temporal_loss, label='Temporal Loss', marker='x')
    # plt.plot(epochs, ave_freq_loss, label='Frequency Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot RMSE values for each parameter
    plt.figure(figsize=(10, 6))
    for param in rmse_values:
        plt.plot(epochs, rmse_values[param], label=f'RMSE ({param})', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot MAE values for each parameter
    plt.figure(figsize=(10, 6))
    for param in mae_values:
        plt.plot(epochs, mae_values[param], label=f'MAE ({param})', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot aveMSE(scaled) values
    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs, mse_values, label='aveMSE(scaled)', marker='o', color='b')
    # plt.xlabel('Epoch')
    # plt.ylabel('aveMSE(scaled)')
    # plt.title('MSE vs Epochs')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # Plot aveMAE(scaled) values
    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs, mae_scaled_values, label='aveMAE(scaled)', marker='x', color='r')
    # plt.xlabel('Epoch')
    # plt.ylabel('aveMAE(scaled)')
    # plt.title('MAE(scaled) vs Epochs')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

def save_to_csv(dir, ave_train_loss, ave_temporal_loss, ave_freq_loss, rmse_values, mae_values, mse_values, mae_scaled_values):

    # Calculate mean RMSE for position
    mean_rmse_position = []
    for i in range(len(rmse_values['lon'])):
        mean_rmse_position.append(np.sqrt((rmse_values['lon'][i]**2 + rmse_values['lat'][i]**2 + rmse_values['alt'][i]**2) / 3))

    # Calculate mean RMSE for velocity
    mean_rmse_velocity = []
    for i in range(len(rmse_values['spdx'])):
        mean_rmse_velocity.append(np.sqrt((rmse_values['spdx'][i]**2 + rmse_values['spdy'][i]**2 + rmse_values['spdz'][i]**2) / 3))

    print(mae_scaled_values)

    # Combine the lists into rows
    data = zip(ave_train_loss, ave_temporal_loss, ave_freq_loss, mean_rmse_position, mean_rmse_velocity)
    csv_file_name = dir + "output_data.csv"

    # Write data to CSV
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['ave_train_loss', 'ave_temporal_loss', 'ave_freq_loss', 'mean_rmse_position', 'mean_rmse_velocity'])
        # Write data rows
        writer.writerows(data)

    print(f"Data has been written to {csv_file_name}")

# Main function to parse the log and plot results
def main():

    log_file_path = '/home/vs/wtftp-model/log/24-12-19_no_att_1_len_1/train.log'  # no att 1
    save_path_data = '/home/vs/wtftp-model/log/24-12-19_no_att_1_len_1/'

    log_file_path = '/home/vs/wtftp-model/log/24-12-19_no_att_2_len_1/train.log'  # no att 2
    save_path_data = '/home/vs/wtftp-model/log/24-12-19_no_att_2_len_1/'

    log_file_path = '/home/vs/wtftp-model/log/24-12-19_no_att_3_len_1/train.log'  # no att 3
    save_path_data = '/home/vs/wtftp-model/log/24-12-19_no_att_3_len_1/'

    # log_file_path = '/home/vs/wtftp-model/log/24-12-19_att_1_len_1/train.log'  # att 1
    # save_path_data = '/home/vs/wtftp-model/log/24-12-19_att_1_len_1/'

    # log_file_path = '/home/vs/wtftp-model/log/24-12-19_att_2_len_1/train.log'  # att 2
    # save_path_data = '/home/vs/wtftp-model/log/24-12-19_att_2_len_1/'

    # log_file_path = '/home/vs/wtftp-model/log/24-12-19_att_3_len_1/train.log'  # att 3
    # save_path_data = '/home/vs/wtftp-model/log/24-12-19_att_3_len_1/'

    


    # Parse the log file
    ave_train_loss, ave_temporal_loss, ave_freq_loss, rmse_values, mae_values, mse_values, mae_scaled_values = parse_log_file(log_file_path)

    # Save traning data to csv
    save_to_csv(save_path_data, ave_train_loss, ave_temporal_loss, ave_freq_loss, rmse_values, mae_values, mse_values, mae_scaled_values)

    # Plot the results
    plot_results(ave_train_loss, ave_temporal_loss, ave_freq_loss, rmse_values, mae_values, mse_values, mae_scaled_values)

# Run the main function
if __name__ == '__main__':
    main()
