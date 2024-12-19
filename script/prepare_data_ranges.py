import numpy as np

# Function to parse the txt file and get min and max values for each parameter
def get_min_max_values(filenames):
    min_max_values = {
        "lon": {"min": float('inf'), "max": float('-inf')},
        "lat": {"min": float('inf'), "max": float('-inf')},
        "alt": {"min": float('inf'), "max": float('-inf')},
        "spdx": {"min": float('inf'), "max": float('-inf')},
        "spdy": {"min": float('inf'), "max": float('-inf')},
        "spdz": {"min": float('inf'), "max": float('-inf')}
    }

    # Loop over all provided filenames
    for filename in filenames:
        with open(filename, 'r') as file:
            for line in file:
                # Split the line into values
                values = line.split()

                # Convert values to float
                lon, lat, alt, spdx, spdy, spdz = map(float, values)

                # Update min and max values for each parameter
                min_max_values["lon"]["min"] = min(min_max_values["lon"]["min"], lon)
                min_max_values["lon"]["max"] = max(min_max_values["lon"]["max"], lon)
                min_max_values["lat"]["min"] = min(min_max_values["lat"]["min"], lat)
                min_max_values["lat"]["max"] = max(min_max_values["lat"]["max"], lat)
                min_max_values["alt"]["min"] = min(min_max_values["alt"]["min"], alt)
                min_max_values["alt"]["max"] = max(min_max_values["alt"]["max"], alt)
                min_max_values["spdx"]["min"] = min(min_max_values["spdx"]["min"], spdx)
                min_max_values["spdx"]["max"] = max(min_max_values["spdx"]["max"], spdx)
                min_max_values["spdy"]["min"] = min(min_max_values["spdy"]["min"], spdy)
                min_max_values["spdy"]["max"] = max(min_max_values["spdy"]["max"], spdy)
                min_max_values["spdz"]["min"] = min(min_max_values["spdz"]["min"], spdz)
                min_max_values["spdz"]["max"] = max(min_max_values["spdz"]["max"], spdz)

    return min_max_values

# Function to save the dictionary to a .npy file
def save_dict_to_npy(data, file_path):
    np.save(file_path, data, allow_pickle=True)

def main():
    # List of data files to analyze
    filenames = [
        '/home/vs/wtftp-model/data/dev/data_cyrcle.txt',
        '/home/vs/wtftp-model/data/dev/data_spline8.txt',
        '/home/vs/wtftp-model/data/dev/data_square.txt',
        '/home/vs/wtftp-model/data/test/data_spline8.txt',
        '/home/vs/wtftp-model/data/test/data_square_2.txt',
        '/home/vs/wtftp-model/data/test/data.txt',        
        '/home/vs/wtftp-model/data/train/data_cyrcle_long.txt',
        '/home/vs/wtftp-model/data/train/data_cyrcle_short.txt',
        '/home/vs/wtftp-model/data/train/data_spline8.txt',
        '/home/vs/wtftp-model/data/train/dataset_square_counter.txt'
    ]
    
    # Get the min/max values from all the files
    min_max_values = get_min_max_values(filenames)

    # Save the result to a .npy file
    save_dict_to_npy(min_max_values, 'data_ranges_3.npy')

    print(min_max_values)

    print("Min and max values have been saved to 'data_ranges_2.npy'")

if __name__ == '__main__':
    main()
