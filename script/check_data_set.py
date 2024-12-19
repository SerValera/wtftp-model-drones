import numpy as np

file_path = '/home/vs/wtftp-model/data_ranges.npy'

example_data = np.load(file_path, allow_pickle=True) 
    
print(example_data)

np.save('example_data_0.npy', example_data[0])