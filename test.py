from glob import glob
import os    #reading file paths or directories.
import mne   # For working with EEG data
import numpy as np
import pandas
import matplotlib.pyplot as pltp
from keras.models import Sequential
from keras.layers import Conv1D   # Importing layers suitable for 1D data
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense   

downloads_folder = os.path.expanduser("~/Downloads")  # Get the path to the "Downloads" directory
dataverse_files_folder = os.path.join(downloads_folder, "dataverse_files")  # Combine with the folder name

# Use glob to get a list of file paths matching a specific pattern
file_paths = glob(os.path.join(dataverse_files_folder, '*.*'))  #Getting all files with any extension

# Print the list of file paths with folder names prefixed
for file_path in file_paths:
    folder_name, file_name = os.path.split(file_path)
    print(f"'{folder_name}|{file_name}'")



