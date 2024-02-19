from glob import glob
import os    #reading file paths or directories.
import mne   # For working with EEG data
import numpy as np
import pandas as pd
import matplotlib.pyplot as pltp
from keras.models import Sequential
from keras.layers import Conv1D ,BatchNormalization,LeakyReLU ,MaxPooling1D # Importing layers suitable for 1D data
from keras.layers import Flatten ,GlobalAveragePooling1D,Dense,Dropout,AveragePooling1D
from keras.backend import clear_session
import sys
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold,GridSearchCV, LeaveOneGroupOut
  

downloads_folder = os.path.expanduser("~/Downloads")  # Get the path to the "Downloads" directory
dataverse_files_folder = os.path.join(downloads_folder, "EEG_data")  # Combine with the folder name

# Use glob to get a list of file paths matching a specific pattern
file_paths = glob(os.path.join(dataverse_files_folder, '*.*'))  #Getting all files with any extension

# Print the list of file paths with folder names prefixed
for file_path in file_paths:
    folder_name, file_name = os.path.split(file_path)
    print(f"'{folder_name}|{file_name}'")


def extract_csv_info(file_paths):
    for file_path in file_paths:
        if file_path.endswith('.csv'):  # Check if the file is a CSV file
            folder_name, file_name = os.path.split(file_path)
            print(f"File: {file_name}")
            
            df = pd.read_csv(file_path)   # Read the CSV file using pandas

            columns = df.columns.tolist()  # Extract column names
            print("Column Names:")
            print(columns)

            print("Unique Values in Each Column:")
            for column in columns:
                unique_values = df[column].unique()  # Get unique values for each column
                print(f"{column}: {unique_values}")
            
            # print("Data:")     # Extract data below each column
            # for column in columns:
            #     column_data = df[column].tolist()
            #     print(f"{column}: {column_data}")
            
            print()  

# Call the function with the list of file paths
extract_csv_info(file_paths)

def read_csv_eeg(file_path):
    df = pd.read_csv(file_path)   # Read the CSV file using pandas
    
    # Check if all data in the DataFrame are numeric
    if not df.applymap(np.isreal).all().all():
        print(f"Skipping file {file_path} because it contains non-numeric data.")
        return None

    # Extract EEG data from DataFrame
    eeg_data = df.values.T  # Transpose the DataFrame to have channels as rows and samples as columns

    # Create info structure for MNE
    ch_names = df.columns.tolist()  # Extract column names as channel names
    ch_types = ['eeg'] * len(ch_names)  # Assuming all channels are EEG
    sfreq = 1000  # Set sampling frequency (adjust as needed)

    info = mne.create_info(ch_names, sfreq, ch_types)

    # Create RawArray object
    raw = mne.io.RawArray(eeg_data, info)

    return raw

# def visualize_eeg(file_paths):
#     for file_path in file_paths:
#         if file_path.endswith('.csv'):
#             folder_name, file_name = os.path.split(file_path)
#             print(f"File: {file_name}")

#             raw = read_csv_eeg(file_path)

#             if raw is None:
#                 continue

#             # Plot EEG signals before filtering
#             raw.plot(n_channels=len(raw.ch_names), title='EEG Signals before filtering', scalings='auto', show=False)
#             pltp.subplots_adjust(hspace=1.0)  # Adjust vertical space between subplots
#             pltp.show()

#             # Filter data
#             raw.filter(l_freq=0.5, h_freq=45)

#             # Plot EEG signals after filtering
#             raw.plot(n_channels=len(raw.ch_names), title='EEG Signals after filtering', scalings='auto', show=False)
#             pltp.subplots_adjust(hspace=1.0)  # Adjust vertical space between subplots
#             pltp.show()
            
#         elif file_path.endswith('.edf'):
#             folder_name, file_name = os.path.split(file_path)
#             print(f"File: {file_name}")

#             raw=mne.io.read_raw_edf(file_path,preload=True)
                
#             if raw is None:
#                 continue

#             # Plot EEG signals before filtering
#             raw.plot(n_channels=len(raw.ch_names), title='EEG Signals before filtering', scalings='auto', show=False)
#             pltp.subplots_adjust(hspace=1.0)  # Adjust vertical space between subplots
#             pltp.show()

#             # Filter data
#             raw.filter(l_freq=0.5, h_freq=45)

#             # Plot EEG signals after filtering
#             raw.plot(n_channels=len(raw.ch_names), title='EEG Signals after filtering', scalings='auto', show=False)
#             pltp.subplots_adjust(hspace=1.0)  # Adjust vertical space between subplots
#             pltp.show()

# visualize_eeg(file_paths)


def extract_features_from_epochs(epochs):
    features = []
    for epoch_data in epochs:
        if np.isnan(epoch_data).any() or np.isinf(epoch_data).any():
            print("Warning: NaN or infinite values encountered in epoch data. Skipping epoch.")
            continue
        
        flattened_epoch_data = epoch_data.flatten()  # Flatten the epoch data
        
        epoch_features = [
            np.mean(flattened_epoch_data),         # Mean
            np.std(flattened_epoch_data),          # Standard deviation
            np.var(flattened_epoch_data),          # Variance
            np.min(flattened_epoch_data),          # Minimum
            np.max(flattened_epoch_data),          # Maximum
            np.argmin(flattened_epoch_data),       # Index of minimum value
            np.argmax(flattened_epoch_data),       # Index of maximum value
            np.sqrt(np.nanmean(flattened_epoch_data)),  # Square root of mean
            np.sum(flattened_epoch_data),          # Sum
            stats.skew(flattened_epoch_data),      # Skewness
            stats.kurtosis(flattened_epoch_data)   # Kurtosis
        ]
        
        features.append(epoch_features)
    return np.array(features)

def extract_csv_filter(file_paths):
    for file_path in file_paths:
        if file_path.endswith('.csv'):  # Check if the file is a CSV file
            folder_name, file_name = os.path.split(file_path)
            print(f"File: {file_name}")
            
            raw = read_csv_eeg(file_path)  # Convert CSV to MNE RawArray
            if raw is None:
                continue
            
            # Filter data if needed
            raw.set_eeg_reference()
            raw.filter(l_freq=0.5, h_freq=45)
            epochs = mne.make_fixed_length_epochs(raw, duration=5, overlap=1)  # divides continuous data into smaller segments called epochs
            array = epochs.get_data()
            
            features = extract_features_from_epochs(array)
            
            # Concatenate features
            concatenated_features = np.concatenate(features, axis=0)
            
            print("Shape of concatenated features:", concatenated_features.shape)
            print("EEG Info:")
            print(raw.info)
            print()

        elif file_path.endswith('.edf'):
            folder_name, file_name = os.path.split(file_path)
            print(f"File: {file_name}")
            data = mne.io.read_raw_edf(file_path, preload=True)
            data.set_eeg_reference()  # helps to remove common noise sources and artifacts from the EEG data
            data.filter(l_freq=0.5, h_freq=45)
            epochs = mne.make_fixed_length_epochs(data, duration=5, overlap=1)  # divides continuous data into smaller segments called epochs
            array = epochs.get_data()
            
            features = extract_features_from_epochs(array)
            
            # Concatenate features
            concatenated_features = np.concatenate(features, axis=0)
            
            print("Shape of concatenated features:", concatenated_features.shape)
            print("EEG Info:")
            print(data.info)
            print()  

# Call the function with the list of file paths
extract_csv_filter(file_paths)























