from glob import glob
import os    #reading file paths or directories.
import mne   # For working with EEG data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import eeglib.preprocessing
import sklearn.svm as svm
from mne.io import RawArray
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC #classifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


downloads_folder = os.path.expanduser("~/Downloads")  # Get the path to the "Downloads" directory
dataverse_files_folder = os.path.join(downloads_folder, "complete")  # Combine with the folder name

# Use glob to get a list of file paths matching a specific pattern
file_paths = glob(os.path.join(dataverse_files_folder, '*.*'))  #Getting all files with any extension

# # Print the list of file paths with folder names prefixed
# for file_path in file_paths:
#     folder_name, file_name = os.path.split(file_path)
#     print(f"'{folder_name}|{file_name}'")


LABELS = {
    'Normal Activity': 0,
    'Brain Disorder': 1,
    'Emotions': 2,
    'Motor Movement': 3
}

def label_file(file_path):
    file_name = os.path.basename(file_path)
    if 'normal' in file_name.lower():
        return LABELS['Normal Activity']
    elif 'disorder' in file_name.lower():
        return LABELS['Brain Disorder']
    elif 'emotion' in file_name.lower():
        return LABELS['Emotions']
    elif 'movement' in file_name.lower():
        return LABELS['Motor Movement']
    elif 'seizure' in file_name.lower():
        return LABELS['Brain Disorder']
    elif 'eplipsy' in file_name.lower():
        return LABELS['Brain Disorder']
    else:
        return None


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

extract_csv_info(file_paths)

def preprocess_data(raw):
    
   # raw.filter(l_freq=0.5, h_freq=45)    #filter to remove noise and unwanted frequencies
    
    low_freq=0.5
    high_freq=45
    sample_rate=raw.info['sfreq']

    raw_data=raw.get_data()
    filtered_data=eeglib.preprocessing.bandPassFilter(raw_data,sampleRate=sample_rate,highpass=low_freq,lowpass=high_freq)

    raw._data=filtered_data
    
    # Apply ICA for artifact removal
    ica = mne.preprocessing.ICA(n_components=8, random_state=97, max_iter=800)
    ica.fit(raw)
    
    # Apply ICA transformation to the data
    raw_corrected = ica.apply(raw)
    
    return raw_corrected

def read_csv_eeg(file_path):
    df = pd.read_csv(file_path)   # Read the CSV file using pandas
    
    # Check if all data in the DataFrame are numeric
    numeric_columns = []
    for column in df.columns:
        if df[column].dtype in [np.int64, np.float64]:
            numeric_columns.append(column)
        else:
            print(f"Skipping column '{column}' in file {file_path} because it contains non-numeric data.")
    
    if not numeric_columns:
        print(f"Skipping file {file_path} because it contains no numeric columns.")
        return None

    # Extract EEG data from DataFrame
    eeg_data = df[numeric_columns].values.T  # Transpose the DataFrame to have channels as rows and samples as columns

    # Create info structure for MNE
    ch_names = numeric_columns  # Extract column names as channel names
    ch_types = ['eeg'] * len(ch_names)  # Assuming all channels are EEG
    sfreq = 1000  # Set sampling frequency (adjust as needed)

    info = mne.create_info(ch_names, sfreq, ch_types)

    # Create RawArray object
    raw = mne.io.RawArray(eeg_data, info)

    # Preprocess the data
    raw_corrected = preprocess_data(raw)

    # Extract features
    features = extract_features(raw_corrected)
    print("Extracted Features Shape from CSV file:", features.shape)

    # Assign label
    label = label_file(file_path)

    return features, label

def read_edf_eeg(file_path):

    raw_corrected = mne.io.read_raw_edf(file_path, preload=True)

    # Preprocess the data
    raw_corrected = preprocess_data(raw_corrected)

    # Extract features
    features = extract_features(raw_corrected)
    print("Extracted Features Shape from EDF file:", features.shape)

    # Assign label
    label = label_file(file_path)

    return features, label


def extract_features(raw_corrected):
    # Initialize an empty list to store the extracted features
    features = []
    
    # Define the frequency bands of interest
    freq_bands = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 45)]
    
    # Compute FFT for each channel and extract features
    for ch in raw_corrected.ch_names:
        data = raw_corrected.get_data(ch)
        fft_result = np.abs(np.fft.fft(data))
        freqs = np.fft.fftfreq(data.shape[1], 1/raw_corrected.info['sfreq'])
        
        # Calculate power spectral density for each frequency band
        psd = []
        for f_band in freq_bands:
            freq_mask = (freqs >= f_band[0]) & (freqs < f_band[1])
            psd.append(np.sum(fft_result[:, freq_mask], axis=1))
        
        # Append the extracted features to the list
        features.append(np.concatenate(psd))
    
    return np.array(features)

# Load data and extract features with labels
data = []
labels = []
for file_path in file_paths:
    if file_path.endswith('.csv'):
        features_csv, label_csv = read_csv_eeg(file_path)
        if features_csv is not None:
            print("File:", file_path)
            print("Features shape:", features_csv.shape)
            print("Label:", label_csv)
            data.append(features_csv)
            labels.append(label_csv)
    elif file_path.endswith('.edf'):
        features_edf, label_edf = read_edf_eeg(file_path)
        if features_edf is not None:
            print("File:", file_path)
            print("Features shape:", features_edf.shape)
            print("Label:", label_edf)
            data.append(features_edf)
            labels.append(label_edf)


# Convert lists to arrays
data = np.array(data)
labels = np.array(labels)

# Check the shapes of data and labels
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# # Initialize and train your classification model (e.g., SVM)
# clf = svm.SVC()
# clf.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = clf.predict(X_test)

# # Evaluate the performance of your classification model
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)



# def visualize_eeg_time_domain(file_paths):
#     for file_path in file_paths:
#         if file_path.endswith('.csv'):
#             folder_name, file_name = os.path.split(file_path)
#             print(f"File: {file_name}")

#             raw_corrected = read_csv_eeg(file_path)

#             if raw_corrected is None:
#                 continue

#             # Plot EEG signals before filtering
#             raw_corrected.plot(n_channels=len(raw_corrected.ch_names), title='EEG Signals after filtering in time domain', scalings='auto', show=False)
#             plt.subplots_adjust(hspace=1.0)  # Adjust vertical space between subplots
#             plt.show()

#             features=extract_features(raw_corrected)  #feature extraction
#             print("Extracted Features Shape from CSV file:", features.shape)

            
#         elif file_path.endswith('.edf'):
#             folder_name, file_name = os.path.split(file_path)
#             print(f"File: {file_name}")

#             raw_corrected = read_edf_eeg(file_path)

#             if raw_corrected is None:
#                 continue

#             # Plot EEG signals after artifact removal
#             raw_corrected.plot(n_channels=len(raw_corrected.ch_names), title='EEG Signals after artifact removal', scalings='auto', show=False)
#             plt.subplots_adjust(hspace=1.0)  # Adjust vertical space between subplots
#             plt.show()

#             features=extract_features(raw_corrected)  #feature extraction
#             print("Extracted Features Shape from EDF file:", features.shape)



# visualize_eeg_time_domain(file_paths)

# def visualize_eeg_frequency_domain(file_paths):
#     for file_path in file_paths:
#         if file_path.endswith('.csv'):
#             folder_name, file_name = os.path.split(file_path)
#             print(f"File: {file_name}")

#             raw_corrected = read_csv_eeg(file_path)

#             if raw_corrected is None:
#                 continue

#             # Extract features
#             features = extract_features(raw_corrected)

#             # Plot the power spectral density (PSD) for each channel
#             for i, ch_name in enumerate(raw_corrected.ch_names):
#                 plt.figure()
#                 plt.plot(features[i])
#                 plt.title(f'Power Spectral Density (PSD) - Channel {ch_name}')
#                 plt.xlabel('Frequency (Hz)')
#                 plt.ylabel('Power')
#                 plt.grid(True)
#                 plt.show()

#         elif file_path.endswith('.edf'):
#             folder_name, file_name = os.path.split(file_path)
#             print(f"File: {file_name}")

#             raw_corrected = read_edf_eeg(file_path)

#             if raw_corrected is None:
#                 continue

#             # Extract features
#             features = extract_features(raw_corrected)

#             # Plot the power spectral density (PSD) for each channel
#             for i, ch_name in enumerate(raw_corrected.ch_names):
#                 plt.figure()
#                 plt.plot(features[i])
#                 plt.title(f'Power Spectral Density (PSD) - Channel {ch_name}')
#                 plt.xlabel('Frequency (Hz)')
#                 plt.ylabel('Power')
#                 plt.grid(True)
#                 plt.show()

# visualize_eeg_frequency_domain(file_paths)


































