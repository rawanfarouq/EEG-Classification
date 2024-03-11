from glob import glob
import os    #reading file paths or directories.
import mne   # For working with EEG data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.svm as svm
from mne.io import RawArray
from scipy.stats import entropy
from scipy.fft import rfft
from pyprep.find_noisy_channels import NoisyChannels
from mne.preprocessing import ICA

downloads_folder = os.path.expanduser("~/Downloads")  # Get the path to the "Downloads" directory
dataverse_files_folder = os.path.join(downloads_folder, "seizure")  # Combine with the folder name

# Use glob to get a list of file paths matching a specific pattern
file_paths = glob(os.path.join(dataverse_files_folder, '*.*'))  #Getting all files with any extension


def read_csv_eeg(file_path):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        if check_processed_from_columns(df, processed_data_keywords):
            print("The data appears to be processed.")
            return df, None  # Return the DataFrame and None for sfreq
        
        # Check if the first column is non-numeric and should be used as the index
        if not np.issubdtype(df.iloc[:, 0].dtype, np.number):
            df.set_index(df.columns[0], inplace=True)
        
        print(df.head())  # Displays the first 5 rows of the DataFrame

        # Check for a 'timestamp' column and calculate sampling frequency
        timestamp_present = 'timestamp' in df.columns
        if timestamp_present:
            time_diffs = np.diff(df['timestamp'].values)
            if np.any(time_diffs == 0):
                print("Duplicate timestamps found. Sampling frequency cannot be determined.")
                return None, None
            avg_sampling_period = np.mean(time_diffs)
            sfreq = 1 / avg_sampling_period             #number of samples obtained per second.
            df.drop('timestamp', axis=1, inplace=True)  # Remove timestamp for MNE
        else:
            sfreq = 500  

        
        eeg_data = df.values.T

        # Create MNE info structure
        ch_names = df.columns.tolist()
        ch_types = ['eeg'] * len(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        # Create RawArray
        raw = mne.io.RawArray(data=eeg_data, info=info)

        return raw, sfreq

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None, None


def read_edf_eeg(file_path):
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True)
        sfreq = raw.info['sfreq']  # Gets the sampling frequency from the EDF file's header

        # print(raw.info)  # Summarizes the info structure of the raw object
        # print(raw.ch_names)  # List of all channel names
        # print(raw.get_channel_types())  # List of channel types
        print(raw.to_data_frame().describe())  # For EDF files converted to DataFrame

        return raw, sfreq

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None, None

def plot_raw_eeg(raw, title='EEG Data'):
    # Plot the data
    raw.plot(title=title, scalings='auto' ,show=True, block=True)
    

def preprocess_raw_eeg(raw, sfreq):
    # High-pass filtering to remove slow drifts
    raw.filter(l_freq=0.5, h_freq=None)

    # Low-pass filtering to remove high-frequency noise
    raw.filter(l_freq=None, h_freq=40.0)

    # Notch filter to remove power line noise at 50 Hz or 60 Hz and its harmonics
    notch_freqs = np.arange(50, sfreq / 2, 50)  # Assuming 50 Hz power line noise
    raw.notch_filter(notch_freqs)

    # Instantiate NoisyChannels with the filtered raw data
    noisy_detector = NoisyChannels(raw)

    # Now you can use the instance methods to find bad channels
    noisy_detector.find_bad_by_correlation()
    noisy_detector.find_bad_by_deviation()

    # Combine all the bad channels
    bads = noisy_detector.get_bads()

    # Mark bad channels in the info structure
    raw.info['bads'] = bads

    # Interpolate bad channels using good ones
    if bads:
        raw.drop_channels(bads)


     # Independent Component Analysis (ICA) to remove artifacts
    ica = ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw)
    # Apply ICA to the raw data to remove the bad components
    raw = ica.apply(raw)

    return raw 

def extract_features(raw):
    # Initialize a dictionary to hold features
    features = {}
    
    # Get the data from the Raw object
    data = raw.get_data()
    
    # Apply FFT to each channel in the data
    fft_vals = np.fft.rfft(data, axis=1)
    
    # Get the magnitude of the FFT
    fft_magnitude = np.abs(fft_vals)
    
    # Get the power spectral density (PSD) by squaring the magnitudes
    psd_vals = np.square(fft_magnitude)
    
    # Frequency resolution
    freq_res = np.fft.rfftfreq(n=data.shape[1], d=1.0/raw.info['sfreq'])
    
    # Define frequency bands of interest (example bands)
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 45)
    }
    
    # Calculate power in each frequency band
    for band, (fmin, fmax) in bands.items():
        # Find the indexes of the frequencies that are within the band
        freq_inds = np.where((freq_res >= fmin) & (freq_res <= fmax))[0]
        
        # Sum the PSD values within the band to get the band power
        band_power = psd_vals[:, freq_inds].sum(axis=1)
        features[f'{band}_power'] = band_power
    
   
    return features

label_mapping = {
    'happy': ('Emotions', 'Happy'),
    'sad': ('Emotions', 'Sad'),
    'angry': ('Emotions', 'Angry'),
    'seizure': ('Brain Disorder', 'Seizure'),
    'epilepsy': ('Brain Disorder', 'Epilepsy'),
    'epliepsy seizure' : ('Brain Disorder','Eplieplsy Seizure'),
    'motor movement':('Motor Movement','Motor Movement'),
    'right hand': ('Motor Movement','Moving Right Hand'),
    'left hand': ('Motor Movement','Moving Left Hand'),
}

def extract_labels_from_filename(file_path):
    # Extract labels from the file name or path
    for keyword, (main_class, subclass) in label_mapping.items():
        if keyword in file_path.lower():
            return main_class, subclass
    return 'Normal Activity', None  # Default to Normal Activity if no keyword matches


def check_processed_from_columns(df, keywords):
    for column in df.columns:
        for keyword in keywords:
            if keyword in column.lower():  # Check if keyword is a substring of column name
                return True
    return False

processed_data_keywords = [
    'feature', 'power', 'mean', 'stddev', 'variance', 
    'entropy', 'fft', 'frequency', 'band', 'delta', 
    'theta', 'alpha', 'beta', 'gamma','max','min','avgmax','avgmin','skew','kurt',
    'median','var','energy','sigma','coefficient','D0','D1','D2','KFD','PFD','HFD','LZC',
    'CTM','AMI','RR','determinant','det','Lam','avg','average','Lmax','Vmin','TT','divergence'
]

# Iterate through each file path in the list of file paths
for file_path in file_paths:
    if file_path.endswith('.csv'):
        main_class, subclass = extract_labels_from_filename(file_path)
        print(f"Main class: {main_class}, Subclass: {subclass}")
        
        # Read the CSV file and determine if it's processed
        df_or_raw, sfreq = read_csv_eeg(file_path)
        
        if isinstance(df_or_raw, pd.DataFrame):
            # Handle the processed data (since df_or_raw is a DataFrame, it's processed)
            print("The data appears to be processed and features are already extracted.")
        elif df_or_raw is not None:
            # Handle the raw data
            print("The data appears to be raw.")
            print(f"CSV file {file_path} read successfully with sampling frequency {sfreq} Hz.")
            plot_raw_eeg(df_or_raw, title=f'Raw EEG Data: {file_path}')
            raw_preprocessed = preprocess_raw_eeg(df_or_raw, sfreq)
            plot_raw_eeg(raw_preprocessed, title=f'Preprocessed EEG Data: {file_path}')
            if raw_preprocessed is not None:
                fft_features = extract_features(raw_preprocessed)
                print(fft_features)

    elif file_path.lower().endswith('.edf'):
        main_class, subclass = extract_labels_from_filename(file_path)
        print(f"Main class: {main_class}, Subclass: {subclass}")
        raw, sfreq = read_edf_eeg(file_path)
        
        if raw is not None:
            print(f"EDF file {file_path} read successfully with sampling frequency {sfreq} Hz.")
            raw_preprocessed = preprocess_raw_eeg(raw, sfreq)
            # plot_raw_eeg(raw_preprocessed, title=f'Preprocessed EEG Data: {file_path}')
            if raw_preprocessed is not None:
                fft_features = extract_features(raw_preprocessed)
                print(fft_features)

    else:
        print(f"File {file_path} is not a recognized EEG file type.")