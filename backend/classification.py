from glob import glob
import os    #reading file paths or directories.
import mne   # For working with EEG data
import numpy as np
import pandas as pd
import random
import scipy.io
import h5py
import re
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tsfresh import select_features
from tsfresh.feature_extraction import ComprehensiveFCParameters,extract_features,MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute
from mne.io import RawArray
from mne import make_fixed_length_events, Epochs, create_info
from mne.filter import filter_data
from mne.channels import make_standard_montage
from mne.viz import plot_alignment, Brain
from scipy.stats import entropy,skew,kurtosis
from scipy.fft import rfft
from scipy.signal import welch, find_peaks
from pyprep.find_noisy_channels import NoisyChannels
from mne.preprocessing import ICA
from mne.decoding import CSP
from eeglib.preprocessing import bandPassFilter
from mpl_toolkits.mplot3d import Axes3D
from nilearn import plotting
import plotly.graph_objects as go
import cProfile
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from collections import defaultdict





downloads_folder = os.path.expanduser("~/Downloads")  # Get the path to the "Downloads" directory
dataverse_files_folder = os.path.join(downloads_folder, "features")  # Combine with the folder name

# Use glob to get a list of file paths matching a specific pattern
file_paths = glob(os.path.join(dataverse_files_folder, '*.*'))  #Getting all files with any extension
all_features = pd.DataFrame()


def read_eeg_file(file_path):
    try:
        # Check the file extension and read the file into a DataFrame accordingly
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx', '.xlsm', '.xlsb')):
            df = pd.read_excel(file_path)  # This reads the first sheet by default
        else:
            print(f"Unsupported file type for {file_path}.")
            return None, None
        
        print(f"Number of samples (time points) in the recording: {len(df)}")

        # unique_prefixes = get_unique_prefixes(df.columns)
        # print("Unique prefixes:", unique_prefixes)

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
            sfreq = 1 / avg_sampling_period  # number of samples obtained per second.
            df.drop('timestamp', axis=1, inplace=True)  # Remove timestamp for MNE
        else:
            sfreq = 500  # You may want to adjust this default frequency

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

        print("Raw info:",raw.info)  # Summarizes the info structure of the raw object
        #print("Raw ch_names:",raw.ch_names)  # List of all channel names
        # print(raw.get_channel_types())  # List of channel types
        print(raw.to_data_frame().describe())  # For EDF files converted to DataFrame

        return raw, sfreq

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None, None

def plot_raw_eeg(raw, title='EEG Data', picks=None, separate_channels=True):
    if picks is not None:
        # Create a new Raw object with only the picked channels
        raw = raw.copy().pick_channels(picks)
    
    # If separate_channels is True, adjust the scalings to avoid overlap
    if separate_channels:
        # Define a large scaling factor to ensure channels are separated
        scaling_factor = 0.5 * np.max(np.abs(raw._data))
        scalings = {'eeg': scaling_factor}
    else:
        scalings = 'auto'
    
    # Plot the data
    raw.plot(title=title, scalings=scalings, show=True, block=True)

def read_mat_eeg(file_path):
    try:
        # Load MATLAB file
        mat = scipy.io.loadmat(file_path)

        # Access the EEG data under the 'data' key
        eeg_data = mat['data']

        # Check the shape of the data
        if len(eeg_data.shape) == 3:
            # Reshape data assuming it is in the format (trials, channels, samples)
            n_trials, n_channels, n_samples = eeg_data.shape

            # Concatenate trials to have a continuous data stream
            eeg_data = np.concatenate(eeg_data, axis=1)  # This concatenates along the second axis (samples)
            eeg_data = eeg_data.reshape(n_channels, n_trials * n_samples)
        else:
            raise ValueError(f"Unexpected data dimensions {eeg_data.shape}")

        # Define channel names and types
        ch_names = ['EEG ' + str(i) for i in range(n_channels)]
        ch_types = ['eeg'] * n_channels

        # Define the sampling frequency
        sfreq = 500  # Replace with the actual sampling frequency if available

        # Create MNE info structure
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        # Create RawArray
        raw = mne.io.RawArray(eeg_data, info=info)
        
        # Extract labels if 'labels' key exists
        if 'labels' in mat:
            labels = mat['labels'].flatten()  # Ensure it is a 1D array
        else:
            raise ValueError("The key 'labels' was not found in the .mat file.")

        return raw, sfreq, labels

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None, None, None
    
    
def visualize_3d_brain_model(raw):

    
    # Define the correct mapping from the current channel names to the desired channel names
    channel_mapping = {'EEG 0': 'Fp1','EEG 1': 'Fp2','EEG 2': 'F7','EEG 3': 'F3',
                     'EEG 4': 'Fz','EEG 5': 'F4', 'EEG 6': 'F8', 'EEG 7': 'FC5','EEG 8': 'FC1',
                    'EEG 9': 'FC2','EEG 10': 'FC6', 'EEG 11': 'T3',
                    'EEG 12': 'C3','EEG 13': 'Cz', 'EEG 14': 'C4', 'EEG 15': 'T4',
                    'EEG 16': 'CP5', 'EEG 17': 'CP1','EEG 18': 'CP2','EEG 19': 'CP6','EEG 20': 'T5',
                    'EEG 21': 'P3','EEG 22': 'Pz','EEG 23': 'P4','EEG 24': 'T6',
                    'EEG 25': 'PO3','EEG 26': 'PO4', 'EEG 27': 'O1', 'EEG 28': 'Oz',
                         'EEG 29': 'O2','EEG 30': 'A1',   'EEG 31': 'A2', }
    

   # Rename the channels in the raw object
    raw.rename_channels(channel_mapping)

    # Set the montage (standard 10-20 system) for EEG data after renaming channels
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    
     # Ensure 'fsaverage' is correctly installed and get its path
    subject = 'sample'
    fs_dir = mne.datasets.sample.data_path()
    subjects_dir = os.path.join(fs_dir, 'subjects')

    # Set the SUBJECTS_DIR environment variable
    os.environ['SUBJECTS_DIR'] = subjects_dir

    # Now check if the necessary files exist
    bem_dir = os.path.join(subjects_dir, subject, 'bem')
    head_surf_files = [
        'outer_skin.surf',
        'flash/outer_skin.surf',
        'fsaverage-head-sparse.fif',
        'fsaverage-head.fif'
    ]
    for file in head_surf_files:
        full_path = os.path.join(bem_dir, file)
        if not os.path.isfile(full_path):
            print(f'Missing file: {full_path}')

    mne.viz.set_3d_backend('notebook')        
    
    # Plot the sensor locations, including the head surface
    fig = mne.viz.plot_alignment(
        raw.info,
        trans='fsaverage',
        subject=subject,
        subjects_dir=subjects_dir,
        dig=True,
        eeg=['original', 'projected'],
        show_axes=True,
        surfaces='head-dense',  # Use a denser head surface for better visualization (can be changed to 'head' for sparser)
       
    )

    # Set the 3D view of the figure
    mne.viz.set_3d_view(figure=fig, azimuth=90, elevation=90, distance=0.6)

    return fig



def preprocess_channel_data(channel_data, sfreq, l_freq_hp, h_freq_lp):
    # High-pass filtering to remove slow drifts
    channel_data = filter_data(channel_data, sfreq, l_freq=l_freq_hp, h_freq=h_freq_lp, verbose=False)
    
    # Low-pass filtering to remove high-frequency noise
    #channel_data = filter_data(channel_data, sfreq, l_freq=None, h_freq=h_freq_lp, verbose=False)
    
    # Notch filter to remove power line noise at 50 Hz or 60 Hz and its harmonics
    # for notch_freq in notch_freqs:
    #     channel_data = filter_data(channel_data, sfreq, l_freq=notch_freq - 0.1, h_freq=notch_freq + 0.1, method='iir', verbose=False)
    
    return channel_data



def preprocess_raw_eeg(raw, sfreq):

     # Get the data from the Raw object
    eeg_data = raw.get_data()

    preprocessed_data = np.empty(eeg_data.shape)
    

    # Notch filter to remove power line noise at 50 Hz or 60 Hz and its harmonics
    #notch_freqs = np.arange(50, sfreq / 2, 50)  # Assuming 50 Hz power line noise

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
    ica = ICA(n_components=10, random_state=97, max_iter=800)
    ica.fit(raw)
    # Apply ICA to the raw data to remove the bad components
    raw = ica.apply(raw)

    for i, channel in enumerate(eeg_data):
        preprocessed_data[i] = preprocess_channel_data(channel, sfreq, l_freq_hp=0.5, h_freq_lp=45.0)
    
    # Create an MNE RawArray object with the preprocessed data
    ch_names = ['EEG %03d' % i for i in range(preprocessed_data.shape[0])]  # Create channel names
    ch_types = ['eeg'] * preprocessed_data.shape[0]
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    preprocessed_raw = RawArray(preprocessed_data, info)

    return preprocessed_raw 


def extract_features_from_channel(channel_data, sfreq, epoch_length=1.0):
    # Assuming channel_data is a 1D numpy array
    n_samples = channel_data.shape[0]
    n_epochs = int(n_samples / (epoch_length * sfreq))
    features_list = []

    for i in range(n_epochs):
        start_sample = int(i * epoch_length * sfreq)
        end_sample = int(start_sample + epoch_length * sfreq)
        epoch = channel_data[start_sample:end_sample]

        # Time-domain features
        mean_val = np.mean(epoch)
        std_val = np.std(epoch)
        skew_val = skew(epoch)
        rms_val = np.sqrt(np.mean(epoch**2))  # Root mean square
        variance_val = np.var(epoch)
        average_val = np.mean(epoch) 
       

        # # Frequency-domain features
        # fft_vals = rfft(epoch)
        # psd_vals = np.abs(fft_vals) ** 2
        # freq_res = np.fft.rfftfreq(n=len(epoch), d=1.0 / sfreq)
        

        # # Welch's method to estimate power spectral density
        # f, psd_welch = welch(epoch, sfreq)

        # Define frequency bands of interest (bands)
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 45)
        }
        
        # Initialize a dictionary to hold features for the current epoch
        features = {
            'mean': mean_val,
            'std': std_val,
            'skew': skew_val,
            'rms': rms_val,
            'variance': variance_val,
           'peak_to_peak': np.ptp(epoch),
            # 'zero_crossing_rate': ((epoch[:-1] * epoch[1:]) < 0).sum() / (epoch_length * sfreq),
            # 'signal_magnitude_area': np.trapz(np.abs(epoch), dx=1/sfreq),
            'waveform_length': np.sum(np.abs(np.diff(epoch))),
        }
        # Calculate power in each frequency band
       
        # for band, (fmin, fmax) in bands.items():
        #     # Find the power spectral density within the band
        #     band_inds = np.where((f >= fmin) & (f <= fmax))[0]
        #     band_psd = psd_welch[band_inds]
            
        #     # Find the peak frequency and its power
        #     peak_freq = f[band_inds][np.argmax(band_psd)]
        #     peak_power = np.max(band_psd)
            
        #     features[f'{band}_peak_freq'] = peak_freq
        #     features[f'{band}_peak_power'] = peak_power
        

        # Append the features of the current epoch to the list
        features_list.append(features)
  
    
    return features_list

def extract_features_mat(raw, sfreq, labels, epoch_length=1.0):
    print("Entered extract_features method.")

    # Get data from the Raw object
    data = raw.get_data()
    n_channels, n_times = data.shape

    # Calculate the number of samples per epoch
    samples_per_epoch = int(epoch_length * sfreq)
    
    # Calculate the total number of epochs that will be created
    n_epochs = n_times // samples_per_epoch
    
    # If there are more epochs than labels, we should only take as many epochs as there are labels
    if n_epochs > len(labels):
        n_epochs = len(labels)
    
    # Initialize a list to hold features from all epochs
    all_features = []

    # Extract features for each channel and epoch
    for channel_idx in range(n_channels):
        channel_data = data[channel_idx]
        # Split the channel data into epochs
        epochs = np.array_split(channel_data, n_epochs)
        for epoch in epochs:
            channel_features = extract_features_from_channel(epoch, sfreq, epoch_length)
            # Each epoch should result in a single feature vector
            all_features.extend(channel_features)  # Assuming extract_features_from_channel returns a list of feature dicts for each epoch

    # Now, ensure we only take as many feature sets as there are labels
    all_features = all_features[:len(labels)]

    # Create a DataFrame from the features list
    feature_df = pd.DataFrame(all_features)
    print("Length of features before adding labels:", len(feature_df))

    # Add the labels to the features DataFrame
    feature_df['label'] = labels

    return feature_df



def extract_features_edf(raw, epoch_length=1.0):
    # Define the epochs data structure
    events = mne.make_fixed_length_events(raw, duration=epoch_length)
    epochs = mne.Epochs(raw, events, tmin=0.0, tmax=epoch_length, baseline=None, preload=True)
    
    # Initialize a list to hold features from all epochs
    all_features = []

    # Extract features for each epoch
    for _, epoch in enumerate(epochs):
        # Initialize a dictionary to hold features for the current epoch
        features = {}
        
        # Apply FFT to the epoch
        fft_vals = rfft(epoch, axis=1)
        
        # Compute power spectral density (PSD)
        psd_vals = np.abs(fft_vals) ** 2
        
        # Frequency resolution
        freq_res = np.fft.rfftfreq(n=epoch.shape[1], d=1.0 / epochs.info['sfreq'])
        
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
            # Average the band power across channels
            mean_band_power = np.mean(band_power, axis=0)
            
            # Store the mean band power in the features dictionary
            features[f'{band}_power'] = mean_band_power
        
        # Add entropy of the distribution of power values
        features['entropy'] = entropy(np.mean(psd_vals, axis=0))
        
        #print("test1:", features)  # debug
        # Append the features of the current epoch to the list
        all_features.append(features)
    
    # Create a DataFrame from the features list
    feature_df = pd.DataFrame(all_features)
    
    return feature_df

def check_processed_from_columns(df, keywords):
    for column in df.columns:
        for keyword in keywords:
            if keyword in column.lower():  # Check if keyword is a substring of column name
                return True
    return False

def identify_feature_extraction_methods(df, keywords):
    identified_methods = []
    for column in df.columns:
        for keyword in keywords:
            if keyword in column.lower():  # Check if keyword is a substring of column name
                if keyword not in identified_methods:
                    identified_methods.append(keyword)
    return identified_methods if identified_methods else None

def identify_feature_calculation_methods(df, bands_keywords, calculation_keywords):
    identified_methods = {}
    for band in bands_keywords:
        for column in df.columns:
            if band in column.lower():
                # Initialize the band key in the dictionary if not present
                if band not in identified_methods:
                    identified_methods[band] = []
                # Check for calculation method in the column name
                for calc_method in calculation_keywords:
                    if calc_method in column.lower() and calc_method not in identified_methods[band]:
                        identified_methods[band].append(calc_method)
    return identified_methods if identified_methods else None

def identify_calculation_method(column_name, processed_data_keywords):
    for method in processed_data_keywords:
        if method.lower() in column_name.lower():
            return method
    return "unknown"

def create_epochs_from_preprocessed_features(data, epoch_length=1.0, sfreq=None):
    if isinstance(data, mne.io.Raw):
        # If the passed data is a Raw object, get the DataFrame representation
        df = data.to_data_frame()
    elif isinstance(data, pd.DataFrame):
        # If the data is already a DataFrame, use it directly
        df = data
    else:
        raise ValueError("Data must be a pandas DataFrame or MNE Raw object.")
                         
    if sfreq is None:
        raise ValueError("Sampling frequency must be provided if not inherent in the DataFrame index.")

    # Separate the feature columns and the label column
    feature_columns = df.columns[:-1]  # Excludes the last column, which is assumed to be labels
    label_column = df.columns[-1]  # The last column is labels

    # Work with features only for the mean calculation
    features_df = df[feature_columns]

    # Calculate the number of samples per epoch
    samples_per_epoch = int(sfreq * epoch_length)

    # Calculate the number of complete epochs that can be formed
    num_complete_epochs = len(features_df) // samples_per_epoch

    # Extract only the part of the DataFrame that can be evenly divided into epochs
    complete_epoch_features = features_df.iloc[0:num_complete_epochs * samples_per_epoch]

    # Reshape the features data to have an index for epochs
    reshaped_features = complete_epoch_features.values.reshape(num_complete_epochs, samples_per_epoch, -1)

    # Initialize a list to hold features from all epochs
    all_features = []

    # Iterate over the epochs to calculate mean features
    for epoch_data in reshaped_features:
        # Calculate features for the current epoch
        epoch_features = epoch_data.mean(axis=0).tolist()  # Example feature calculation

        # Append the features of the current epoch to the list
        all_features.append(epoch_features)

    # Create a DataFrame from the features list
    feature_df = pd.DataFrame(all_features, columns=feature_columns)

    # Now, handle the label column separately
    # We assume the label for each epoch is the label of the first sample in the epoch
    labels = df[label_column].iloc[::samples_per_epoch].values[:num_complete_epochs]

    # Add the labels column to the features DataFrame
    feature_df[label_column] = labels


    print(f'Feature DataFrame shape: {feature_df.shape}')
    return feature_df

  

processed_data_keywords = [
    'feature', 'power', 'mean', 'stddev', 'variance', 
    'entropy', 'fft', 'frequency', 'band', 'delta', 
    'theta', 'alpha', 'beta', 'gamma','max','min','avgmax','avgmin','skew','kurt',
    'median','var','energy','sigma','coefficient','D0','D1','D2','KFD','PFD','HFD','LZC',
    'CTM','AMI','RR','determinant','det','Lam','avg','average','Lmax','Vmin','TT','divergence',
    'time frequency distributions','TFD','fast fourier transformation','EM',
    'eigenvector methods','WT','wavelet transform','correlate'
]


# Update the feature extraction to return a DataFrame
def extract_features_as_df(raw):
    features_df = extract_features_mat(raw)  # Should now be a DataFrame
    return features_df


# def process_processed_data(data, sfreq):
#     global all_features
#     print("The data appears to be processed and features are already extracted.")
#     if isinstance(data, pd.DataFrame):
#         # If data is a DataFrame, pass it directly
#         epochs_df = create_epochs_from_preprocessed_features(data, epoch_length=1.0, sfreq=500)
#         print("Epochs DataFrame created with preprocessed features.")
#         print(epochs_df.head()) 
#         all_features = pd.concat([all_features, epochs_df], ignore_index=True)
#     elif isinstance(data, mne.io.BaseRaw):
#         # If data is a Raw object, convert it to a DataFrame first
#         data_df = data.to_data_frame()
#         epochs_df = create_epochs_from_preprocessed_features(data_df, epoch_length=1.0, sfreq=sfreq)
#         print(epochs_df.head()) 
#         all_features = pd.concat([all_features, epochs_df], ignore_index=True)
#     else:
#         raise ValueError("Data must be a pandas DataFrame or MNE Raw object.")

def add_features_to_all_features(data):
    # Check if the data is a pandas DataFrame or an MNE Raw object
    if isinstance(data, pd.DataFrame):
        # If data is already a DataFrame, use it directly
        df = data
    elif isinstance(data, mne.io.BaseRaw):
        # If data is a Raw MNE object, convert to a DataFrame
        df = data.to_data_frame()
    else:
        raise ValueError("Data must be a pandas DataFrame or MNE Raw object.")
    
    # Here we assume that the DataFrame df contains features in its columns
    # and that the last column is the label
    return df

def process_processed_data(data):
    global all_features
    print("The data appears to be processed and features are already extracted.")

    # Add a print statement to check the type of data received
    print(f"Type of data received: {type(data)}")  # This will show you the type of data
    
    try:
        features_df = add_features_to_all_features(data)
        print("Features have been added to the all_features DataFrame.")
        
        # Assuming all_features is defined globally and is a pd.DataFrame
        all_features = pd.concat([all_features, features_df], ignore_index=True)
        print(all_features.head())
    except ValueError as e:
        print(e)
        # Here you can handle the case when data is not the correct type
        # For example, you might want to log the error and stop processing


# Custom transformer to handle EEG data reshaping
class EEGToDataFrame(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # This transformer does not need to learn anything from the data
        return self

    def transform(self, X, y=None):
        # Convert the 3D EEG array into a long pandas DataFrame suitable for tsfresh
        # Assuming X is of shape [trials, channels, time samples]
        n_trials, n_channels, n_samples = X.shape
        # Create a DataFrame with columns id, time, and one column per channel
        stacked_eeg = X.transpose(0, 2, 1).reshape(-1, n_channels)
        trial_ids = np.repeat(np.arange(n_trials), n_samples)
        time_samples = np.tile(np.arange(n_samples), n_trials)
        df = pd.DataFrame(stacked_eeg, columns=[f'channel_{i}' for i in range(n_channels)])
        df['id'] = trial_ids
        df['time'] = time_samples
        return df
      

def get_subject_identifier(file_path):
    # Extracts 'sub-001' from 'sub-001_ses-01_task_motorimagery_eeg.edf'
    return file_path.split('_')[0]

def get_unique_prefixes(column_names):
    # Use a set to keep track of unique prefixes
    unique_prefixes = set()
    
    # Regular expression to match the prefix before the first underscore
    pattern = re.compile(r'^[^_]+')
    
    for name in column_names:
        # Use the regex pattern to search for the prefix
        match = pattern.search(name)
        if match:
            unique_prefixes.add(match.group(0))
    
    # Return a sorted list of unique prefixes
    return sorted(unique_prefixes)

def extract_features_csp(raw, sfreq, labels, epoch_length=1.0):
    print("Entering extract_features_csp method.")

    # Define CSP
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # Get data from the Raw object and split into epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_length, preload=True, reject_by_annotation=False)
    epochs_data = epochs.get_data()

    # Ensure the number of epochs and labels match
    min_len = min(len(epochs_data), len(labels))
    epochs_data = epochs_data[:min_len]
    labels = labels[:min_len]

    # Fit CSP
    csp.fit(epochs_data, labels)

    # Transform the data using CSP to get the log-variance features
    csp_features = csp.transform(epochs_data)

    # Initialize a list to store combined features
    combined_features = []

    # Calculate time-domain features for each CSP-transformed epoch
    for epoch in epochs_data:
        epoch_features = []
        for channel_data in epoch:
            # Calculate traditional time-domain features for each channel
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            skew_val = skew(channel_data)
            rms_val = np.sqrt(np.mean(channel_data**2))
            variance_val = np.var(channel_data)
            peak_to_peak_val = np.ptp(channel_data)
            waveform_length_val = np.sum(np.abs(np.diff(channel_data)))

            # Append channel features to epoch features
            channel_features = [
                mean_val, std_val, skew_val, 
                rms_val, variance_val, 
                peak_to_peak_val, waveform_length_val
            ]
            epoch_features.extend(channel_features)

       
        # Append to the combined features list
        combined_features.append(epoch_features)

    lda = LDA()
    lda.fit(combined_features, labels[:len(epochs_data)])
    combined_features = lda.transform(combined_features)    

    # Create a DataFrame from the combined features
    features_df = pd.DataFrame(combined_features)

    # Add labels to the DataFrame
    features_df['label'] = labels

    return features_df

def csv_identification(file_paths,processed_data_keywords):
    global messages
    messages=[]
    csv_only=False

    for file_path in file_paths:
        csv_only=True
        
        # Determine the type of file and handle it accordingly
        if file_path.lower().endswith(('.csv', '.xls', '.xlsx', '.xlsm', '.xlsb')):
            raw_data, sfreq = read_eeg_file(file_path)
            # picks = random.sample(raw_data.ch_names, 10)
            # plot_raw_eeg(raw_data, title=f'EEG Data from {file_path}', picks=picks)

            if isinstance(raw_data, mne.io.RawArray):
                # Convert to DataFrame here
                data_df = raw_data.to_data_frame()
            
                identified_methods = identify_feature_extraction_methods(data_df, processed_data_keywords)
                if identified_methods:
                    message = f"The following feature extraction methods were identified: {', '.join(identified_methods)}"
                    messages.append(message)                    
                    # Identify calculation methods for each band of interest
                    

                else:
                    messages.append("There is no known extracted feature methods")  

            else:
                identified_methods = identify_feature_extraction_methods(raw_data, processed_data_keywords)
                if identified_methods:
                    message = f"The following feature extraction methods were identified: {', '.join(identified_methods)}"
                    messages.append(message)                    
                    
                else:
                    messages.append("There is no known extracted feature methods")  

        process_processed_data(raw_data)   #CHECK IT PLEASE
    return messages,csv_only    

 
def create_cnn_model(input_shape):
    model = Sequential()
    # Ensure kernel_size is less than or equal to the number of features
    kernel_size = min(3, input_shape[0])
    model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Or 'softmax' for multi-class classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():

    global all_features
    global labels_list
    
    all_features = pd.DataFrame()  # Initialize an empty DataFrame to hold all features
    labels_list=[]
    

    all_true_labels = []
    all_predictions = []
    subject_data = {}
    

    process_with_builtin_functions = False   #Toggling
    proces_with_builtin_accuracy= False
    csv_only= False
    edf_only=False

    accuracy_dict_rnd = defaultdict(list)
    accuracy_dict_svc = defaultdict(list)
    accuracy_dict_gbc = defaultdict(list)
    accuracy_dict_knn = defaultdict(list)

    total_accuracy_dict_rnd = {}
    total_accuracy_dict_svc = {}
    total_accuracy_dict_gbc = {}
    total_accuracy_dict_knn={}
    subject_session_counts = {}

    accuracy_dict = defaultdict(lambda: defaultdict(list))
    overall_accuracy_dict = defaultdict(dict)
    subject_scores = defaultdict(lambda: defaultdict(list))


    for file_path in file_paths:
        
        
        # Determine the type of file and handle it accordingly
        if file_path.lower().endswith(('.csv', '.xls', '.xlsx', '.xlsm', '.xlsb')):
            csv_only= True
            messages,csv_only=csv_identification(file_paths,processed_data_keywords)
            for message in messages:
                print("Message:", message)

        elif file_path.lower().endswith('.edf'):
            edf_only= True
            raw_data, sfreq = read_edf_eeg(file_path)
            # picks = random.sample(df_or_raw.ch_names, 10)
            # plot_raw_eeg(df_or_raw, title=f'EEG Data from {file_path}', picks=picks)
           #preprocessed_raw= preprocess_raw_eeg(raw_data, sfreq)
            features_df = extract_features_edf(raw_data,epoch_length=1.0)
            all_features = pd.concat([all_features, features_df], ignore_index=True)
            print("Last column:", all_features.iloc[:, -1])
            print("All features shape:", all_features.shape)


        elif file_path.lower().endswith('.mat'):

            filename = os.path.basename(file_path)  # Extract the filename from the file path
            subject_identifier = filename.split('_')[0] + '_' + filename.split('_')[1]  # sub-XXX_ses-XX

            
            if process_with_builtin_functions:
                mat_contents = scipy.io.loadmat(file_path)
                eeg_data = mat_contents.get('data')  # Replace 'data' with the actual key for data
                labels = mat_contents.get('labels')  # Replace 'labels' with the actual key for labels
                channel_names = [ "Fp1","GND", "Fp2","F7", "F3", "Fz", "F4", "F8",
                                "FC5", "FC1", "FC2", "FC6",
                                    "T3", "C3", "Cz", "C4","A1", "T4","A2"
                                    , "CP5", "CP1", "CP2","Ref", "CP6", "T5",
                                    "P3", "Pz", "P4", "T6", "PO3","PO4", "O1", "Oz", "O2"]
                

                if eeg_data is not None and labels is not None:
                    labels = labels.flatten()

                    eeg_data_filtered = np.array([bandPassFilter(trial, sampleRate=250, highpass=1, lowpass=40, order=2) for trial in eeg_data])

                    # Initialize the custom EEG data transformer
                    eeg_to_df_transformer = EEGToDataFrame()
                    
                    # Convert EEG data into DataFrame for tsfresh
                    eeg_data_df = eeg_to_df_transformer.transform(eeg_data_filtered)
                    print("EEG column:",eeg_data_df.columns)
                    
                    print('Number of samples after EEGToDataFrame:', eeg_data_df.shape[0])

                    data_as_array = eeg_data_df.values.T  # Assuming eeg_data_df is in the correct shape
                    info = mne.create_info(ch_names=channel_names, sfreq=250, ch_types='eeg')
                    raw = mne.io.RawArray(data_as_array, info)

                    extraction_settings = MinimalFCParameters()

                    # Extract features using tsfresh
                    # You should not get a TypeError if you use the function correctly
                    extracted_features = extract_features(eeg_data_df, column_id='id', column_sort='time', default_fc_parameters=extraction_settings)
                    
                    
                    # Impute missing values
                    imputed_features = impute(extracted_features)

                    
                    print('Number of samples after extract_features:', extracted_features.shape[0])
                    print('Number of samples after impute:', imputed_features.shape[0])
                    
                    final_labels = labels[imputed_features.index]

                    print('Number of features before selection:', imputed_features.shape[1])
                    # Select only relevant features based on the labels
                    relevant_features = select_features(imputed_features, pd.Series(final_labels))
                    print('Number of features after selection:', relevant_features.shape[1])
                    

                    print("Length label:", len(final_labels))


                    # Split the data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(imputed_features, final_labels, test_size=0.2, random_state=42)

                    print("eeg_data_filtered shape:", eeg_data_filtered.shape)
                    print("eeg_data_df shape:", eeg_data_df.shape)
                    print("extracted_features shape:", extracted_features.shape)
                    print("imputed_features shape:", imputed_features.shape)
                    print("relevant_features shape:", relevant_features.shape)
                    print("X_train shape:", X_train.shape)
                    print("y_train shape:", y_train.shape)

                    # Define a pipeline with modeling step only, as tsfresh has already done the scaling and feature selection
                    pipeline = Pipeline([
                        ('classifier', RandomForestClassifier())  # Modeling step
                    ])

                    # Train the model using the pipeline
                    pipeline.fit(X_train, y_train)

                    # Make predictions
                    y_pred = pipeline.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    print(f"Random Accuracy for {subject_identifier}: {accuracy * 100:.2f}%")

                    # Store the accuracy in accuracy_dict_rnd for RandomForestClassifier
                    accuracy_dict_rnd[subject_identifier].append(accuracy)

                    #SUPPORT VECTOR MACHINE

                    pipeline_svc = Pipeline([
                        ('classifier', SVC())  # Modeling step
                    ])

                    # Train the model using the pipeline
                    pipeline_svc.fit(X_train, y_train)

                    # Make predictions
                    y_pred_svc = pipeline_svc.predict(X_test)

                    accuracy_svc = accuracy_score(y_test, y_pred_svc)
                    print(f"SVM Accuracy for {subject_identifier}: {accuracy_svc * 100:.2f}%")

                    accuracy_dict_svc[subject_identifier].append(accuracy_svc)

                    #GRADIENT CLASSIFICATION

                    pipeline_gdn = Pipeline([
                        ('classifier', GradientBoostingClassifier())  # Modeling step
                    ])

                    # Train the model using the pipeline
                    pipeline_gdn.fit(X_train, y_train)

                    # Make predictions
                    y_pred_gdn = pipeline_gdn.predict(X_test)

                    accuracy_gdn = accuracy_score(y_test, y_pred_gdn)
                    print(f"Gradient Accuracy for {subject_identifier}: {accuracy_gdn * 100:.2f}%")

                    accuracy_dict_gbc[subject_identifier].append(accuracy_gdn)

                    #K-NN CLASSIFIER

                    pipeline_knn = Pipeline([
                        ('classifier', MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, alpha=0.001,
                    solver='sgd', random_state=21, tol=0.000000001))  # Modeling step
                    ])

                    # Train the model using the pipeline
                    pipeline_knn.fit(X_train, y_train)

                    # Make predictions
                    y_pred_knn = pipeline_knn.predict(X_test)

                    accuracy_knn = accuracy_score(y_test, y_pred_knn)
                    print(f"KNN Accuracy for {subject_identifier}: {accuracy_knn * 100:.2f}%")

                    accuracy_dict_knn[subject_identifier].append(accuracy_knn)


                    # Store true labels and predictions
                    all_true_labels.extend(y_test)
                    all_predictions.extend(y_pred)

                    if subject_identifier not in subject_data:
                       subject_data[subject_identifier] = {'true_labels': [], 'predictions': []}
                
                    # Append the true labels and predictions to the subject's entry in the dictionary
                    subject_data[subject_identifier]['true_labels'].extend(y_test)
                    subject_data[subject_identifier]['predictions'].extend(y_pred)
                    
                else:
                    print(f"Data or labels could not be found in {file_path}. Please check the file structure.")
            else:
                raw_data, sfreq, labels = read_mat_eeg(file_path) # Handling .mat file 
                 #fig = visualize_3d_brain_model(raw_data)
                preprocessed_raw = preprocess_raw_eeg(raw_data, 250)

                #features_df = extract_features_mat(preprocessed_raw, sfreq, labels,epoch_length=1.0)
                features_df = extract_features_csp(preprocessed_raw, sfreq, labels, epoch_length=1.0)


                if subject_identifier not in subject_data:
                    subject_data[subject_identifier] = {'features': pd.DataFrame(), 'labels': []}
                # Append features and labels to the subject's data
                subject_data[subject_identifier]['features'] = pd.concat([subject_data[subject_identifier]['features'], features_df], ignore_index=True)
                subject_data[subject_identifier]['labels'].extend(labels)  
               
        else:
            print(f"File {file_path} is not a recognized EEG file type.")
            continue

    if proces_with_builtin_accuracy: 
        print("Accuracy Dictionary:",accuracy_dict_rnd)
  
        # for subject, data in subject_data.items():
        #     print(f"Accuracy for {subject_identifier}: {accuracy * 100:.2f}%") 

        for subject_session in accuracy_dict_rnd:
            
            # Extract the subject identifier without the session
            subject_identifier = subject_session.rsplit('_', 1)[0]
            # Sum the accuracies for RandomForest
            total_accuracy_dict_rnd[subject_identifier] = total_accuracy_dict_rnd.get(subject_identifier, 0) + sum(accuracy_dict_rnd[subject_session])
            # Sum the accuracies for SVC
            total_accuracy_dict_svc[subject_identifier] = total_accuracy_dict_svc.get(subject_identifier, 0) + sum(accuracy_dict_svc[subject_session])
            # Sum the accuracies for GradientBoosting
            total_accuracy_dict_gbc[subject_identifier] = total_accuracy_dict_gbc.get(subject_identifier, 0) + sum(accuracy_dict_gbc[subject_session])

            total_accuracy_dict_knn[subject_identifier] = total_accuracy_dict_knn.get(subject_identifier, 0) + sum(accuracy_dict_knn[subject_session])

            # Count the number of sessions for each subject
            subject_session_counts[subject_identifier] = subject_session_counts.get(subject_identifier, 0) + len(accuracy_dict_rnd[subject_session])

        # Now calculate the average accuracy for each classifier for each subject
        average_accuracy_dict_rnd = {subject: total_accuracy / subject_session_counts[subject] for subject, total_accuracy in total_accuracy_dict_rnd.items()}
        average_accuracy_dict_svc = {subject: total_accuracy / subject_session_counts[subject] for subject, total_accuracy in total_accuracy_dict_svc.items()}
        average_accuracy_dict_gbc = {subject: total_accuracy / subject_session_counts[subject] for subject, total_accuracy in total_accuracy_dict_gbc.items()}
        average_accuracy_dict_knn = {subject: total_accuracy / subject_session_counts[subject] for subject, total_accuracy in total_accuracy_dict_knn.items()}


        # Print the average accuracy for each subject
        for subject in average_accuracy_dict_rnd.keys():
            print(f"Subject {subject} Accuracy RandomForest: {average_accuracy_dict_rnd[subject] * 100:.2f}%")
            print(f"Subject {subject} Accuracy SVC: {average_accuracy_dict_svc[subject] * 100:.2f}%")
            print(f"Subject {subject} Accuracy GradientBoosting: {average_accuracy_dict_gbc[subject] * 100:.2f}%")
            print(f"Subject {subject} Accuracy KNN: {average_accuracy_dict_knn[subject] * 100:.2f}%")

           
    else:        
    # This part should be outside (below) the for loop that goes through file_paths

        # Loop over your subject_data to perform cross-validation for each subject
        for subject_identifier, data in subject_data.items():
            X = data['features'].iloc[:, :-1]  # Features: all columns except the last
            y = LabelEncoder().fit_transform(data['features'].iloc[:, -1])  # Labels: the last column
            X_scaled = StandardScaler().fit_transform(X)  # Standardize features

           # Assuming X_scaled.shape is (n_samples, n_features)
            n_samples, n_features = X_scaled.shape

            # Check if n_features is less than the desired kernel_size
            if n_features < 3:
                # Code to handle the situation, e.g., reducing kernel_size or feature engineering
                # For example, reducing kernel_size:
                kernel_size = n_features  # Set kernel size equal to the number of features
            else:
                kernel_size = 3  # Or any other value greater than or equal to n_features

            # Reshape input data for the CNN
            X_reshaped = X_scaled.reshape((n_samples, n_features, 1))

            # Now define the input shape for the CNN model
            input_shape = (n_features, 1)  # Revised input shape
           
            cnn_classifier = KerasClassifier(build_fn=create_cnn_model, input_shape=input_shape, epochs=10, batch_size=10, verbose=0)

            # Define classifiers
            classifiers = {
                'RandomForest': RandomForestClassifier(),
                'SVC': SVC(),
                'GradientBoosting': GradientBoostingClassifier(),
                'KNN':KNeighborsClassifier(n_neighbors=5) ,
                'CNN': cnn_classifier
            }

            # Reshape the input data for CNN
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

            # Perform cross-validation and collect accuracy for each classifier
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            for clf_name, clf in classifiers.items():
                if clf_name=='CNN':
                     cv_scores = cross_val_score(clf, X_reshaped, y, cv=skf)
                else:
                    cv_scores = cross_val_score(clf, X_scaled, y, cv=skf)     
                
                # Get the base subject identifier without session (e.g., "sub-001")
                base_subject_identifier = subject_identifier.rsplit('_', 1)[0]
                subject_scores[base_subject_identifier][clf_name].extend(cv_scores)

        # Now calculate the overall accuracy for each subject by taking the mean of the aggregated scores
        overall_accuracy_dict = defaultdict(dict)
        for base_subject_identifier, clf_scores in subject_scores.items():
            for clf_name, scores in clf_scores.items():
                overall_accuracy = np.mean(scores)
                overall_accuracy_dict[base_subject_identifier][clf_name] = overall_accuracy
                print(f"Overall accuracy for {clf_name} on subject {base_subject_identifier}: {overall_accuracy * 100:.2f}%")

        # for subject_identifier, data in subject_data.items():
        #     label_encoder = LabelEncoder()
            
        #     X = data['features'].iloc[:, :-1]  # features: all columns except the last
        #     y_binned = label_encoder.fit_transform(data['features'].iloc[:, -1])  # labels: the last column

        #     # Perform a train-test split
        #     X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2)

        #     # Standardize the features
        #     scaler = StandardScaler()
        #     X_train = scaler.fit_transform(X_train)
        #     X_test = scaler.transform(X_test)

        #     # Fit the model (you can choose SVC, RandomForestClassifier, or GradientBoostingClassifier as shown in your code)
        #     clf_rnd = RandomForestClassifier()
        #     clf_rnd.fit(X_train, y_train)

        #     # Predict on the test data
        #     y_pred_rnd = clf_rnd.predict(X_test)

        #     # Calculate the accuracy
        #     acc_rnd = accuracy_score(y_test, y_pred_rnd)
        #     print(f"Accuracy RandomForest for subject {subject_identifier}: {acc_rnd * 100:.2f}%")
        #     accuracy_dict_rnd[subject_identifier].append(acc_rnd)

        #     # Fit the model (you can choose SVC, RandomForestClassifier, or GradientBoostingClassifier as shown in your code)
        #     clf_SVC = SVC()
        #     clf_SVC.fit(X_train, y_train)

        #     # Predict on the test data
        #     y_pred_SVC = clf_SVC.predict(X_test)

        #     # Calculate the accuracy
        #     acc_svc = accuracy_score(y_test, y_pred_SVC)
        #     print(f"Accuracy SVC for subject {subject_identifier}: {acc_svc * 100:.2f}%")  
        #     accuracy_dict_svc[subject_identifier].append(acc_svc)

        #     clf_gbc = GradientBoostingClassifier()
        #     clf_gbc.fit(X_train, y_train)

        #     # Predict on the test data
        #     y_pred_gbc = clf_gbc.predict(X_test)

        #     # Calculate the accuracy
        #     acc_gbc = accuracy_score(y_test, y_pred_gbc)
        #     print(f"Accuracy Gradient for subject {subject_identifier}: {acc_gbc * 100:.2f}%")
        #     accuracy_dict_gbc[subject_identifier].append(acc_gbc)

        #     clf_knn = MLPClassifier(hidden_layer_sizes=(40,), max_iter=500, alpha=0.001,
        #             solver='sgd', tol=0.000000001)
        #     clf_knn.fit(X_train, y_train)

        #     # Predict on the test data
        #     y_pred_knn = clf_knn.predict(X_test)

        #     # Calculate the accuracy
        #     acc_knn = accuracy_score(y_test, y_pred_knn)
        #     print(f"Accuracy Gradient for subject {subject_identifier}: {acc_knn * 100:.2f}%")
        #     accuracy_dict_knn[subject_identifier].append(acc_knn)  
        
        # for subject_session in accuracy_dict_rnd:
            
        #     # Extract the subject identifier without the session
        #     subject_identifier = subject_session.rsplit('_', 1)[0]
        #     # Sum the accuracies for RandomForest
        #     total_accuracy_dict_rnd[subject_identifier] = total_accuracy_dict_rnd.get(subject_identifier, 0) + sum(accuracy_dict_rnd[subject_session])
        #     # Sum the accuracies for SVC
        #     total_accuracy_dict_svc[subject_identifier] = total_accuracy_dict_svc.get(subject_identifier, 0) + sum(accuracy_dict_svc[subject_session])
        #     # Sum the accuracies for GradientBoosting
        #     total_accuracy_dict_gbc[subject_identifier] = total_accuracy_dict_gbc.get(subject_identifier, 0) + sum(accuracy_dict_gbc[subject_session])

        #     total_accuracy_dict_knn[subject_identifier] = total_accuracy_dict_knn.get(subject_identifier, 0) + sum(accuracy_dict_knn[subject_session])

        #     # Count the number of sessions for each subject
        #     subject_session_counts[subject_identifier] = subject_session_counts.get(subject_identifier, 0) + len(accuracy_dict_rnd[subject_session])

        # # Now calculate the average accuracy for each classifier for each subject
        # average_accuracy_dict_rnd = {subject: total_accuracy / subject_session_counts[subject] for subject, total_accuracy in total_accuracy_dict_rnd.items()}
        # average_accuracy_dict_svc = {subject: total_accuracy / subject_session_counts[subject] for subject, total_accuracy in total_accuracy_dict_svc.items()}
        # average_accuracy_dict_gbc = {subject: total_accuracy / subject_session_counts[subject] for subject, total_accuracy in total_accuracy_dict_gbc.items()}
        # average_accuracy_dict_knn = {subject: total_accuracy / subject_session_counts[subject] for subject, total_accuracy in total_accuracy_dict_knn.items()}


        # # Print the average accuracy for each subject
        # for subject in average_accuracy_dict_rnd.keys():
        #     print(f"Subject {subject} Accuracy RandomForest: {average_accuracy_dict_rnd[subject] * 100:.2f}%")
        #     print(f"Subject {subject} Accuracy SVC: {average_accuracy_dict_svc[subject] * 100:.2f}%")
        #     print(f"Subject {subject} Accuracy GradientBoosting: {average_accuracy_dict_gbc[subject] * 100:.2f}%")
        #     print(f"Subject {subject} Accuracy Knn: {average_accuracy_dict_knn[subject] * 100:.2f}%")


    # if process_with_builtin_functions and :
    #     print("Length of all_features: ", len(all_predictions))
    #     print("Length of labels_list: ", len(all_true_labels))

    #     overall_accuracy = accuracy_score(all_true_labels, all_predictions)
    #     print(f'All model accuracy across all files: {overall_accuracy * 100:.2f}%')

    # else:
    if csv_only | edf_only:        
        label_encoder = LabelEncoder()
           
        print("Length of all_features: ", len(all_features))
        print("Length of labels_list: ", len(labels_list))   
        
        X = all_features.iloc[:, :-1]  # features: all columns except the last
        y_binned = label_encoder.fit_transform(all_features.iloc[:, -1])  # labels: the last column


        # clf_svc = SVC()
        # scores_svc = cross_val_score(clf_svc, X, y_binned, cv=5)  # cv=5 for 5-fold cross-validation
        # print("SVM cross-validation accuracy:", np.mean(scores_svc))

        # # For Random Forest
        # clf_rf = RandomForestClassifier(n_estimators=50, random_state=42)
        # scores_rf = cross_val_score(clf_rf, X, y_binned, cv=5)
        # print("Random Forest cross-validation accuracy:", np.mean(scores_rf))

        # # For Gradient Boosting
        # clf_gbc = GradientBoostingClassifier(n_estimators=50, random_state=42)
        # scores_gbc = cross_val_score(clf_gbc, X, y_binned, cv=5)
        # print("Gradient Boosting cross-validation accuracy:", np.mean(scores_gbc))


        print("Y binned:", y_binned)
        print("Last column: ",all_features.iloc[:, -1])

        threshold = np.percentile(y_binned, 50)  # This is essentially the same as the median
        y = pd.cut(y_binned, bins=[-np.inf, threshold, np.inf], labels=[0, 1])

        print("Y head:",y_binned[:5])  # To see the first few entries
        print("Y unique:",np.unique(y_binned))  # To see the unique values

        # Perform a train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2)

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test= scaler.transform(X_test)

        clf = SVC()
        clf.fit(X_train, y_train)
        y_pred_svc = clf.predict(X_test)
        acc_svc = round(accuracy_score(y_test,y_pred_svc)*100,2)
        
        

        # Initialize the classifier
        clf_rf = RandomForestClassifier(n_estimators=50)

        # Fit the classifier to the training data
        clf_rf.fit(X_train, y_train)

        # Predict on the test data
        y_pred_rf = clf_rf.predict(X_test)

        # Calculate the accuracy
        acc_rf = round(accuracy_score(y_test,y_pred_rf)*100,2)
        

        clf_gbc = GradientBoostingClassifier(n_estimators=50)

        # Fit the classifier to the training data
        clf_gbc.fit(X_train, y_train)

        # Predict on the test data
        y_pred_gbc = clf_gbc.predict(X_test)

        # Calculate the accuracy on the training set
        acc_gbc = round(accuracy_score(y_test,y_pred_gbc)*100,2)

        clf_knn = KNeighborsClassifier(n_neighbors=5)  # You can tune the n_neighbors parameter.

        # Fit the classifier to the training data
        clf_knn.fit(X_train, y_train)

        # Predict on the test data
        y_pred_knn = clf_knn.predict(X_test)

        # Calculate the accuracy
        acc_knn = round(accuracy_score(y_test, y_pred_knn) * 100, 2)
         

        # Reshape data for 1D CNN input (batch_size, steps, input_dimension)
        X_train_cnn = np.expand_dims(X_train, axis=2)
        X_test_cnn = np.expand_dims(X_test, axis=2)

        # Convert labels to categorical (one-hot encoding)
        y_train_categorical = to_categorical(y_train)
        y_test_categorical = to_categorical(y_test)

        # Define the 1D CNN model
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(y_train_categorical.shape[1], activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Fit the model
        model.fit(X_train_cnn, y_train_categorical, epochs=10, batch_size=32, verbose=1)

        # Evaluate the model
        _, accuracy = model.evaluate(X_test_cnn, y_test_categorical, verbose=0)

        print("SVM Accuracy is:",(str(acc_svc)+'%'))
        print("Random Forest accuracy is:", (str(acc_rf) + '%'))
        print("Gradient Boosting Classifier accuracy is:", (str(acc_gbc) + '%')) 
        print("KNN Accuracy is:", (str(acc_knn) + '%'))
        print(f'CNN 1D - Accuracy: {accuracy * 100:.2f}%')


        

        # print("Classes distribution in training set:", np.unique(y_train, return_counts=True))
        # print("Classes distribution in testing set:", np.unique(y_test, return_counts=True))

        # print("Unique classes in y_train:", np.unique(y_train))
        # print("Unique classes in y_test:", np.unique(y_test))  


   


if __name__ == '__main__':
    main()




# # Iterate through each file path in the list of file paths
# for file_path in file_paths:
#     if file_path.endswith('.csv'):
        
#         # Read the CSV file and determine if it's processed
#         df_or_raw, sfreq = read_eeg_file(file_path)
        
#         if isinstance(df_or_raw, pd.DataFrame):
#             print("The data appears to be processed and features are already extracted.")
#             identified_methods = identify_feature_extraction_methods(df_or_raw, processed_data_keywords)
#             if identified_methods:
#                 print(f"The following feature extraction methods were identified: {', '.join(identified_methods)}")
                
#                 # Identify calculation methods for each band of interest
#                 for band in ['theta', 'alpha', 'beta', 'gamma']:
#                     band_columns = [col for col in df_or_raw.columns if band in col.lower()]
#                     for col in band_columns:
#                         method = identify_calculation_method(col, processed_data_keywords)
#                         print(f"The {band} band feature '{col}' is calculated using: {method}")
            
#                 epochs_df = create_epochs_from_preprocessed_features(df_or_raw, epoch_length=1.0, sfreq=500)
#                 print("Epochs DataFrame created with preprocessed features.")
#                 print(epochs_df.head()) 
#                 all_features = pd.concat([all_features, epochs_df], ignore_index=True)
#                 print("All features shape:", all_features.shape)
                     
#             else:
#                 print("No known feature extraction methods were identified.")
#         elif df_or_raw is not None:
#             # Handle the raw data
#             print("The data appears to be raw.")
#             print(f"CSV file {file_path} read successfully with sampling frequency {sfreq} Hz.")
#             first_five_channels = df_or_raw.ch_names[:10]  # Get the names of the first five channels
#             plot_raw_eeg(df_or_raw, title=f'Raw EEG Data: {file_path}', picks=first_five_channels, separate_channels=True)
#             raw_preprocessed = preprocess_raw_eeg(df_or_raw, sfreq)
#             plot_raw_eeg(raw_preprocessed, title=f'Preprocessed EEG Data: {file_path}', picks=first_five_channels, separate_channels=True)
#             if raw_preprocessed is not None:
#                 fft_features = extract_features(raw_preprocessed)
#                 print(fft_features)
#                 features_df = extract_features_as_df(raw_preprocessed)
#                 all_features = pd.concat([all_features, features_df], ignore_index=True)
#                 print("All features shape:", all_features.shape)


#     elif file_path.lower().endswith('.edf'):
#         raw, sfreq = read_edf_eeg(file_path)
        
#         if raw is not None:
#             print(f"EDF file {file_path} read successfully with sampling frequency {sfreq} Hz.")
#             raw_preprocessed = preprocess_raw_eeg(raw, sfreq)
#             first_five_channels = raw.ch_names[:10]  # Get the names of the first five channels
#            # plot_raw_eeg(raw_preprocessed, title=f'Preprocessed EEG Data: {file_path}', picks=first_five_channels, separate_channels=True)
#             if raw_preprocessed is not None:
#                 fft_features = extract_features(raw_preprocessed)
#                 print(fft_features)
#                 features_df = extract_features_as_df(raw_preprocessed)
#                 all_features = pd.concat([all_features, features_df], ignore_index=True)
#                 print("All features shape:", all_features.shape)
#             else:
#                 print("No known feature extraction methods were identified.")
#         elif df_or_raw is not None:
#             # Handle the raw data
#             print("The data appears to be raw.")
#             print(f"CSV file {file_path} read successfully with sampling frequency {sfreq} Hz.")
#             first_five_channels = df_or_raw.ch_names[:10]  # Get the names of the first five channels
#             plot_raw_eeg(df_or_raw, title=f'Raw EEG Data: {file_path}', picks=first_five_channels, separate_channels=True)
#             raw_preprocessed = preprocess_raw_eeg(df_or_raw, sfreq)
#             plot_raw_eeg(raw_preprocessed, title=f'Preprocessed EEG Data: {file_path}', picks=first_five_channels, separate_channels=True)
#             if raw_preprocessed is not None:
#                 fft_features = extract_features(raw_preprocessed)
#                 print(fft_features)
#                 features_df = extract_features_as_df(raw_preprocessed)
#                 all_features = pd.concat([all_features, features_df], ignore_index=True)
#                 print("All features shape:", all_features.shape)           
   
#     else:
#         print(f"File {file_path} is not a recognized EEG file type.")
