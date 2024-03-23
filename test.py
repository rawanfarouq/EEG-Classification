from glob import glob
import os    #reading file paths or directories.
import mne   # For working with EEG data
import numpy as np
import pandas as pd
import random
import scipy.io
import h5py
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer
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
from eeglib.preprocessing import bandPassFilter
from mpl_toolkits.mplot3d import Axes3D
from nilearn import plotting
import plotly.graph_objects as go
import cProfile





downloads_folder = os.path.expanduser("~/Downloads")  # Get the path to the "Downloads" directory
dataverse_files_folder = os.path.join(downloads_folder, "edf")  # Combine with the folder name

# Use glob to get a list of file paths matching a specific pattern
file_paths = glob(os.path.join(dataverse_files_folder, '*.*'))  #Getting all files with any extension


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

        # print(raw.info)  # Summarizes the info structure of the raw object
        # print(raw.ch_names)  # List of all channel names
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



def preprocess_channel_data(channel_data, sfreq, l_freq_hp, h_freq_lp, notch_freqs):
    # High-pass filtering to remove slow drifts
    channel_data = filter_data(channel_data, sfreq, l_freq=l_freq_hp, h_freq=None, verbose=False)
    
    # Low-pass filtering to remove high-frequency noise
    channel_data = filter_data(channel_data, sfreq, l_freq=None, h_freq=h_freq_lp, verbose=False)
    
    # Notch filter to remove power line noise at 50 Hz or 60 Hz and its harmonics
    for notch_freq in notch_freqs:
        channel_data = filter_data(channel_data, sfreq, l_freq=notch_freq - 0.1, h_freq=notch_freq + 0.1, method='iir', verbose=False)
    
    return channel_data



def preprocess_raw_eeg(raw, sfreq):

     # Get the data from the Raw object
    eeg_data = raw.get_data()

    preprocessed_data = np.empty(eeg_data.shape)
    

    # Notch filter to remove power line noise at 50 Hz or 60 Hz and its harmonics
    notch_freqs = np.arange(50, sfreq / 2, 50)  # Assuming 50 Hz power line noise

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
        preprocessed_data[i] = preprocess_channel_data(channel, sfreq, l_freq_hp=0.5, h_freq_lp=40.0, notch_freqs=notch_freqs)
    
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
       

        # Frequency-domain features
        fft_vals = rfft(epoch)
        psd_vals = np.abs(fft_vals) ** 2
        freq_res = np.fft.rfftfreq(n=len(epoch), d=1.0 / sfreq)
        

        # Welch's method to estimate power spectral density
        f, psd_welch = welch(epoch, sfreq)

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
            'average': average_val,  # This is redundant as it's the same as mean
        }
        # Calculate power in each frequency band
       
        for band, (fmin, fmax) in bands.items():
            # Find the power spectral density within the band
            band_inds = np.where((f >= fmin) & (f <= fmax))[0]
            band_psd = psd_welch[band_inds]
            
            # Find the peak frequency and its power
            peak_freq = f[band_inds][np.argmax(band_psd)]
            peak_power = np.max(band_psd)
            
            features[f'{band}_peak_freq'] = peak_freq
            features[f'{band}_peak_power'] = peak_power
        


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


def process_processed_data(data, sfreq):
    global all_features
    print("The data appears to be processed and features are already extracted.")
    if isinstance(data, pd.DataFrame):
        # If data is a DataFrame, pass it directly
        epochs_df = create_epochs_from_preprocessed_features(data, epoch_length=1.0, sfreq=500)
        print("Epochs DataFrame created with preprocessed features.")
        print(epochs_df.head()) 
        all_features = pd.concat([all_features, epochs_df], ignore_index=True)
    elif isinstance(data, mne.io.BaseRaw):
        # If data is a Raw object, convert it to a DataFrame first
        data_df = data.to_data_frame()
        epochs_df = create_epochs_from_preprocessed_features(data_df, epoch_length=1.0, sfreq=sfreq)
        print(epochs_df.head()) 
        all_features = pd.concat([all_features, epochs_df], ignore_index=True)
    else:
        raise ValueError("Data must be a pandas DataFrame or MNE Raw object.")
    

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
      



def main():

    global all_features
    global labels_list
    
    all_features = pd.DataFrame()  # Initialize an empty DataFrame to hold all features
    labels_list=[]
    

    all_true_labels = []
    all_predictions = []
    subject_data = {}
    

    process_with_builtin_functions = True   #Toggling
    proces_with_builtin_accuracy= True
    csv_and_edf= False

    for file_path in file_paths:
        
        
        # Determine the type of file and handle it accordingly
        if file_path.lower().endswith(('.csv', '.xls', '.xlsx', '.xlsm', '.xlsb')):
            csv_and_edf= True
            raw_data, sfreq = read_eeg_file(file_path)
            # picks = random.sample(df_or_raw.ch_names, 10)
            # plot_raw_eeg(df_or_raw, title=f'EEG Data from {file_path}', picks=picks)
            process_processed_data(raw_data, sfreq)

        elif file_path.lower().endswith('.edf'):
            csv_and_edf= True
            raw_data, sfreq = read_edf_eeg(file_path)
            # picks = random.sample(df_or_raw.ch_names, 10)
            # plot_raw_eeg(df_or_raw, title=f'EEG Data from {file_path}', picks=picks)
            #process_processed_data(raw_data, sfreq)

        elif file_path.lower().endswith('.mat'):

            filename = os.path.basename(file_path)  # Extract the filename from the file path
            subject_identifier = filename.split('_')[0]  # sub-XXX_ses-XX

            
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
                        ('classifier', RandomForestClassifier(random_state=42))  # Modeling step
                    ])

                    # Train the model using the pipeline
                    pipeline.fit(X_train, y_train)

                    # Make predictions
                    y_pred = pipeline.predict(X_test)

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
                # Extract features from the preprocessed raw data and include labels
                features_df = extract_features_mat(preprocessed_raw, sfreq, labels,epoch_length=1.0)
                if subject_identifier not in subject_data:
                    subject_data[subject_identifier] = {'features': pd.DataFrame(), 'labels': []}
                # Append features and labels to the subject's data
                subject_data[subject_identifier]['features'] = pd.concat([subject_data[subject_identifier]['features'], features_df], ignore_index=True)
                subject_data[subject_identifier]['labels'].extend(labels)  
               
        else:
            print(f"File {file_path} is not a recognized EEG file type.")
            continue

    if proces_with_builtin_accuracy:    
        for subject, data in subject_data.items():
            overall_accuracy = accuracy_score(data['true_labels'], data['predictions'])
            print(f"Overall accuracy for {subject}: {overall_accuracy * 100:.2f}%")  

    else:        
    # This part should be outside (below) the for loop that goes through file_paths
        for subject_identifier, data in subject_data.items():
            label_encoder = LabelEncoder()
            
            X = data['features'].iloc[:, :-1]  # features: all columns except the last
            y_binned = label_encoder.fit_transform(data['features'].iloc[:, -1])  # labels: the last column

            # Perform a train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=42)

            # Standardize the features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Fit the model (you can choose SVC, RandomForestClassifier, or GradientBoostingClassifier as shown in your code)
            clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42)
            clf_rnd.fit(X_train, y_train)

            # Predict on the test data
            y_pred_rnd = clf_rnd.predict(X_test)

            # Calculate the accuracy
            acc = accuracy_score(y_test, y_pred_rnd)
            print(f"Accuracy RandomForest for subject {subject_identifier}: {acc * 100:.2f}%")

            # Fit the model (you can choose SVC, RandomForestClassifier, or GradientBoostingClassifier as shown in your code)
            clf_SVC = SVC()
            clf_SVC.fit(X_train, y_train)

            # Predict on the test data
            y_pred_SVC = clf_SVC.predict(X_test)

            # Calculate the accuracy
            acc = accuracy_score(y_test, y_pred_SVC)
            print(f"Accuracy SVC for subject {subject_identifier}: {acc * 100:.2f}%")  

            clf_gbc = GradientBoostingClassifier(n_estimators=50, random_state=42)
            clf_gbc.fit(X_train, y_train)

            # Predict on the test data
            y_pred_gbc = clf_gbc.predict(X_test)

            # Calculate the accuracy
            acc = accuracy_score(y_test, y_pred_gbc)
            print(f"Accuracy Gradient for subject {subject_identifier}: {acc * 100:.2f}%")  




    # if process_with_builtin_functions and :
    #     print("Length of all_features: ", len(all_predictions))
    #     print("Length of labels_list: ", len(all_true_labels))

    #     overall_accuracy = accuracy_score(all_true_labels, all_predictions)
    #     print(f'All model accuracy across all files: {overall_accuracy * 100:.2f}%')

    # else:
    if csv_and_edf:        
        label_encoder = LabelEncoder()
           
        print("Length of all_features: ", len(all_features))
        print("Length of labels_list: ", len(labels_list))   
        
        X = all_features.iloc[:, :-1]  # features: all columns except the last
        y_binned = label_encoder.fit_transform(all_features.iloc[:, -1])  # labels: the last column


        print("Y binned:", y_binned)
        print("Last column: ",all_features.iloc[:, -1])

        threshold = np.percentile(y_binned, 50)  # This is essentially the same as the median
        y = pd.cut(y_binned, bins=[-np.inf, threshold, np.inf], labels=[0, 1])

        print("Y head:",y)  # To see the first few entries
        print("Y unique:",y.unique())  # To see the unique values

        # Perform a train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test= scaler.transform(X_test)

        clf = SVC()
        clf.fit(X_train, y_train)
        y_pred_svc = clf.predict(X_test)
        acc_svc = round(clf.score(X_train, y_train) * 100, 2)
        print("SVM Accuracy is:",(str(acc_svc)+'%'))

        # Initialize the classifier
        clf_rf = RandomForestClassifier(n_estimators=50, random_state=42)

        # Fit the classifier to the training data
        clf_rf.fit(X_train, y_train)

        # Predict on the test data
        y_pred_rf = clf_rf.predict(X_test)

        # Calculate the accuracy
        acc_rf = round(clf_rf.score(X_train, y_train) * 100, 2)
        print("Random Forest accuracy is:", (str(acc_rf) + '%'))

        clf_gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=42)

        # Fit the classifier to the training data
        clf_gbc.fit(X_train, y_train)

        # Predict on the test data
        y_pred_gbc = clf_gbc.predict(X_test)

        # Calculate the accuracy on the training set
        acc_gbc = round(clf_gbc.score(X_train, y_train) * 100, 2)
        print("Gradient Boosting Classifier accuracy is:", (str(acc_gbc) + '%'))   

        print("Classes distribution in training set:", np.unique(y_train, return_counts=True))
        print("Classes distribution in testing set:", np.unique(y_test, return_counts=True))

        print("Unique classes in y_train:", np.unique(y_train))
        print("Unique classes in y_test:", np.unique(y_test))  


   


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
