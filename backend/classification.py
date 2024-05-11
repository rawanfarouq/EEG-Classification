from glob import glob
import os    #reading file paths or directories.
import mne   # For working with EEG data
import numpy as np
import pandas as pd
import random
import scipy.io
import ipywidgets as widgets
import h5py
import re
import matplotlib.pyplot as plt
from IPython.display import display
from PyQt5.QtWidgets import QApplication
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_validate,GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score,precision_score,f1_score,recall_score,make_scorer,confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer, normalize, label_binarize
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.datasets import make_classification
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
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from collections import defaultdict, Counter
from time import sleep
from joblib import Parallel,delayed
import tensorflow as tf
from flask import session
from joblib import dump, load
import stat

downloads_folder = os.path.expanduser("~/Downloads")  # Get the path to the "Downloads" directory
dataverse_files_folder = os.path.join(downloads_folder, "mat2")  # Combine with the folder name

# Use glob to get a list of file paths matching a specific pattern
file_paths = glob(os.path.join(dataverse_files_folder, '*.*'))  # Getting all files with any extension 
label_conditions_path = os.path.join(downloads_folder, 'xlabel_conditions.txt')




all_features = pd.DataFrame()
labels_list=[]
label_encoder = LabelEncoder()
progress_updates=[]
all_bads = set()
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
subject_data = {}
labels=[]
features_df=pd.DataFrame()
features_message=[]
accuracy_mat=[]
accuracy_mat_svc=[]
accuracy_mat_random=[]
accuracy_mat_logistic=[]
accuracy_mat_knn=[]
accuracy_mat_cnn=[]
label_to_condition_mapping = {}
label_conditions={}
accuracy_svc_csv={}
result_svc_csv={}
accuracy_random_csv={}
result_random_csv={}
accuracy_logistic_csv={}
result_logistic_csv={}
accuracy_knn_csv={}
result_knn_csv={}
accuracy_cnn_csv={}
result_cnn_csv={}
model_svc_csv_prediction={}
model_random_csv_prediction={}
model_logistic_csv_prediction={}
model_knn_csv_prediction={}
model_cnn_csv_prediction={}
processed_data_keywords = [
    'feature', 'power', 'mean', 'stddev', 'variance','peak','crestfactor',
    'entropy', 'fft', 'frequency', 'band', 'delta', 
    'theta', 'alpha', 'beta', 'gamma','max','min','avgmax','avgmin','skew','kurt',
    'median','var','energy','sigma','coefficient','D0','D1','D2','KFD','PFD','HFD','LZC',
    'CTM','AMI','RR','determinant','det','Lam','avg','average','Lmax','Vmin','TT','divergence',
    'time frequency distributions','TFD','fast fourier transformation','EM',
    'eigenvector methods','WT','wavelet transform','correlate','rms','p2p'
]

model_directory = 'saved_models'
os.makedirs(model_directory, exist_ok=True)
model_svc = os.path.join(model_directory, 'svc_model.joblib')
model_rf = os.path.join(model_directory, 'rf_model.joblib')
model_lr = os.path.join(model_directory, 'lr_model.joblib')
model_knn = os.path.join(model_directory, 'knn_model.joblib')
model_cnn = os.path.join(model_directory, 'cnn_model.joblib')
encoder_path = os.path.join(model_directory, 'label_encoder.joblib')
scaler_path = os.path.join(model_directory, 'scaler.joblib')
model_directory_mat = 'saved_models_mat'
os.makedirs(model_directory_mat, exist_ok=True)

label_descriptions = {
    1: 'moving left hand',
    2: 'moving right hand'
}
trial_labels_sentences_array = []
labels_array = [] 
result_predict_train=[]
accuracy_rf=0
all_subjects_predictions = []
cleanliness_messages = []

# for file_path in file_paths:
#     filename = os.path.basename(file_path)  # Extract the filename from the file path
#     subject_identifier = filename.split('_')[0] + '_' + filename.split('_')[1]  # sub-XXX_ses-XX



def read_eeg_file(file_path):
    try:
        # Create a list to hold messages about data cleanliness
        cleanliness_messages = []
        global df


        print(f"Entered read_eeg_file with {file_path}")

        # Check the file extension and read the file into a DataFrame accordingly
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx', '.xlsm', '.xlsb')):
            df = pd.read_excel(file_path)  # This reads the first sheet by default 

        else:
            print(f"Unsupported file type for {file_path}.")
            return None, None,None
        
        

        print(f"Number of samples (time points) in the recording: {len(df)}")
        print("DataFrame shape:" ,df.shape)
        cleanliness_messages.append(f"Number of samples (time points) in the recording: {len(df)}")
        cleanliness_messages.append(f"DataFrame shape: {df.shape}")
        


        # unique_prefixes = get_unique_prefixes(df.columns)
        # print("Unique prefixes:", unique_prefixes)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        label_cols = [col for col in df.columns if col.lower() in ['label', 'labels']]
        columns_to_check = numeric_cols + label_cols

         # Check for NaN values
        if df[columns_to_check].isnull().values.any():
            cleanliness_messages.append("Data contains NaN values.")
            print("Data contains NaN values.")
        else:
            cleanliness_messages.append("Data does not contain NaN values")
            print("Data does not contain NaN values")

        # Check for duplicate rows
        if df[columns_to_check].duplicated().any():
            cleanliness_messages.append("Data contains duplicate rows.")
            print("Data contains duplicate rows.")
        else:
            cleanliness_messages.append("Data does not contain duplicate rows")
            print("Data does not contain duplicate rows")

        # Check for columns with all zero variance (flatline signals)
        if df[numeric_cols].var(axis=0).eq(0).any():
            cleanliness_messages.append("Data contains flatline signals.")
            print("Data contains flatline signals.")
        else:
            cleanliness_messages.append("Data does not contain flatline signals")
            print("Data does not contain flatline signals")

        # Remove non-numeric columns except 'label' or 'labels'
        for col in df.select_dtypes(exclude=[np.number]).columns:
            if col not in label_cols:
                print(f"Column '{col}' is non-numeric and will be removed.")
                df.drop(col, axis=1, inplace=True)
                
        print(df.head()) 
        print("DataFrame shape:" ,df.shape)
        cleanliness_messages.append(f"DataFrame Head: {df.head()}")
        

        # Check if the first column is non-numeric and should be used as the index
        if not np.issubdtype(df.iloc[:, 0].dtype, np.number):
            df.set_index(df.columns[0], inplace=True)
        
        print(df.head())  # Displays the first 5 rows of the DataFrame
        print("DataFrame shape:" ,df.shape)
        

        # Check for a 'timestamp' column and calculate sampling frequency
        timestamp_present = 'timestamp' in df.columns
        if timestamp_present:
            time_diffs = np.diff(df['timestamp'].values)
            if np.any(time_diffs == 0):
                print("Duplicate timestamps found. Sampling frequency cannot be determined.")
            avg_sampling_period = np.mean(time_diffs)
            sfreq = 1 / avg_sampling_period
            print("sfreq calculated from timestamps:", sfreq)
        else:
            sfreq=250

        print("sfreq:",sfreq) 

        if check_processed_from_columns(df, processed_data_keywords):
            print("The data appears to be processed.")
            cleanliness_messages.append(f"Sampling frequency is determined: {sfreq}")
            return df, sfreq,cleanliness_messages # Return the DataFrame and None for sfreq

        eeg_data = df.values.T

        # Create MNE info structure
        ch_names = df.columns.tolist()
        ch_types = ['eeg'] * len(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        # Create RawArray
        raw = mne.io.RawArray(data=eeg_data, info=info)
        print(f"Number of samples read: {len(raw)}")
        print("Sampling frequency is determined:",sfreq)
        cleanliness_messages.append(f"Sampling frequency is determined: {sfreq}")
    


        return raw, sfreq,cleanliness_messages

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None, None,None
    

def is_data_clean(df):
    # Check for NaN values in the data
    if df.isnull().any().any():
        print("Data contains NaN values.")
        return False

    # Check for duplicate rows
    if df.duplicated().any():
        print("Data contains duplicate rows.")
        return False

    # Check for flatline signals by looking for columns with no variance
    if (df.std() ==0).any():
        print("Data contains flatline signals.")
        return False

    
    # If all checks pass, the data is considered clean
    return True


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



def is_mat_data_clean(eeg_data, sfreq):
    # Check for NaN values in the data
    if np.isnan(eeg_data).any():
        print("EEG data contains NaN values.")
        return False

    # Check for duplicate rows (trials)
    if np.array_equal(eeg_data, np.roll(eeg_data, 1, axis=0)):
        print("EEG data contains duplicate trials.")
        return False

    # Check for flatline signals by looking for columns with no variance
    if np.var(eeg_data, axis=1).min() == 0:  # axis=1 to check variance across time
        print("EEG data contains flatline channels.")
        return False


    # If all checks pass, the EEG data is considered clean
    return True

def read_mat_eeg(file_path):
    try:
        data_info=[]
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

            print("number of samples:", n_samples)
            print("number of trials:", n_trials)
            print("number of samples+trials:",n_trials * n_samples)
            print("number of channels:", n_channels)

            data_info.append(f"Number of channels: {n_channels}")
            data_info.append(f"Number of samples: {n_samples}")
            data_info.append(f"Number of trials: {n_trials}")
            data_info.append(f"Number of samples*trials: {n_trials * n_samples}")

            if np.isnan(eeg_data).any() or np.isinf(eeg_data).any():
                print("Data contains NaN or Infinity values")
                data_info.append("Data contains NaN or Infinity values")
            else:
                print("Data does not contain NaN or Infinity values")
                data_info.append("Data does not contain NaN or Infinity values")

            if np.var(eeg_data, axis=1).min() == 0:  # axis=1 to check variance across time
                print("EEG data contains flatline channels.")
                data_info.append("EEG data contains flatline channels.") 

        else:
            raise ValueError(f"Unexpected data dimensions {eeg_data.shape}")

        # Define channel names and types
        ch_names = ['EEG ' + str(i) for i in range(n_channels)]
        ch_types = ['eeg'] * n_channels

        # Define the sampling frequency
        sfreq = 250  # Replace with the actual sampling frequency if available
        data_info.append(f"Sampling Frequency is determined {sfreq}")

        # Create MNE info structure
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        # Create RawArray
        raw = mne.io.RawArray(eeg_data, info=info)

        # Extract labels if 'labels' key exists
        if 'labels' in mat:
            labels = mat['labels'].flatten()  # Ensure it is a 1D array

        else:
            raise ValueError("The key 'labels' was not found in the .mat file.")

        return raw, sfreq, labels,data_info

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None, None, None,None
    
    
def visualize_3d_brain_model(raw):
    
    # Define the correct mapping from the current channel names to the desired channel names
    channel_mapping1 = {'EEG 0': 'Fp1','EEG 1': 'Fp2','EEG 2': 'F7','EEG 3': 'F3',
                     'EEG 4': 'Fz','EEG 5': 'F4', 'EEG 6': 'F8', 'EEG 7': 'FC5','EEG 8': 'FC1',
                    'EEG 9': 'FC2','EEG 10': 'FC6', 'EEG 11': 'T3',
                    'EEG 12': 'C3','EEG 13': 'Cz', 'EEG 14': 'C4', 'EEG 15': 'T4',
                    'EEG 16': 'CP5', 'EEG 17': 'CP1','EEG 18': 'CP2','EEG 19': 'CP6','EEG 20': 'T5',
                    'EEG 21': 'P3','EEG 22': 'Pz','EEG 23': 'P4','EEG 24': 'T6',
                    'EEG 25': 'PO3','EEG 26': 'PO4', 'EEG 27': 'O1', 'EEG 28': 'Oz',
                         'EEG 29': 'O2','EEG 30': 'A1',   'EEG 31': 'A2', }
    

   # Rename the channels in the raw object
    raw.rename_channels(channel_mapping1)

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

    mne.viz.set_3d_backend('pyvista')        
    
    # Plot the sensor locations, including the head surface
    fig = mne.viz.plot_alignment(
        raw.info,
        trans='fsaverage',
        subject=subject,
        subjects_dir=subjects_dir,
        #dig=True,
        eeg=['projected'],
        #show_axes=True,
        surfaces='head-dense',  # Use a denser head surface for better visualization (can be changed to 'head' for sparser)
       
    )

     # Set the 3D view of the figure
    mne.viz.set_3d_view(figure=fig, azimuth=90, elevation=90, distance=0.6)

# Extract the 3D positions of the EEG sensors and their corresponding names
    ch_pos = {ch_name: raw.info['chs'][idx]['loc'][:3] for idx, ch_name in enumerate(raw.info['ch_names'])}

    # Get the PyVista plotter from the figure
    plotter = fig.plotter

    # Offset for text annotation
    offset = np.array([0.0, -0.02, -0.012])  # Adjust this offset as needed

    label_actors = {}
   

    # Loop through each channel position and add a text label to the plotter
    for ch_name, pos in ch_pos.items():
        # Apply an offset to each position for better visibility
        text_pos = pos + offset
        # Here, pos is a NumPy array with 3 elements: x, y, and z
        plotter.add_point_labels([text_pos], [ch_name], point_size=20, font_size=20, text_color='black')
        

    # Render the plotter to show the text annotations
    plotter.render()
    plt.show()

    # Wait for the plot to be closed
    plt.waitforbuttonpress()
    

    return fig

def get_sensor_positions(raw):
    positions = {}
    for ch in raw.info['chs']:
        if ch['ch_name'].startswith('EEG'):
            positions[ch['ch_name']] = ch['loc'][:3].tolist()
    return positions


def add_preprocessing_step(step_description, preprocessing_steps):
    preprocessing_steps.add(step_description)

def preprocess_channel_data(channel_data, sfreq, l_freq_hp, h_freq_lp, preprocessing_steps):

    channel_data = filter_data(channel_data, sfreq, l_freq=l_freq_hp, h_freq=h_freq_lp, verbose=False)
    add_preprocessing_step(f"High-pass filtered at {l_freq_hp}Hz and low-pass filtered at {h_freq_lp}Hz with sample frequency {sfreq}Hz.", preprocessing_steps)
    return channel_data



def preprocess_raw_eeg(raw, sfreq,session_name):
    preprocessing_steps = set()

    channel_mapping = {'EEG 0': 'Fp1','EEG 1': 'Fp2','EEG 2': 'F7','EEG 3': 'F3',
                     'EEG 4': 'Fz','EEG 5': 'F4', 'EEG 6': 'F8', 'EEG 7': 'FC5','EEG 8': 'FC1',
                    'EEG 9': 'FC2','EEG 10': 'FC6', 'EEG 11': 'T3',
                    'EEG 12': 'C3','EEG 13': 'Cz', 'EEG 14': 'C4', 'EEG 15': 'T4',
                    'EEG 16': 'CP5', 'EEG 17': 'CP1','EEG 18': 'CP2','EEG 19': 'CP6','EEG 20': 'T5',
                    'EEG 21': 'P3','EEG 22': 'Pz','EEG 23': 'P4','EEG 24': 'T6',
                    'EEG 25': 'PO3','EEG 26': 'PO4', 'EEG 27': 'O1', 'EEG 28': 'Oz',
                         'EEG 29': 'O2','EEG 30': 'A1',   'EEG 31': 'A2', }
    
    raw.rename_channels(channel_mapping)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    

    # Run ICA to remove artifacts.
    ica = ICA(n_components=10, random_state=97, max_iter=800)
    ica.fit(raw)
    raw = ica.apply(raw)
    add_preprocessing_step("Applied ICA for artifact removal.", preprocessing_steps)

    # Instantiate NoisyChannels with the filtered raw data.
    noisy_detector = NoisyChannels(raw)

    # Adjust the thresholds for finding bad channels if necessary.
    noisy_detector.find_bad_by_correlation()
    noisy_detector.find_bad_by_deviation()

    # Combine all the bad channels.
    bads = noisy_detector.get_bads()

    # Mark bad channels in the info structure.
    raw.info['bads'] = bads

    # Interpolate bad channels using good ones.
    if bads:
        raw.drop_channels(bads)
        add_preprocessing_step(f"Interpolated bad channels for {session_name}: {bads}", preprocessing_steps)


    # Extract the data for further processing if needed.
    eeg_data = raw.get_data()
    preprocessed_data = np.empty(eeg_data.shape)

    for i, channel in enumerate(eeg_data):
        preprocessed_data[i] = preprocess_channel_data(channel, sfreq, l_freq_hp=0.5, h_freq_lp=60.0, preprocessing_steps=preprocessing_steps)

    # Create an MNE RawArray object with the preprocessed data.
    ch_names = raw.info['ch_names']
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    preprocessed_raw = RawArray(preprocessed_data, info)

    return preprocessed_raw, list(preprocessing_steps)



def extract_features_from_channel(channel_data, sfreq, epoch_length=1.0):
    # Assuming channel_data is a 1D numpy array
    n_samples = channel_data.shape[0]
    n_epochs = int(n_samples / (epoch_length * sfreq))
    features_list = []
    diffs = np.abs(np.diff(channel_data))


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
       

        
        # Initialize a dictionary to hold features for the current epoch
        features = {
            'mean': mean_val,
            'std': std_val,
            'skew': skew_val,
            'rms': rms_val,
            'variance': variance_val,
           'peak_to_peak': np.ptp(epoch),
            'waveform_length': np.sum(np.abs(np.diff(epoch))),
        }

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

def csv_features(data):
    global all_features
    all_features=pd.DataFrame()
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
    return all_features    


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

    # if is_mat_data_clean(epochs_data, sfreq):
    #     print("Data is clean. Continuing feature extraction.")

    print("Labels before unique_classes check:", labels)
    # Ensure the number of epochs and labels match
    min_len = min(len(epochs_data), len(labels))
    epochs_data = epochs_data[:min_len]
    labels = labels[:min_len]
    unique_classes = np.unique(labels)
    print("Unique classes found:", unique_classes)

    if len(unique_classes) < 2:
        raise ValueError(f"Insufficient number of classes for CSP. Unique classes found: {unique_classes}")


    # Fit CSP
    csp.fit(epochs_data, labels)

    # Transform the data using CSP to get the log-variance features
    csp_features = csp.transform(epochs_data)

    feature_names = [
        'mean', 'standard deviation', 'skewness', 
        'root mean square', 'variance', 
        'peak-to-peak', 'waveform length'
    ]

    # Initialize a list to store combined features
    combined_features = []

    # Calculate time-domain features for each CSP-transformed epoch
    for epoch in csp_features:
        epoch_features = []
        for channel_data in epoch:
            # You can handle NaN values here as shown previously or use an imputer later
            mean_val = np.nanmean(channel_data)
            std_val = np.nanstd(channel_data)
            skew_val = skew(channel_data, nan_policy='omit')  # Handle NaN with nan_policy
            rms_val = np.sqrt(np.nanmean(channel_data**2))
            variance_val = np.nanvar(channel_data)
            peak_to_peak_val = np.nanmax(channel_data) - np.nanmin(channel_data)  # Handle NaN if necessary
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)

            # Append channel features to epoch features
            channel_features = [
                mean_val,
                min_val,
                max_val, 
                std_val,
                skew_val, 
                rms_val,
                variance_val, 
                peak_to_peak_val, 
            ]
            epoch_features.extend(channel_features)

        # Append to the combined features list
        combined_features.append(epoch_features)

    # Now use SimpleImputer to replace any remaining NaN values with the mean of the feature
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    combined_features_imputed = imputer.fit_transform(combined_features)

    # Fit LDA with the imputed data
    lda = LDA()
    lda.fit(combined_features_imputed, labels[:len(combined_features_imputed)])
    combined_features_transformed = lda.transform(combined_features_imputed)

    # Create a DataFrame from the combined features
    features_df = pd.DataFrame(combined_features_transformed)

    # Add labels to the DataFrame
    features_df['label'] = labels[:len(combined_features_imputed)]
    features_message = ("Extracted features using CSP, which maximizes the variance for two "
                        "classes, allowing for better separation. The time-domain features "
                        "extracted are:\n" + "\n".join(f"* {feature}" for feature in feature_names))

    return features_message, features_df


def extract_statistical_features(df):
    labels = df.iloc[:, -1].values  # Extract labels
    data = df.iloc[:, :-1]  # Extract data

    # Prepare column names for the features and labels
    feature_names = ['X' + str(i) for i in range(1, 180)]  # Generate names from X1 to X179

    # Calculate statistical features for each sample/epoch
    calculated_features = {
        'mean': data.mean(axis=1),
        'median': data.median(axis=1),
        'max': data.max(axis=1),
        'min': data.min(axis=1),
        'std': data.std(axis=1),
        'skew': data.apply(lambda x: skew(x), axis=1),
        'entropy': data.apply(lambda x: entropy(np.histogram(x, bins=10)[0]), axis=1),
        'label': labels
    }

    # Create a dictionary with all required feature names (most will be set to NaN)
    all_features = {name: calculated_features.get(name, pd.Series([np.nan]*len(df))) for name in feature_names}
    
    # Assign the calculated features to their respective columns
    for key in calculated_features:
        # Find the appropriate key index (subtract 1 because list is 0-indexed but names start at 1)
        index = feature_names.index(key)  
        all_features[feature_names[index]] = calculated_features[key]

    # Convert the dictionary to a DataFrame
    features_df = pd.DataFrame(all_features)

    return features_df

def csv_identification(file_paths,processed_data_keywords):
    global messages
    messages=[]
    csv_only=False

    for file_path in file_paths:
        csv_only=True
        
        # Determine the type of file and handle it accordingly
        if file_path.lower().endswith(('.csv', '.xls', '.xlsx', '.xlsm', '.xlsb')):
            raw_data, sfreq,cleanliness_messages= read_eeg_file(file_path)
            # picks = random.sample(raw_data.ch_names, 10)
            # plot_raw_eeg(raw_data, title=f'EEG Data from {file_path}', picks=picks)

            # if raw_data is not None and isinstance(raw_data, mne.io.RawArray):
            #     data_df = raw_data.to_data_frame()
            #     if not is_data_clean(data_df):
            #         print(f"Data in {file_path} is not clean.")
            #         continue

            if isinstance(raw_data, mne.io.RawArray):
                # Convert to DataFrame here
                data_df = raw_data.to_data_frame()
            
                identified_methods = identify_feature_extraction_methods(data_df, processed_data_keywords)
                if identified_methods:
                    message = f"The following feature extraction methods were identified: {', '.join(identified_methods)}"
                    messages.append(message)                    
                    # Identify calculation methods for each band of interest
                    csv_features(raw_data)   #function that takes the features and put it in a variable to model

                    

                else:
                    messages.append("There is no known extracted feature methods") 
                    #features_df = extract_statistical_features(data_df)
                    # Save the features to a CSV file
                    # filename = "statistical_features.csv"  # You can change the filename as needed
                    # features_df.to_csv(filename, index=False)
                    # print("Features extracted using CSP and statistical analysis.")
                    # Add the features to the global all_features DataFrame
                    csv_features(data_df) 

            else:
                identified_methods = identify_feature_extraction_methods(raw_data, processed_data_keywords)
                if identified_methods:
                    message = f"The following feature extraction methods were identified: {', '.join(identified_methods)}"
                    messages.append(message)   
                    csv_features(raw_data)   #function that takes the features and put it in a variable to model
                 
                    
                else:
                    messages.append("There is no known extracted feature methods")
                    #features_df = extract_statistical_features(data_df)
                    # Save the features to a CSV file
                    # filename = "statistical_features.csv"  # You can change the filename as needed
                    # features_df.to_csv(filename, index=False)
                    # print("Features extracted using CSP and statistical analysis.")
                    # Add the features to the global all_features DataFrame
                    csv_features(data_df)  

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



def csv_modeling():
        
       
        progress_updates.append(10)
        # session['progress'] = 10
        # sleep(0.1)

        print("Length of all_features: ", len(all_features))
        print("Length of labels_list: ", len(labels_list))   
        
        X = all_features.iloc[:, :-1]  # features: all columns except the last
        y_original = all_features.iloc[:, -1]  # original labels: the last column
    
        # Fit the label encoder on the unique labels in the dataset
        label_encoder.fit(np.unique(y_original))

        # Transform labels to encoded numeric labels
        y_binned = label_encoder.transform(y_original)

    
        dump(label_encoder, encoder_path)


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
        progress_updates.append(20)
        # sleep(0.1)
        # session['progress'] = 20

        clf.fit(X_train, y_train)
        
        y_pred_svc = clf.predict(X_test)
        acc_svc = round(clf.score(X_test, y_test) * 100, 2)
        
        dump(clf, model_svc)
        print(f"Model saved to {model_svc}")

        precision_svc = round(precision_score(y_test, y_pred_svc, average='macro') * 100, 2)

        # Initialize the classifier
        clf_rf = RandomForestClassifier(n_estimators=50)
        progress_updates.append(40)
        # sleep(0.1)
        # session['progress'] = 40

        # Fit the classifier to the training data
        clf_rf.fit(X_train, y_train)

        # Predict on the test data
        y_pred_rf = clf_rf.predict(X_test)

        # Calculate the accuracy
        acc_rf = round(clf_rf.score(X_test, y_test) * 100, 2)

        dump(clf_rf, model_rf)
        print(f"Model saved to {model_rf}")

        precision_rf = round(precision_score(y_test, y_pred_rf, average='macro') * 100, 2)


        # clf_gbc = GradientBoostingClassifier(n_estimators=50)
        # progress_updates.append(60)
        # sleep(0.1)

        # # Fit the classifier to the training data
        # clf_gbc.fit(X_train, y_train)

        # # Predict on the test data
        # y_pred_gbc = clf_gbc.predict(X_test)

        # # Calculate the accuracy on the training set
        # acc_gbc = round(accuracy_score(y_test,y_pred_gbc)*100,2)

        clf_lr = LogisticRegression()
        progress_updates.append(60)
        # sleep(0.1)
        # session['progress'] = 60

        # Fit the classifier to the training data
        clf_lr.fit(X_train, y_train)

        # Predict on the test data
        y_pred_lr = clf_lr.predict(X_test)

        # Calculate the accuracy
        acc_lr = round(clf_lr.score(X_test, y_test) * 100, 2)

        dump(clf_lr, model_lr)
        print(f"Model saved to {model_lr}")

        precision_lr = round(precision_score(y_test, y_pred_lr, average='macro') * 100, 2)


        clf_knn = KNeighborsClassifier(n_neighbors=5)  # You can tune the n_neighbors parameter.
        progress_updates.append(80)
        # sleep(0.1) 
        # session['progress'] = 80

        # Fit the classifier to the training data
        clf_knn.fit(X_train, y_train)

        # Predict on the test data
        y_pred_knn = clf_knn.predict(X_test)

        # Calculate the accuracy
        acc_knn = round(clf_knn.score(X_test, y_test) * 100, 2)

        dump(clf_knn, model_knn)
        print(f"Model saved to {model_knn}")

        precision_knn = round(precision_score(y_test, y_pred_knn, average='macro') * 100, 2)
        

        # Reshape data for 1D CNN input (batch_size, steps, input_dimension)
        X_train_cnn = np.expand_dims(X_train, axis=2)
        X_test_cnn = np.expand_dims(X_test, axis=2)

        # Convert labels to categorical (one-hot encoding)
        y_train_categorical = to_categorical(y_train)
        y_test_categorical = to_categorical(y_test)

        # Define the 1D CNN model
        #filters detect different features in the data, kernel size of the sliding window
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
        #epochs=5 means that the entire dataset will be passed through the CNN five times
        #batch size means 
        model.fit(X_train_cnn, y_train_categorical, epochs=5)

        # Evaluate the model
        _, accuracy = model.evaluate(X_test_cnn, y_test_categorical, verbose=0)
        y_pred_cnn = model.predict(X_test_cnn)
        y_pred_cnn_classes = y_pred_cnn.argmax(axis=-1)

        dump(model, model_cnn)
        print(f"Model saved to {model_cnn}")

        # Calculate precision for CNN
        precision_cnn = round(precision_score(y_test, y_pred_cnn_classes, average='macro') * 100, 2)

        progress_updates.append(100)
        # sleep(0.1)
        # session['progress'] = 100

        # Train and evaluate models
        models = [clf, clf_rf, clf_lr, clf_knn,model]
        model_names = ['SVM', 'Random Forest', 'Logistic Regression', 'KNN','CNN']

        results = {'Model': [], 'Accuracy': [], 'F1 Score': [], 'Precision': [], 'Recall': []}

        for model, name in zip(models, model_names):
            if name == 'CNN':
                # Fit the CNN model with the correctly shaped data
                model.fit(X_train_cnn, y_train_categorical, epochs=5)
                # Use the CNN-specific data to make predictions
                y_pred_probs = model.predict(X_test_cnn)
                y_pred = y_pred_probs.argmax(axis=-1)
            else:
                # For non-CNN models, fit with the original data
                model.fit(X_train, y_train)
                # Use the original data to make predictions
                y_pred = model.predict(X_test)
                
            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred) *100
            f1 = f1_score(y_test, y_pred, average='weighted')*100
            precision = precision_score(y_test, y_pred, average='weighted')*100
            recall = recall_score(y_test, y_pred, average='weighted')*100
            # Store results
            results['Model'].append(name)
            results['Accuracy'].append(accuracy)
            results['F1 Score'].append(f1)
            results['Precision'].append(precision)
            results['Recall'].append(recall)

        # Display results
        results_df = pd.DataFrame(results)
        print(results_df)


        print("SVM Accuracy is:",(str(acc_svc)+'%'))
        print("Random Forest accuracy is:", (str(acc_rf) + '%'))
        print("Logistic Regression Classifier accuracy is:", (str(acc_lr) + '%')) 
        print("KNN Accuracy is:", (str(acc_knn) + '%'))
        print(f'CNN 1D - Accuracy: {accuracy * 100:.2f}%')

        # print("SVM Precision is:", (str(precision_svc) + '%'))
        # print("Random Forest Precision is:", (str(precision_rf) + '%'))
        # print("Logistic Regression Precision is:", (str(precision_lr) + '%'))
        # print("KNN Precision is:", (str(precision_knn) + '%'))
        # print("CNN Precision is:", (str(precision_cnn) + '%'))



        accuracies = {
        'SVM': acc_svc,
        'Random_Forest': acc_rf,
        'Logistic Regression': acc_lr,
        'KNN': acc_knn,
        'CNN': accuracy * 100  # Assuming 'accuracy' is the accuracy of the CNN model
    }
        return accuracies,progress_updates

        # print("Classes distribution in training set:", np.unique(y_train, return_counts=True))
        # print("Classes distribution in testing set:", np.unique(y_test, return_counts=True))

        # print("Unique classes in y_train:", np.unique(y_train))
        # print("Unique classes in y_test:", np.unique(y_test)) 

labels_text_array=[]
test_indices=None
train_indices=None
y_original = [] # original labels: the last column
def csv_svc_model_new(label_conditions_path):
    global y_original
    class_info={}
    print("Length of all_features: ", len(all_features))
    print("Length of labels_list: ", len(labels_list))   
    labels_text_array=[]
    X = all_features.iloc[:, :-1]  # features: all columns except the last
    y_original = all_features.iloc[:, -1]  # original labels: the last column

    # Process labels to scale down by millions if applicable and leave non-numeric as is
    def scale_if_numeric(x):
        if isinstance(x, (int, float)):  # Check if the value is numeric
            return x // 1_000_000 if x >= 1_000_000 else x
        return x  # Return as is if non-numeric

    scaled_y_original = y_original.apply(scale_if_numeric)
    # Calculate class counts on the processed data
    class_counts = scaled_y_original.value_counts()
    print("Class distribution:", class_counts)

    labels_array = []

    for label in y_original:
        if isinstance(label, (int, float)):  # Check if the label is numeric
            if label >= 1_000_000:  # Check if the label is in millions
                label = int(label // 1_000_000)  # Scale down to just the millions and convert to int
            else:
                label = int(label)  # Ensure it's an integer to remove trailing decimals
        labels_array.append(label)
        
    #print("Labels array:",labels_array)    

    if os.path.isfile(label_conditions_path):
        label_conditions = read_label_conditions(label_conditions_path)
    else:
        raise ValueError(f"Label conditions file not found at {label_conditions_path}")
    
    # Convert numeric labels in labels_array to text using label_conditions mapping
    labels_text_array = [label_conditions.get(str(label).lower(), "Unknown label") for label in labels_array]
    #print("labels text array:",labels_text_array)
    

    # Fit the label encoder on the unique labels in the dataset
    label_encoder.fit(np.unique(y_original))

    # Transform labels to encoded numeric labels
    y_binned = label_encoder.transform(y_original)

    # Calculate the number of unique classes
    num_classes = len(np.unique(y_binned))
    print("Number of classes:", num_classes)
    class_info = {
        "Number of classes": num_classes,
        "Class distribution": class_counts.to_dict()  # Convert to dict to ensure compatibility
    }

    dump(label_encoder, encoder_path)

    # print("Y binned:", y_binned)
    # print("Last column: ",all_features.iloc[:, -1])

    print("Y head:",y_binned[:5])  # To see the first few entries
    print("Y unique:",np.unique(y_binned))  # To see the unique values

    # Perform a train-test split
    global X_train, X_test, y_train, y_test 
    global train_indices
    global test_indices
    X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2)
    train_indices = X_train.index
    test_indices = X_test.index

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)

    # Instantiate and train the SVC model
    clf =SVC()
    clf.fit(X_train, y_train)
    y_pred_svc = clf.predict(X_test)
    
    acc_svc = round(clf.score(X_train, y_train) * 100, 2)
    # acc_test = round(clf.score(X_test, y_test) * 100, 2)
    print("acc svc",acc_svc)

    
    dump(clf, model_svc)
    print(f"Model saved to {model_svc}")


    dump(scaler, scaler_path)

    accuracy_svc_csv = {
        'SVM': acc_svc}
    
    model_names = ['SVM']

    result_svc_csv = {'Model': [], 'Accuracy': [], 'F1 Score': [], 'Precision': [], 'Recall': []}

   
    accuracy = accuracy_score(y_test, y_pred_svc) * 100
    f1 = f1_score(y_test, y_pred_svc, average='weighted') * 100
    precision = precision_score(y_test, y_pred_svc, average='weighted') * 100
    recall = recall_score(y_test, y_pred_svc, average='weighted') * 100
    # Store results
    result_svc_csv['Model'].append(model_names[0])
    result_svc_csv['Accuracy'].append(accuracy)
    result_svc_csv['F1 Score'].append(f1)
    result_svc_csv['Precision'].append(precision)
    result_svc_csv['Recall'].append(recall)

    
    return accuracy_svc_csv, result_svc_csv, labels_text_array,y_original,class_info

def csv_svc_model(label_conditions_path):

    print("Length of all_features: ", len(all_features))
    print("Length of labels_list: ", len(labels_list))   
    labels_text_array=[]
    X = all_features.iloc[:, :-1]  # features: all columns except the last
    y_original = all_features.iloc[:, -1]  # original labels: the last column
    labels_array = []
    X, y_original = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    # class_distribution = y_original.value_counts(normalize=True)
    # print("Class distribution:\n", class_distribution)

    for label in y_original:
        if isinstance(label, (int, float)):  # Check if the label is numeric
            if label >= 1_000_000:  # Check if the label is in millions
                label = int(label // 1_000_000)  # Scale down to just the millions and convert to int
            else:
                label = int(label)  # Ensure it's an integer to remove trailing decimals
        labels_array.append(label)
        
    #print("Labels array:",labels_array)    

    if os.path.isfile(label_conditions_path):
        label_conditions = read_label_conditions(label_conditions_path)
    else:
        raise ValueError(f"Label conditions file not found at {label_conditions_path}")
    
    # Convert numeric labels in labels_array to text using label_conditions mapping
    labels_text_array = [label_conditions.get(str(label).lower(), "Unknown label") for label in labels_array]
    #print("labels text array:",labels_text_array)
    

    # Fit the label encoder on the unique labels in the dataset
    label_encoder.fit(np.unique(y_original))

    # Transform labels to encoded numeric labels
    y_binned = label_encoder.transform(y_original)

    dump(label_encoder, encoder_path)

    # print("Y binned:", y_binned)
    # print("Last column: ",all_features.iloc[:, -1])

    print("Y head:",y_binned[:5])  # To see the first few entries
    print("Y unique:",np.unique(y_binned))  # To see the unique values

    # Perform a train-test split
    global X_train, X_test, y_train, y_test 
    X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)

    C_values = [0.1, 1, 10]
    for C in C_values:
       clf =SVC(kernel='linear',C=C)
       clf.fit(X_train, y_train)
       y_pred_svc = clf.predict(X_test)
       accuracy = accuracy_score(y_test, y_pred_svc)*100
       print(f"Accuracy with C={C}: {accuracy}")

    
    progress_updates.append(20)
    # sleep(0.1)
    # session['progress'] = 20
    
    
    #acc_svc = round(clf.score(X_train, y_train) * 100, 2)
    # acc_test = round(clf.score(X_test, y_test) * 100, 2)
    
    dump(clf, model_svc)
    print(f"Model saved to {model_svc}")
    # print("SVM Accuracy is:",(str(acc_svc)+'%'))
    # print("SVM Accuracy is:",(str(acc_test)+'%'))


    dump(scaler, scaler_path)

    accuracy_svc_csv = {
        'SVM': accuracy}
    
    model_names = ['SVM']

    result_svc_csv = {'Model': [], 'Accuracy': [], 'F1 Score': [], 'Precision': [], 'Recall': []}

    # for model, name in zip(models, model_names):
        
    #     # For non-CNN models, fit with the original data
    #     model.fit(X_train, y_train)
    #     # Use the original data to make predictions
    #     y_pred = model.predict(X_test)
            
    #     # Calculate evaluation metrics
    #     accuracy = accuracy_score(y_test, y_pred) *100
    #     f1 = f1_score(y_test, y_pred, average='weighted')*100
    #     precision = precision_score(y_test, y_pred, average='weighted')*100
    #     recall = recall_score(y_test, y_pred, average='weighted')*100
    accuracy = accuracy_score(y_test, y_pred_svc) * 100
    f1 = f1_score(y_test, y_pred_svc, average='weighted') * 100
    precision = precision_score(y_test, y_pred_svc, average='weighted') * 100
    recall = recall_score(y_test, y_pred_svc, average='weighted') * 100
    # Store results
    result_svc_csv['Model'].append(model_names[0])
    result_svc_csv['Accuracy'].append(accuracy)
    result_svc_csv['F1 Score'].append(f1)
    result_svc_csv['Precision'].append(precision)
    result_svc_csv['Recall'].append(recall)

    
    return accuracy_svc_csv, result_svc_csv, labels_text_array


def csv_random_model(label_conditions_path):

    global y_original
    class_info={}
    print("Length of all_features: ", len(all_features))
    print("Length of labels_list: ", len(labels_list))   
    labels_text_array=[]
    X = all_features.iloc[:, :-1]  # features: all columns except the last
    y_original = all_features.iloc[:, -1]  # original labels: the last column
    # Process labels to scale down by millions if applicable and leave non-numeric as is
    def scale_if_numeric(x):
        if isinstance(x, (int, float)):  # Check if the value is numeric
            return x // 1_000_000 if x >= 1_000_000 else x
        return x  # Return as is if non-numeric

    scaled_y_original = y_original.apply(scale_if_numeric)
    # Calculate class counts on the processed data
    class_counts = scaled_y_original.value_counts()
    print("Class distribution:", class_counts)

    labels_array = []

    for label in y_original:
        if isinstance(label, (int, float)):  # Check if the label is numeric
            if label >= 1_000_000:  # Check if the label is in millions
                label = int(label // 1_000_000)  # Scale down to just the millions and convert to int
            else:
                label = int(label)  # Ensure it's an integer to remove trailing decimals
        labels_array.append(label)
        
    #print("Labels array:",labels_array)  

    if os.path.isfile(label_conditions_path):
        label_conditions = read_label_conditions(label_conditions_path)
    else:
        raise ValueError(f"Label conditions file not found at {label_conditions_path}")
    
    # Convert numeric labels in labels_array to text using label_conditions mapping
    labels_text_array = [label_conditions.get(str(label).lower(), "Unknown label") for label in labels_array]
    print("labels text array:",labels_text_array)

    # Fit the label encoder on the unique labels in the dataset
    label_encoder.fit(np.unique(y_original))

    # Transform labels to encoded numeric labels
    y_binned = label_encoder.transform(y_original)

    # Calculate the number of unique classes
    num_classes = len(np.unique(y_binned))
    print("Number of classes:", num_classes)
    class_info = {
        "Number of classes": num_classes,
        "Class distribution": class_counts.to_dict()  # Convert to dict to ensure compatibility
    }

    dump(label_encoder, encoder_path)

    print("Y binned:", y_binned)
    print("Last column: ",all_features.iloc[:, -1])

    print("Y head:",y_binned[:5])  # To see the first few entries
    print("Y unique:",np.unique(y_binned))  # To see the unique values

    # Perform a train-test split
    global X_train, X_test, y_train, y_test 
    global train_indices
    global test_indices
    X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2)
    train_indices = X_train.index
    test_indices = X_test.index

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)
    # Initialize the classifier
    clf_rf = RandomForestClassifier(n_estimators=50)
    progress_updates.append(40)
    # sleep(0.1)
    # session['progress'] = 40

    # Fit the classifier to the training data
    clf_rf.fit(X_train, y_train)

    dump(scaler, scaler_path)

    # Predict on the test data
    y_pred_rf = clf_rf.predict(X_test)

    # Calculate the accuracy
    # acc_rf = round(clf_rf.score(X_train, y_train) * 100, 2)
    # acc_test = round(clf_rf.score(X_test, y_test) * 100, 2)


    accuracy_rf = accuracy_score(y_test, y_pred_rf) * 100
    f1 = f1_score(y_test, y_pred_rf, average='weighted') * 100
    precision = precision_score(y_test, y_pred_rf, average='weighted') * 100
    recall = recall_score(y_test, y_pred_rf, average='weighted') * 100

    dump(clf_rf, model_rf)
    print(f"Model saved to {model_rf}")
    #print("Random Forest accuracy is:", (str(acc_rf) + '%'))
    print("Random Forest accuracy is:", (str(accuracy_rf) + '%'))

    accuracy_random_csv = {
        'Random Forest': accuracy_rf}
    
    models = [clf_rf]
    model_names = ['Random Forest']

    result_random_csv = {'Model': [], 'Accuracy': [], 'F1 Score': [], 'Precision': [], 'Recall': []}

   

    # for model, name in zip(models, model_names):
        
    #     # For non-CNN models, fit with the original data
    #     model.fit(X_train, y_train)
    #     # Use the original data to make predictions
    #     y_pred = model.predict(X_test)
            
    #     # Calculate evaluation metrics
    #     accuracy = accuracy_score(y_test, y_pred) *100
    #     f1 = f1_score(y_test, y_pred, average='weighted')*100
    #     precision = precision_score(y_test, y_pred, average='weighted')*100
    #     recall = recall_score(y_test, y_pred, average='weighted')*100
    # Store results
    result_random_csv['Model'].append(model_names[0])
    result_random_csv['Accuracy'].append(accuracy_rf)
    result_random_csv['F1 Score'].append(f1)
    result_random_csv['Precision'].append(precision)
    result_random_csv['Recall'].append(recall)

    return accuracy_random_csv,result_random_csv,labels_text_array,y_original,class_info

def csv_logistic_model(label_conditions_path):
     
    global y_original
    class_info={}
    print("Length of all_features: ", len(all_features))
    print("Length of labels_list: ", len(labels_list))   
    
    labels_text_array=[]
    X = all_features.iloc[:, :-1]  # features: all columns except the last
    y_original = all_features.iloc[:, -1]  # original labels: the last column
    # Process labels to scale down by millions if applicable and leave non-numeric as is
    def scale_if_numeric(x):
        if isinstance(x, (int, float)):  # Check if the value is numeric
            return x // 1_000_000 if x >= 1_000_000 else x
        return x  # Return as is if non-numeric

    scaled_y_original = y_original.apply(scale_if_numeric)
    # Calculate class counts on the processed data
    class_counts = scaled_y_original.value_counts()
    print("Class distribution:", class_counts)

    labels_array = []

    for label in y_original:
        if isinstance(label, (int, float)):  # Check if the label is numeric
            if label >= 1_000_000:  # Check if the label is in millions
                label = int(label // 1_000_000)  # Scale down to just the millions and convert to int
            else:
                label = int(label)  # Ensure it's an integer to remove trailing decimals
        labels_array.append(label)
        
    print("Labels array:",labels_array) 

    if os.path.isfile(label_conditions_path):
        label_conditions = read_label_conditions(label_conditions_path)
    else:
        raise ValueError(f"Label conditions file not found at {label_conditions_path}")
    
    # Convert numeric labels in labels_array to text using label_conditions mapping
    labels_text_array = [label_conditions.get(str(label).lower(), "Unknown label") for label in labels_array]
    print("labels text array:",labels_text_array)

    # Fit the label encoder on the unique labels in the dataset
    label_encoder.fit(np.unique(y_original))

    # Transform labels to encoded numeric labels
    y_binned = label_encoder.transform(y_original)

    # Calculate the number of unique classes
    num_classes = len(np.unique(y_binned))
    print("Number of classes:", num_classes)
    class_info = {
        "Number of classes": num_classes,
        "Class distribution": class_counts.to_dict()  # Convert to dict to ensure compatibility
    }

    dump(label_encoder, encoder_path)

    print("Y binned:", y_binned)
    print("Last column: ",all_features.iloc[:, -1])

    print("Y head:",y_binned[:5])  # To see the first few entries
    print("Y unique:",np.unique(y_binned))  # To see the unique values

    # Perform a train-test split
    global X_train, X_test, y_train, y_test 
    global train_indices
    global test_indices
    X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2)
    train_indices = X_train.index
    test_indices = X_test.index

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)

    clf_lr = LogisticRegression()
    progress_updates.append(60)
    # sleep(0.1)
    # session['progress'] = 60

    # Fit the classifier to the training data
    clf_lr.fit(X_train, y_train)

    dump(scaler, scaler_path)

    # Predict on the test data
    y_pred_lr = clf_lr.predict(X_test)

    # Calculate the accuracy
    acc_lr = round(clf_lr.score(X_train, y_train) * 100, 2)
    print("acc train:",acc_lr)
    acc_test = round(clf_lr.score(X_test, y_test) * 100, 2)
    print("acc test:",acc_test)

    dump(clf_lr, model_lr)
    print(f"Model saved to {model_lr}")
    print("Logistic Regression Classifier accuracy is:", (str(acc_lr) + '%')) 
    print("Logistic Regression Classifier accuracy is:", (str(acc_test) + '%')) 


    accuracy_logistic_csv = {
        'Logistic Regression': acc_lr}
    
    models = [clf_lr]
    model_names = ['Logistic Regression']

    result_logistic_csv = {'Model': [], 'Accuracy': [], 'F1 Score': [], 'Precision': [], 'Recall': []}

    for model, name in zip(models, model_names):
        
        # For non-CNN models, fit with the original data
        model.fit(X_train, y_train)
        # Use the original data to make predictions
        y_pred = model.predict(X_test)
            
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred) *100
        f1 = f1_score(y_test, y_pred, average='weighted')*100
        precision = precision_score(y_test, y_pred, average='weighted')*100
        recall = recall_score(y_test, y_pred, average='weighted')*100
        # Store results
        result_logistic_csv['Model'].append(name)
        result_logistic_csv['Accuracy'].append(accuracy)
        result_logistic_csv['F1 Score'].append(f1)
        result_logistic_csv['Precision'].append(precision)
        result_logistic_csv['Recall'].append(recall)
    
    return accuracy_logistic_csv,result_logistic_csv,labels_text_array,y_original,class_info


def csv_knn_model(label_conditions_path):

    global y_original
    class_info={}

    print("Length of all_features: ", len(all_features))
    print("Length of labels_list: ", len(labels_list))   

    labels_text_array=[]
    X = all_features.iloc[:, :-1]  # features: all columns except the last
    y_original = all_features.iloc[:, -1]  # original labels: the last column
    # Process labels to scale down by millions if applicable and leave non-numeric as is
    def scale_if_numeric(x):
        if isinstance(x, (int, float)):  # Check if the value is numeric
            return x // 1_000_000 if x >= 1_000_000 else x
        return x  # Return as is if non-numeric

    scaled_y_original = y_original.apply(scale_if_numeric)
    # Calculate class counts on the processed data
    class_counts = scaled_y_original.value_counts()
    print("Class distribution:", class_counts)

    labels_array = []

    for label in y_original:
        if isinstance(label, (int, float)):  # Check if the label is numeric
            if label >= 1_000_000:  # Check if the label is in millions
                label = int(label // 1_000_000)  # Scale down to just the millions and convert to int
            else:
                label = int(label)  # Ensure it's an integer to remove trailing decimals
        labels_array.append(label)
        
    print("Labels array:",labels_array) 

    if os.path.isfile(label_conditions_path):
        label_conditions = read_label_conditions(label_conditions_path)
    else:
        raise ValueError(f"Label conditions file not found at {label_conditions_path}")
    
    # Convert numeric labels in labels_array to text using label_conditions mapping
    labels_text_array = [label_conditions.get(str(label).lower(), "Unknown label") for label in labels_array]
    print("labels text array:",labels_text_array)

    # Fit the label encoder on the unique labels in the dataset
    label_encoder.fit(np.unique(y_original))

    # Transform labels to encoded numeric labels
    y_binned = label_encoder.transform(y_original)

    # Calculate the number of unique classes
    num_classes = len(np.unique(y_binned))
    print("Number of classes:", num_classes)
    class_info = {
        "Number of classes": num_classes,
        "Class distribution": class_counts.to_dict()  # Convert to dict to ensure compatibility
    }

    dump(label_encoder, encoder_path)

    print("Y binned:", y_binned)
    print("Last column: ",all_features.iloc[:, -1])

    print("Y head:",y_binned[:5])  # To see the first few entries
    print("Y unique:",np.unique(y_binned))  # To see the unique values

    # Perform a train-test split
    global X_train, X_test, y_train, y_test 
    global train_indices
    global test_indices
    X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2)
    train_indices = X_train.index
    test_indices = X_test.index

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)

    clf_knn = KNeighborsClassifier(n_neighbors=5)  # You can tune the n_neighbors parameter.
    progress_updates.append(80)
    # sleep(0.1) 
    # session['progress'] = 80

    # Fit the classifier to the training data
    clf_knn.fit(X_train, y_train)

    dump(scaler, scaler_path)

    # Predict on the test data
    y_pred_knn = clf_knn.predict(X_test)

    # Calculate the accuracy
    acc_knn = round(clf_knn.score(X_train, y_train) * 100, 2)
    acc_test = round(clf_knn.score(X_test, y_test) * 100, 2)


    dump(clf_knn, model_knn)
    print(f"Model saved to {model_knn}")
    print("KNN Accuracy is:", (str(acc_knn) + '%'))
    print("KNN Accuracy is:", (str(acc_test) + '%'))


    accuracy_knn_csv = {
        'KNN': acc_knn}
    
    models = [clf_knn]
    model_names = ['KNN']

    result_knn_csv = {'Model': [], 'Accuracy': [], 'F1 Score': [], 'Precision': [], 'Recall': []}

    for model, name in zip(models, model_names):
        
        # For non-CNN models, fit with the original data
        model.fit(X_train, y_train)
        # Use the original data to make predictions
        y_pred = model.predict(X_test)
            
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred) *100
        f1 = f1_score(y_test, y_pred, average='weighted')*100
        precision = precision_score(y_test, y_pred, average='weighted')*100
        recall = recall_score(y_test, y_pred, average='weighted')*100
        # Store results
        result_knn_csv['Model'].append(name)
        result_knn_csv['Accuracy'].append(accuracy)
        result_knn_csv['F1 Score'].append(f1)
        result_knn_csv['Precision'].append(precision)
        result_knn_csv['Recall'].append(recall)
    
    return accuracy_knn_csv,result_knn_csv,labels_text_array,y_original,class_info


def csv_cnn_model(label_conditions_path):

    global y_original
    class_info={}

    print("Length of all_features: ", len(all_features))
    print("Length of labels_list: ", len(labels_list))   
    
    labels_text_array=[]
    X = all_features.iloc[:, :-1]  # features: all columns except the last
    y_original = all_features.iloc[:, -1]  # original labels: the last column
    # Process labels to scale down by millions if applicable and leave non-numeric as is
    def scale_if_numeric(x):
        if isinstance(x, (int, float)):  # Check if the value is numeric
            return x // 1_000_000 if x >= 1_000_000 else x
        return x  # Return as is if non-numeric

    scaled_y_original = y_original.apply(scale_if_numeric)
    # Calculate class counts on the processed data
    class_counts = scaled_y_original.value_counts()
    print("Class distribution:", class_counts)

    labels_array = []

    for label in y_original:
        if isinstance(label, (int, float)):  # Check if the label is numeric
            if label >= 1_000_000:  # Check if the label is in millions
                label = int(label // 1_000_000)  # Scale down to just the millions and convert to int
            else:
                label = int(label)  # Ensure it's an integer to remove trailing decimals
        labels_array.append(label)
        
    print("Labels array:",labels_array) 

    if os.path.isfile(label_conditions_path):
        label_conditions = read_label_conditions(label_conditions_path)
    else:
        raise ValueError(f"Label conditions file not found at {label_conditions_path}")
    
    # Convert numeric labels in labels_array to text using label_conditions mapping
    labels_text_array = [label_conditions.get(str(label).lower(), "Unknown label") for label in labels_array]
    print("labels text array:",labels_text_array)

    # Fit the label encoder on the unique labels in the dataset
    label_encoder.fit(np.unique(y_original))

    # Transform labels to encoded numeric labels
    y_binned = label_encoder.transform(y_original)

    # Calculate the number of unique classes
    num_classes = len(np.unique(y_binned))
    print("Number of classes:", num_classes)
    class_info = {
        "Number of classes": num_classes,
        "Class distribution": class_counts.to_dict()  # Convert to dict to ensure compatibility
    }

    dump(label_encoder, encoder_path)

    print("Y binned:", y_binned)
    print("Last column: ",all_features.iloc[:, -1])

    print("Y head:",y_binned[:5])  # To see the first few entries
    print("Y unique:",np.unique(y_binned))  # To see the unique values

    # Perform a train-test split
    global X_train, X_test, y_train, y_test 
    global train_indices
    global test_indices
    X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2)
    train_indices = X_train.index
    test_indices = X_test.index

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)

    # Reshape data for 1D CNN input (batch_size, steps, input_dimension)
    X_train_cnn = np.expand_dims(X_train, axis=2)
    X_test_cnn = np.expand_dims(X_test, axis=2)

    # Convert labels to categorical (one-hot encoding)
    y_train_categorical = to_categorical(y_train)
    y_test_categorical = to_categorical(y_test)

    # Define the 1D CNN model
    #filters detect different features in the data, kernel size of the sliding window
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
    #epochs=5 means that the entire dataset will be passed through the CNN five times
    #batch size means 
    model.fit(X_train_cnn, y_train_categorical, epochs=5)

    dump(scaler, scaler_path)

    # Evaluate the model
    _, accuracy = model.evaluate(X_train_cnn, y_train_categorical, verbose=0)
    _1, accuracy_test = model.evaluate(X_test_cnn, y_test_categorical, verbose=0)

    y_pred_cnn = model.predict(X_test_cnn)
    y_pred_cnn_classes = y_pred_cnn.argmax(axis=-1)

    dump(model, model_cnn)
    print(f"Model saved to {model_cnn}")
    print(f'CNN 1D - Accuracy: {accuracy * 100:.2f}%')
    print(f'CNN 1D - Accuracy: {accuracy_test * 100:.2f}%')

    accuracy_cnn_csv = {
        'CNN': accuracy*100}
    
    models = [model]
    model_names = ['CNN']

    result_cnn_csv= {'Model': [], 'Accuracy': [], 'F1 Score': [], 'Precision': [], 'Recall': []}

    for model, name in zip(models, model_names):
        if name == 'CNN':
            # Fit the CNN model with the correctly shaped data
            model.fit(X_train_cnn, y_train_categorical, epochs=5)
            # Use the CNN-specific data to make predictions
            y_pred_probs = model.predict(X_test_cnn)
            y_pred = y_pred_probs.argmax(axis=-1)
            
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred) *100
        f1 = f1_score(y_test, y_pred, average='weighted')*100
        precision = precision_score(y_test, y_pred, average='weighted')*100
        recall = recall_score(y_test, y_pred, average='weighted')*100
        # Store results
        result_cnn_csv['Model'].append(name)
        result_cnn_csv['Accuracy'].append(accuracy)
        result_cnn_csv['F1 Score'].append(f1)
        result_cnn_csv['Precision'].append(precision)
        result_cnn_csv['Recall'].append(recall)
    
    return accuracy_cnn_csv,result_cnn_csv,labels_text_array,y_original,class_info

def read_label_conditions(file_path):
    print("entered read label")
    with open(file_path, 'r') as file:
        label_conditions = {}

        for line in file:
            parts = line.split(':')
            if len(parts) == 2:
                # Trim whitespace and convert keys to strings
                key = parts[0].strip()
                # Ensure that numeric keys are stored as strings
                if key.isdigit():
                    key = str(int(key))  # Convert digits to integers, then to strings to remove leading zeros
                # Convert everything to a consistent case
                label_conditions[key.lower()] = parts[1].strip()   
    return label_conditions

def load_and_predict(new_data, label_conditions):
    if not os.path.exists(encoder_path):
        raise FileNotFoundError("Could not find the label encoder at the specified path.")

    label_encoder = load(encoder_path)
    formatted_predictions = {}
    # Dictionary to store model paths
    model_paths = {
        'Model_SVC': model_svc,
        'Model_RF': model_rf,
        'Model_LR': model_lr,
        'Model_KNN': model_knn,
        'Model_CNN': model_cnn
    }

    scaler = StandardScaler()
    new_data_scaled = scaler.fit_transform(new_data)

    # Dictionary to store predictions
    model_predictions = {}
    formatted_model_predictions={}

    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find the model at the specified path: {model_path}")

        model_loaded = load(model_path)
        print("Label conditions dictionary:", label_conditions)

        if model_name == 'Model_CNN':
            # Assuming new_data is a DataFrame
            new_data_transformed = np.expand_dims(new_data_scaled, axis=2)
            numeric_predictions = model_loaded.predict(new_data_transformed).argmax(axis=-1)
        else:
            numeric_predictions = model_loaded.predict(new_data_scaled)
    
        
        print(f"Numeric predictions from {model_name}:", numeric_predictions)

        # Check if inverse_transform returns floating numbers and convert them accordingly
        inverse_transformed = label_encoder.inverse_transform(numeric_predictions)
        if np.issubdtype(inverse_transformed.dtype, np.floating):
            label_predictions = [str(int(float(label)/1000000)) for label in inverse_transformed]
        else:
            label_predictions = [str(label).lower() for label in inverse_transformed]
        
        print(f"Label predictions from {model_name}:", label_predictions)
        conditions = [label_conditions.get(label, "Unknown condition") for label in label_predictions]
        formatted_predictions = [f"Person {i+1} = {condition}" 
                                 for i, condition in enumerate(conditions)]
        
        # Store the formatted predictions in the dictionary
        formatted_model_predictions[model_name] = formatted_predictions
    
    # Print the formatted predictions
    for model_name, formatted_predictions in formatted_model_predictions.items():
        print(f"Predictions from {model_name}:")
        for prediction in formatted_predictions:
            print(prediction)
        print()  # Adds an empty line for better readability

    return model_predictions  # You can also choose to return the dictionary if needed elsewhere


def predict_on_training_data(model_name,label_conditions):

    model_paths = {
        'svc': model_svc,
        'random': model_rf,
        'logistic': model_lr,
        'knn': model_knn,
        'cnn': model_cnn
    }

    detailed_messages=[]
    
    if model_name not in model_paths:
        raise ValueError(f"The model name {model_name} is not recognized.")
    
    # Load the pre-fitted scaler, label encoder, and trained model
    model_path = model_paths[model_name]
    print("model path in train:",model_path)
    if not (os.path.exists(encoder_path) and os.path.exists(scaler_path) and os.path.exists(model_path)):
        raise FileNotFoundError("Required files for prediction are missing.")

    result_predict_train=[]
    scaler = load(scaler_path)
    label_encoder = load(encoder_path)
    clf = load(model_path)
    # Check file paths:
    print(f"Model file path: {model_path}")
    print(f"Scaler file path: {scaler_path}")
    print(f"Label encoder file path: {encoder_path}")

    # Number of classes
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    detailed_messages.append(f"Number of classes: {num_classes}")


    # Assuming X_test and y_test are defined and accessible
    num_samples_test = len(y_test)
    num_samples_train=len(y_train)
    original_num_samples = len(df)

    print(f"Number of samples (time points) in the recording: {len(df)}")
    print(f"Number of samples: {num_samples_test}")
    print(f"Number of samples: {num_samples_train}")

    detailed_messages.append(f"Original number of samples is {original_num_samples}.")
    detailed_messages.append("Data is split into 80% training and 20% testing.")
    detailed_messages.append(f"Number of training set samples: {num_samples_train}")
    detailed_messages.append(f"Number of testing set samples: {num_samples_test}")
    detailed_messages.append(f"Sum of training and testing samples: {num_samples_train + num_samples_test}")

    
    if (num_samples_train + num_samples_test) == original_num_samples:
        detailed_messages.append("The total number of training and testing samples matches the original number of samples.")
    else:
        detailed_messages.append("Warning: The total number of training and testing samples does not match the original number of samples.")


    #X_train_scaled = scaler.transform(X_train)
    #clf.fit(X_test,y_test)

    if model_name=='cnn':
        X_test_reshaped=np.expand_dims(X_test,axis=2)
        y_pred = clf.predict(X_test_reshaped)
        y_pred_train=y_pred.argmax(axis=-1)
    else:
        y_pred_train = clf.predict(X_test)


    print(f"Shape of y_pred_train before inverse_transform: {y_pred_train.shape}")
    y_train_inverse = label_encoder.inverse_transform(y_test)
    y_pred_train_inverse = label_encoder.inverse_transform(y_pred_train)
    print("Inverse Transformed Predictions:", y_pred_train_inverse)
    print("Label Conditions:", label_conditions)

    

    if np.issubdtype(y_pred_train_inverse.dtype, np.floating):
        label_predictions = [str(int(float(label)/1000000)) for label in y_pred_train_inverse]
    else:
        label_predictions = [str(label).lower() for label in y_pred_train_inverse]

    # Calculate the distribution of classes
    unique_classes, class_counts = np.unique(label_predictions, return_counts=True)
    class_distribution = dict(zip(unique_classes, class_counts))
    print("Class Distribution in Training Data:", class_distribution)
    detailed_messages.append(f"Class Distribution in Training Data: {class_distribution}")

    

    accuracy = accuracy_score(y_train_inverse, y_pred_train_inverse)*100
    precision = precision_score(y_train_inverse, y_pred_train_inverse, average='weighted')*100
    recall = recall_score(y_train_inverse, y_pred_train_inverse, average='weighted')*100
    f1 = f1_score(y_train_inverse, y_pred_train_inverse, average='weighted')*100  
    conf_mat = confusion_matrix(y_train_inverse, y_pred_train_inverse)


    # Convert labels to conditions
    conditions = [label_conditions.get(label, "Unknown condition") for label in label_predictions]
    correct_predictions = y_train_inverse == y_pred_train_inverse  # This creates a boolean array

    formatted_predictions_train = []
    # Accessing True Positives, False Positives, True Negatives, False Negatives
    TP = np.diag(conf_mat)  # True Positives are on the diagonal
    print("tp:",TP)
    FP = conf_mat.sum(axis=0) - TP  # False Positives are the column sum minus diagonal
    print("FP:",FP)
    FN = conf_mat.sum(axis=1) - TP  # False Negatives are the row sum minus diagonal
    print("FN:",FN)
    TN = conf_mat.sum() - (FP + FN + TP)  # True Negatives are total sum minus TP, FP, FN
    print("TN:",TN)

    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred_train)

    # Enhance the output to show detailed classification results
    for i, (index, true_label, predicted_label, condition) in enumerate(zip(test_indices, y_test_labels, y_pred_labels, conditions)):
        true_label = y_test_labels[i]
        predicted_label = y_pred_labels[i]
        
        if true_label == predicted_label:
            if true_label in TP:
                detail = 'True Negative'
            else:
                detail = 'True Positive'
        else:
            if predicted_label in FP:
                detail = 'False Negative'
            else:
                detail = 'False Positive'

        formatted_predictions_train.append(f"Data Index {index+2}: Condition = {condition} - {detail}")

    detailed_messages.append(f"Green Check means True Positive")
    detailed_messages.append(f"Blue Check means True Negative")
    detailed_messages.append(f"Yellow Check means False Positive")
    detailed_messages.append(f"Red Check means False Negative")

    
  

    # for message in formatted_predictions_train:
    #         print("Message:", message)    


    result_predict_train = {'Model': [], 'Accuracy': [], 'F1 Score': [], 'Precision': [], 'Recall': [], 'Confusion Matrix':[]}

    result_predict_train['Accuracy'].append(accuracy)
    result_predict_train['F1 Score'].append(f1)
    result_predict_train['Precision'].append(precision)
    result_predict_train['Recall'].append(recall)
    result_predict_train['Confusion Matrix'].append(conf_mat)
     

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Confusion Matrix:\n{conf_mat}')
    print("detailed message:",detailed_messages)

    return formatted_predictions_train,result_predict_train,detailed_messages


def load_and_predict_svc(new_data, label_conditions,y_original):
    result_predict=[]
    if not os.path.exists(encoder_path):
        raise FileNotFoundError("Could not find the label encoder at the specified path.")

    label_encoder = load(encoder_path)

    # Dictionary to store model paths
    model_paths = {
        'Model_SVC': model_svc,
    }

    # Load the scaler
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Could not find the scaler at the specified path.")
    
    scaler = load(scaler_path)
    new_data_scaled = scaler.transform(new_data)

    # Dictionary to store predictions
    model_predictions = {}
    formatted_model_predictions = {}

    # Dictionary to store all formatted predictions
    #model_svc_csv_prediction = {}

    print("y original:",len(y_original))

    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find the model at the specified path: {model_path}")

        
        model_loaded = load(model_path)


        numeric_predictions = model_loaded.predict(new_data_scaled)

        if len(y_original) > len(numeric_predictions):
            y_original = y_original[:len(numeric_predictions)]   

        
        # Convert numeric predictions to labels
        inverse_transformed = label_encoder.inverse_transform(numeric_predictions)
        if np.issubdtype(inverse_transformed.dtype, np.floating):
            label_predictions = [str(int(float(label)/1000000)) for label in inverse_transformed]
        else:
            label_predictions = [str(label).lower() for label in inverse_transformed]

        # Convert labels to conditions
        conditions = [label_conditions.get(label, "Unknown condition") for label in label_predictions]
        formatted_predictions = [f"Person {i+1} = {condition}" 
                                 for i, condition in enumerate(conditions)]
        
        # Store the formatted predictions in the dictionary
        formatted_model_predictions[model_name] = formatted_predictions

    # Calculate accuracy
    accuracy = accuracy_score(y_original, inverse_transformed)*100
    print("Accuracy of predictions:", accuracy)

    accuracy = accuracy_score(y_original, inverse_transformed)*100
    precision = precision_score(y_original, inverse_transformed, average='weighted')*100
    recall = recall_score(y_original, inverse_transformed, average='weighted')*100
    f1 = f1_score(y_original, inverse_transformed, average='weighted')*100  
    conf_mat = confusion_matrix(y_original, inverse_transformed)


    # Store all formatted predictions in the dictionary
    for model_name, formatted_predictions in formatted_model_predictions.items():
        model_svc_csv_prediction[model_name] = formatted_predictions

    result_predict = {'Accuracy': [], 'F1 Score': [], 'Precision': [], 'Recall': []}

    result_predict['Accuracy'].append(accuracy)
    result_predict['F1 Score'].append(f1)
    result_predict['Precision'].append(precision)
    result_predict['Recall'].append(recall)
     

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Confusion Matrix:\n{conf_mat}')    

    # for model_name, formatted_predictions in formatted_model_predictions.items():
    #     print(f"Predictions from {model_name}:")
    #     for prediction in formatted_predictions:
    #         print(prediction)
    #     print()      

    # Return the dictionary containing all model predictions
    return model_svc_csv_prediction


def load_and_predict_random(new_data, label_conditions,y_original):
    result_predict=[]

    if not os.path.exists(encoder_path):
        raise FileNotFoundError("Could not find the label encoder at the specified path.")

    label_encoder = load(encoder_path)

    # Dictionary to store model paths
    model_paths = {
        'Model_RF': model_rf,
    }

    scaler = StandardScaler()
    new_data_scaled = scaler.fit_transform(new_data)

    # Dictionary to store predictions
    model_predictions = {}
    formatted_model_predictions = {}

    # Dictionary to store all formatted predictions
    #model_random_csv_prediction = {}

    print("y original:",len(y_original))

    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find the model at the specified path: {model_path}")

        model_loaded = load(model_path)

        
        numeric_predictions = model_loaded.predict(new_data_scaled)

        if len(y_original) > len(numeric_predictions):
            y_original = y_original[:len(numeric_predictions)]  


        # Convert numeric predictions to labels
        inverse_transformed = label_encoder.inverse_transform(numeric_predictions)
        if np.issubdtype(inverse_transformed.dtype, np.floating):
            label_predictions = [str(int(float(label)/1000000)) for label in inverse_transformed]
        else:
            label_predictions = [str(label).lower() for label in inverse_transformed]

        # Convert labels to conditions
        conditions = [label_conditions.get(label, "Unknown condition") for label in label_predictions]
        formatted_predictions = [f"Person {i+1} = {condition}" 
                                 for i, condition in enumerate(conditions)]
        
        # Store the formatted predictions in the dictionary
        formatted_model_predictions[model_name] = formatted_predictions


    accuracy = np.sum(inverse_transformed == y_original) / len(y_original) * 100
    print("accuracy pred:",accuracy)
    precision = precision_score(y_original, inverse_transformed, average='weighted')*100
    recall = recall_score(y_original, inverse_transformed, average='weighted')*100
    f1 = f1_score(y_original, inverse_transformed, average='weighted')*100  
    conf_mat = confusion_matrix(y_original, inverse_transformed)
    

    # Store all formatted predictions in the dictionary
    for model_name, formatted_predictions in formatted_model_predictions.items():
        model_random_csv_prediction[model_name] = formatted_predictions

    
    result_predict = {'Accuracy': [], 'F1 Score': [], 'Precision': [], 'Recall': []}

    result_predict['Accuracy'].append(accuracy)
    result_predict['F1 Score'].append(f1)
    result_predict['Precision'].append(precision)
    result_predict['Recall'].append(recall)
     

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Confusion Matrix:\n{conf_mat}') 

    # for model_name, formatted_predictions in formatted_model_predictions.items():
    #     print(f"Predictions from {model_name}:")
    #     for prediction in formatted_predictions:
    #         print(prediction)
    #     print()

    # Return the dictionary containing all model predictions
    return model_random_csv_prediction

def load_and_predict_logisitc(new_data, label_conditions,y_original):
    result_predict=[]
    if not os.path.exists(encoder_path):
        raise FileNotFoundError("Could not find the label encoder at the specified path.")

    label_encoder = load(encoder_path)

    # Dictionary to store model paths
    model_paths = {
        'Model_LR': model_lr,
    }

    scaler = StandardScaler()
    new_data_scaled = scaler.fit_transform(new_data)

    # Dictionary to store predictions
    model_predictions = {}
    formatted_model_predictions = {}

    # Dictionary to store all formatted predictions
    #model_logistic_csv_prediction = {}

    
    print("y original:",len(y_original))

    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find the model at the specified path: {model_path}")

        model_loaded = load(model_path)

        
        numeric_predictions = model_loaded.predict(new_data_scaled)

        if len(y_original) > len(numeric_predictions):
            y_original = y_original[:len(numeric_predictions)] 

        # Convert numeric predictions to labels
        inverse_transformed = label_encoder.inverse_transform(numeric_predictions)
        if np.issubdtype(inverse_transformed.dtype, np.floating):
            label_predictions = [str(int(float(label)/1000000)) for label in inverse_transformed]
        else:
            label_predictions = [str(label).lower() for label in inverse_transformed]

        # Convert labels to conditions
        conditions = [label_conditions.get(label, "Unknown condition") for label in label_predictions]
        formatted_predictions = [f"Person {i+1} = {condition}" 
                                 for i, condition in enumerate(conditions)]
        
        # Store the formatted predictions in the dictionary
        formatted_model_predictions[model_name] = formatted_predictions

    accuracy = accuracy_score(y_original, inverse_transformed)*100
    precision = precision_score(y_original, inverse_transformed, average='weighted')*100
    recall = recall_score(y_original, inverse_transformed, average='weighted')*100
    f1 = f1_score(y_original, inverse_transformed, average='weighted')*100  
    conf_mat = confusion_matrix(y_original, inverse_transformed)
    

    # Store all formatted predictions in the dictionary
    for model_name, formatted_predictions in formatted_model_predictions.items():
        model_logistic_csv_prediction[model_name] = formatted_predictions

    result_predict = {'Accuracy': [], 'F1 Score': [], 'Precision': [], 'Recall': []}

    result_predict['Accuracy'].append(accuracy)
    result_predict['F1 Score'].append(f1)
    result_predict['Precision'].append(precision)
    result_predict['Recall'].append(recall)
     

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Confusion Matrix:\n{conf_mat}')     
    
    # for model_name, formatted_predictions in formatted_model_predictions.items():
    #     print(f"Predictions from {model_name}:")
    #     for prediction in formatted_predictions:
    #         print(prediction)
    #     print()

    # Return the dictionary containing all model predictions
    return model_logistic_csv_prediction


def load_and_predict_knn(new_data, label_conditions,y_original):
    result_predict=[]

    if not os.path.exists(encoder_path):
        raise FileNotFoundError("Could not find the label encoder at the specified path.")

    label_encoder = load(encoder_path)

    # Dictionary to store model paths
    model_paths = {
        'Model_KNN': model_knn,
    }

    scaler = StandardScaler()
    new_data_scaled = scaler.fit_transform(new_data)

    # Dictionary to store predictions
    model_predictions = {}
    formatted_model_predictions = {}

    # Dictionary to store all formatted predictions
    #model_knn_csv_prediction = {}

    print("y original:",len(y_original))

    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find the model at the specified path: {model_path}")

        model_loaded = load(model_path)

       
        numeric_predictions = model_loaded.predict(new_data_scaled)

        if len(y_original) > len(numeric_predictions):
            y_original = y_original[:len(numeric_predictions)]

        # Convert numeric predictions to labels
        inverse_transformed = label_encoder.inverse_transform(numeric_predictions)
        if np.issubdtype(inverse_transformed.dtype, np.floating):
            label_predictions = [str(int(float(label)/1000000)) for label in inverse_transformed]
        else:
            label_predictions = [str(label).lower() for label in inverse_transformed]

        # Convert labels to conditions
        conditions = [label_conditions.get(label, "Unknown condition") for label in label_predictions]
        formatted_predictions = [f"Person {i+1} = {condition}" 
                                 for i, condition in enumerate(conditions)]
        
        # Store the formatted predictions in the dictionary
        formatted_model_predictions[model_name] = formatted_predictions

    accuracy = accuracy_score(y_original, inverse_transformed)*100
    precision = precision_score(y_original, inverse_transformed, average='weighted')*100
    recall = recall_score(y_original, inverse_transformed, average='weighted')*100
    f1 = f1_score(y_original, inverse_transformed, average='weighted')*100  
    conf_mat = confusion_matrix(y_original, inverse_transformed)

    
    # Store all formatted predictions in the dictionary
    for model_name, formatted_predictions in formatted_model_predictions.items():
        model_knn_csv_prediction[model_name] = formatted_predictions

    
    result_predict = {'Accuracy': [], 'F1 Score': [], 'Precision': [], 'Recall': []}

    result_predict['Accuracy'].append(accuracy)
    result_predict['F1 Score'].append(f1)
    result_predict['Precision'].append(precision)
    result_predict['Recall'].append(recall)
     

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Confusion Matrix:\n{conf_mat}')     
    
    # for model_name, formatted_predictions in formatted_model_predictions.items():
    #     print(f"Predictions from {model_name}:")
    #     for prediction in formatted_predictions:
    #         print(prediction)
    #     print()

    # Return the dictionary containing all model predictions
    return model_knn_csv_prediction

def load_and_predict_cnn(new_data, label_conditions,y_original):
    result_predict=[]

    if not os.path.exists(encoder_path):
        raise FileNotFoundError("Could not find the label encoder at the specified path.")

    label_encoder = load(encoder_path)

    # Dictionary to store model paths
    model_paths = {
        'Model_CNN': model_cnn
    }

    scaler = StandardScaler()
    new_data_scaled = scaler.fit_transform(new_data)

    # Dictionary to store predictions
    model_predictions = {}
    formatted_model_predictions = {}

    # Dictionary to store all formatted predictions
    #model_cnn_csv_prediction = {}

    print("y original:",len(y_original))

    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find the model at the specified path: {model_path}")

        model_loaded = load(model_path)

        if model_name == 'Model_CNN':
            # Assuming new_data is a DataFrame
            new_data_transformed = np.expand_dims(new_data_scaled, axis=2)
            numeric_predictions = model_loaded.predict(new_data_transformed).argmax(axis=-1)
        else:
            numeric_predictions = model_loaded.predict(new_data_scaled)

        if len(y_original) > len(numeric_predictions):
            y_original = y_original[:len(numeric_predictions)]     

        # Convert numeric predictions to labels
        inverse_transformed = label_encoder.inverse_transform(numeric_predictions)
        if np.issubdtype(inverse_transformed.dtype, np.floating):
            label_predictions = [str(int(float(label)/1000000)) for label in inverse_transformed]
        else:
            label_predictions = [str(label).lower() for label in inverse_transformed]

        # Convert labels to conditions
        conditions = [label_conditions.get(label, "Unknown condition") for label in label_predictions]
        formatted_predictions = [f"Person {i+1} = {condition}" 
                                 for i, condition in enumerate(conditions)]
        
        # Store the formatted predictions in the dictionary
        formatted_model_predictions[model_name] = formatted_predictions

    accuracy = accuracy_score(y_original, inverse_transformed)*100
    precision = precision_score(y_original, inverse_transformed, average='weighted')*100
    recall = recall_score(y_original, inverse_transformed, average='weighted')*100
    f1 = f1_score(y_original, inverse_transformed, average='weighted')*100  
    conf_mat = confusion_matrix(y_original, inverse_transformed)
    

    # Store all formatted predictions in the dictionary
    for model_name, formatted_predictions in formatted_model_predictions.items():
        model_cnn_csv_prediction[model_name] = formatted_predictions

    result_predict = {'Accuracy': [], 'F1 Score': [], 'Precision': [], 'Recall': []}

    result_predict['Accuracy'].append(accuracy)
    result_predict['F1 Score'].append(f1)
    result_predict['Precision'].append(precision)
    result_predict['Recall'].append(recall)
     

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Confusion Matrix:\n{conf_mat}')     
    
    # for model_name, formatted_predictions in formatted_model_predictions.items():
    #     print(f"Predictions from {model_name}:")
    #     for prediction in formatted_predictions:
    #         print(prediction)
    #     print()

    # Return the dictionary containing all model predictions
    return model_cnn_csv_prediction



def mat_modeling(subject_identifier,features_df,labels):
    subject_scores = defaultdict(lambda: defaultdict(list))
    subject_data = {}
    accuracy_mat=[]


    if subject_identifier not in subject_data:
            subject_data[subject_identifier] = {'features': pd.DataFrame(), 'labels': []}
    # Append features and labels to the subject's data
    subject_data[subject_identifier]['features'] = pd.concat([subject_data[subject_identifier]['features'], features_df], ignore_index=True)
    subject_data[subject_identifier]['labels'].extend(labels) 
    trial_labels = subject_data[subject_identifier]['labels']
    trial_labels_sentences = [label_descriptions[label] for label in trial_labels]
    #print(f"Labels for each trial for subject {subject_identifier}: {trial_labels_sentences}")  # This prints the labels as sentences
    

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
                'Logistic Regression': LogisticRegression(),
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
 
        
        for clf_name, clf in classifiers.items():
            if clf_name == 'CNN':
                clf.fit(X_reshaped, y)
                model_path = os.path.join(model_directory_mat, f'{clf_name.lower()}_model.h5')
                clf.model_.save(model_path)
            else:
                clf.fit(X_scaled, y)
                model_path = os.path.join(model_directory_mat, f'{clf_name.lower().replace(" ", "_")}_model.joblib')
                dump(clf, model_path)

        # Now calculate the overall accuracy for each subject by taking the mean of the aggregated scores
        overall_accuracy_dict = defaultdict(dict)
        for base_subject_identifier, clf_scores in subject_scores.items():
            for clf_name, scores in clf_scores.items():
                overall_accuracy = np.mean(scores)
                overall_accuracy_dict[base_subject_identifier][clf_name] = overall_accuracy
                accuracy_message = f"Overall accuracy for {clf_name} on subject {base_subject_identifier}: {overall_accuracy * 100:.2f}%"
                accuracy_mat.append(accuracy_message)  # Store the message in the list
                print(f"Overall accuracy for {clf_name} on subject {base_subject_identifier}: {overall_accuracy * 100:.2f}%")

        return accuracy_mat
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

def mat_modeling_svc(subject_identifier, features_df, labels):
    subject_scores = defaultdict(lambda: defaultdict(list))
    subject_data = {}
    accuracy_mat_svc = []
    trial_labels_sentences_array = []  # Array to store labels for each trial


    if subject_identifier not in subject_data:
        subject_data[subject_identifier] = {'features': pd.DataFrame(), 'labels': []}
    
    # Append features and labels to the subject's data
    subject_data[subject_identifier]['features'] = pd.concat([subject_data[subject_identifier]['features'], features_df], ignore_index=True)
    subject_data[subject_identifier]['labels'].extend(labels)
    trial_labels = subject_data[subject_identifier]['labels']
    trial_labels_sentences = [label_descriptions[label] for label in trial_labels]    
    labels_message = f"Labels for each trial for subject {subject_identifier}: {trial_labels_sentences}"
    trial_labels_sentences_array.append(labels_message)
    
    
    # Loop over your subject_data to perform cross-validation for each subject
    for subject_identifier, data in subject_data.items():
        X = data['features'].iloc[:, :-1]  # Features: all columns except the last
        y = LabelEncoder().fit_transform(data['features'].iloc[:, -1])  # Labels: the last column
        X_scaled = StandardScaler().fit_transform(X)  # Standardize features
        class_distribution = Counter(y)
        print("Class distribution:", class_distribution)

        for label, count in class_distribution.items():
            if count < 5:
                print(f"Class {label} has fewer than 5 samples: {count}")

        # Define classifiers
        classifiers = {
            'SVC': SVC(),
        }

        # Define scoring metrics
        scoring_metrics = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'f1_score': make_scorer(f1_score, average='weighted'),
        }

        min_samples_per_class = min(np.bincount(y))

        # Perform cross-validation and collect metrics for each classifier
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        for clf_name, clf in classifiers.items():
            clf.fit(X_scaled, y)
            model_path = os.path.join(model_directory_mat, f'{clf_name.lower().replace(" ", "_")}_model.joblib')
            print("Model path name:",model_path)
            dump(clf, model_path)

        for clf_name, clf in classifiers.items():
            cv_results = cross_validate(clf, X_scaled, y, cv=skf, scoring=scoring_metrics)
            
            # Calculate mean of each metric
            mean_accuracy = np.mean(cv_results['test_accuracy'])
            mean_precision = np.mean(cv_results['test_precision'])
            mean_recall = np.mean(cv_results['test_recall'])
            mean_f1_score = np.mean(cv_results['test_f1_score'])

            # Get the base subject identifier without session (e.g., "sub-001")
            base_subject_identifier = subject_identifier.rsplit('_', 1)[0]
            subject_scores[base_subject_identifier][clf_name].extend([mean_accuracy, mean_precision, mean_recall, mean_f1_score])

            # Generate the message with all metrics
            accuracy_message = (f"Metrics for {clf_name} on subject {base_subject_identifier}: "
                                f"Accuracy: {mean_accuracy * 100:.2f}%, "
                                f"Precision: {mean_precision * 100:.2f}%, "
                                f"Recall: {mean_recall * 100:.2f}%, "
                                f"F1 Score: {mean_f1_score * 100:.2f}%")
            accuracy_mat_svc.append(accuracy_message)  # Store the message in the list
            print(accuracy_message)
    #print(trial_labels_sentences_array)
    return accuracy_mat_svc,trial_labels_sentences_array


def train_test_split_models(subject_identifier, features_df, labels, model_name, test_size=0.2):

    labels = np.array(labels)  # Convert labels to a NumPy array if it's not already

    metrics={}

    # Make sure that features and labels have the same length
    min_length = min(len(features_df), len(labels))
    features_df = features_df.iloc[:min_length, :]
    labels = labels[:min_length]

    # Now that features and labels are consistent, you can proceed
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    for train_idx, test_idx in stratified_split.split(features_df.iloc[:, :-1], labels):
        X_train = features_df.iloc[train_idx, :-1]  # Features: all columns except the last
        X_test = features_df.iloc[test_idx, :-1]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

    # Standardize the features
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit LabelEncoder on the training labels and transform both train and test labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Define the model file name based on the model_name parameter
    model_file_mapping = {
        'svc': 'svc_model.joblib',
        'random': 'randomforest_model.joblib',
        'logistic':'logistic_regression_model.joblib',
        'knn': 'knn_model.joblib',
        'cnn': 'cnn_model.h5',

    }

    # Get the file name for the specific model
    model_file_name = model_file_mapping.get(model_name)

    if not model_file_name:
        raise ValueError("Invalid model name provided.")

    # Define the path to the model file
    model_directory_mat = 'saved_models_mat'
    model_path = os.path.join(model_directory_mat, model_file_name)

    # Load the selected model and make predictions
    if model_name == 'cnn':  # Keras model has a different file extension
        model = load_model(model_path)
        # Reshape the data for CNN
        X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        predictions = model.predict(X_test_reshaped)
        y_pred_encoded = np.argmax(predictions, axis=1)
    else:  # Sklearn models
        model = load(model_path)
        y_pred_encoded = model.predict(X_test_scaled)

    # Calculate accuracy, precision, recall, and F1 score
    test_accuracy = accuracy_score(y_test_encoded, y_pred_encoded)*100
    test_precision = precision_score(y_test_encoded, y_pred_encoded, average='weighted')*100
    test_recall = recall_score(y_test_encoded, y_pred_encoded, average='weighted')*100
    test_f1_score = f1_score(y_test_encoded, y_pred_encoded, average='weighted')*100
    test_confusion_matrix = confusion_matrix(y_test_encoded, y_pred_encoded)

    # Print metrics
    print("test accuracy:", test_accuracy)
    print("test precision:", test_precision)
    print("test recall:", test_recall)
    print("test f1 score:", test_f1_score)
    print("confusion matrix:\n", test_confusion_matrix)

    # Translate the encoded labels back into the original label descriptions
    label_descriptions = label_encoder.classes_
    y_pred_descriptions = label_descriptions[y_pred_encoded]
    y_test_descriptions = label_descriptions[y_test_encoded]

    # Prepare the predictions info
    all_subjects_predictions = []
    for true, pred in zip(y_test_descriptions, y_pred_descriptions):
        correctness = "Correct" if true == pred else "Wrong"
        prediction_info = {
            'subject_identifier': subject_identifier,
            'actual_label': true,
            'predicted_label': pred,
            'correctness': correctness
        }
        all_subjects_predictions.append(prediction_info)

    # Prepare a dictionary to hold all metrics
    metrics = {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1_score,
        'confusion_matrix': test_confusion_matrix.tolist()  # Convert to list for JSON serializability if necessary
    }    
    print("all subject prediction:",all_subjects_predictions)    

    # Return the collection of predictions for all subject_identifiers
    return all_subjects_predictions,metrics

def mat_modeling_random(subject_identifier,features_df,labels):
    subject_scores = defaultdict(lambda: defaultdict(list))
    subject_data = {}
    accuracy_mat_random=[]
    trial_labels_sentences_array = []  # Array to store labels for each trial


    if subject_identifier not in subject_data:
            subject_data[subject_identifier] = {'features': pd.DataFrame(), 'labels': []}
    # Append features and labels to the subject's data
    subject_data[subject_identifier]['features'] = pd.concat([subject_data[subject_identifier]['features'], features_df], ignore_index=True)
    subject_data[subject_identifier]['labels'].extend(labels)
    trial_labels = subject_data[subject_identifier]['labels']
    trial_labels_sentences = [label_descriptions[label] for label in trial_labels]
    labels_message = f"Labels for each trial for subject {subject_identifier}: {trial_labels_sentences}"
    trial_labels_sentences_array.append(labels_message)
    
     
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

            # Define classifiers
        classifiers = {
                'RandomForest': RandomForestClassifier(),
            }

         # Define scoring metrics
        scoring_metrics = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'f1_score': make_scorer(f1_score, average='weighted'),
        }

        for clf_name, clf in classifiers.items():
            clf.fit(X_scaled, y)
            model_path = os.path.join(model_directory_mat, f'{clf_name.lower().replace(" ", "_")}_model.joblib')
            dump(clf, model_path)

        # Perform cross-validation and collect metrics for each classifier
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        for clf_name, clf in classifiers.items():
            cv_results = cross_validate(clf, X_scaled, y, cv=skf, scoring=scoring_metrics)
            
            # Calculate mean of each metric
            mean_accuracy = np.mean(cv_results['test_accuracy'])
            mean_precision = np.mean(cv_results['test_precision'])
            mean_recall = np.mean(cv_results['test_recall'])
            mean_f1_score = np.mean(cv_results['test_f1_score'])

            # Get the base subject identifier without session (e.g., "sub-001")
            base_subject_identifier = subject_identifier.rsplit('_', 1)[0]
            subject_scores[base_subject_identifier][clf_name].extend([mean_accuracy, mean_precision, mean_recall, mean_f1_score])

            # Generate the message with all metrics
            accuracy_message = (f"Metrics for {clf_name} on subject {base_subject_identifier}: "
                                f"Accuracy: {mean_accuracy * 100:.2f}%, "
                                f"Precision: {mean_precision * 100:.2f}%, "
                                f"Recall: {mean_recall * 100:.2f}%, "
                                f"F1 Score: {mean_f1_score * 100:.2f}%")
            accuracy_mat_random.append(accuracy_message)  # Store the message in the list
            print(accuracy_message)
        return accuracy_mat_random,trial_labels_sentences_array
    

def mat_modeling_logistic(subject_identifier,features_df,labels):
    subject_scores = defaultdict(lambda: defaultdict(list))
    subject_data = {}
    accuracy_mat_logistic=[]
    trial_labels_sentences_array = []  # Array to store labels for each trial


    if subject_identifier not in subject_data:
            subject_data[subject_identifier] = {'features': pd.DataFrame(), 'labels': []}
    # Append features and labels to the subject's data
    subject_data[subject_identifier]['features'] = pd.concat([subject_data[subject_identifier]['features'], features_df], ignore_index=True)
    subject_data[subject_identifier]['labels'].extend(labels) 
    trial_labels = subject_data[subject_identifier]['labels']
    trial_labels_sentences = [label_descriptions[label] for label in trial_labels]
    labels_message = f"Labels for each trial for subject {subject_identifier}: {trial_labels_sentences}"
    trial_labels_sentences_array.append(labels_message)
    
    
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

            # Define classifiers
        classifiers = {
                'Logistic Regression': LogisticRegression(),
            }

         # Define scoring metrics
        scoring_metrics = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'f1_score': make_scorer(f1_score, average='weighted'),
        }

        for clf_name, clf in classifiers.items():
            clf.fit(X_scaled, y)
            model_path = os.path.join(model_directory_mat, f'{clf_name.lower().replace(" ", "_")}_model.joblib')
            dump(clf, model_path)

        # Perform cross-validation and collect metrics for each classifier
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        for clf_name, clf in classifiers.items():
            cv_results = cross_validate(clf, X_scaled, y, cv=skf, scoring=scoring_metrics)
            
            # Calculate mean of each metric
            mean_accuracy = np.mean(cv_results['test_accuracy'])
            mean_precision = np.mean(cv_results['test_precision'])
            mean_recall = np.mean(cv_results['test_recall'])
            mean_f1_score = np.mean(cv_results['test_f1_score'])

            # Get the base subject identifier without session (e.g., "sub-001")
            base_subject_identifier = subject_identifier.rsplit('_', 1)[0]
            subject_scores[base_subject_identifier][clf_name].extend([mean_accuracy, mean_precision, mean_recall, mean_f1_score])

            # Generate the message with all metrics
            accuracy_message = (f"Metrics for {clf_name} on subject {base_subject_identifier}: "
                                f"Accuracy: {mean_accuracy * 100:.2f}%, "
                                f"Precision: {mean_precision * 100:.2f}%, "
                                f"Recall: {mean_recall * 100:.2f}%, "
                                f"F1 Score: {mean_f1_score * 100:.2f}%")
            accuracy_mat_logistic.append(accuracy_message)  # Store the message in the list
            print(accuracy_message)
        return accuracy_mat_logistic,trial_labels_sentences_array


def mat_modeling_knn(subject_identifier,features_df,labels):
    subject_scores = defaultdict(lambda: defaultdict(list))
    subject_data = {}
    accuracy_mat_knn=[]
    trial_labels_sentences_array = []  # Array to store labels for each trial


    if subject_identifier not in subject_data:
            subject_data[subject_identifier] = {'features': pd.DataFrame(), 'labels': []}
    # Append features and labels to the subject's data
    subject_data[subject_identifier]['features'] = pd.concat([subject_data[subject_identifier]['features'], features_df], ignore_index=True)
    subject_data[subject_identifier]['labels'].extend(labels) 
    trial_labels = subject_data[subject_identifier]['labels']
    trial_labels_sentences = [label_descriptions[label] for label in trial_labels]
    labels_message = f"Labels for each trial for subject {subject_identifier}: {trial_labels_sentences}"
    trial_labels_sentences_array.append(labels_message)
    
    
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

            # Define classifiers
        classifiers = {
                'KNN':KNeighborsClassifier(n_neighbors=5) ,
            }

          
         # Define scoring metrics
        scoring_metrics = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'f1_score': make_scorer(f1_score, average='weighted'),
        }

        for clf_name, clf in classifiers.items():
            clf.fit(X_scaled, y)
            model_path = os.path.join(model_directory_mat, f'{clf_name.lower().replace(" ", "_")}_model.joblib')
            dump(clf, model_path)

        # Perform cross-validation and collect metrics for each classifier
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        for clf_name, clf in classifiers.items():
            cv_results = cross_validate(clf, X_scaled, y, cv=skf, scoring=scoring_metrics)
            
            # Calculate mean of each metric
            mean_accuracy = np.mean(cv_results['test_accuracy'])
            mean_precision = np.mean(cv_results['test_precision'])
            mean_recall = np.mean(cv_results['test_recall'])
            mean_f1_score = np.mean(cv_results['test_f1_score'])

            # Get the base subject identifier without session (e.g., "sub-001")
            base_subject_identifier = subject_identifier.rsplit('_', 1)[0]
            subject_scores[base_subject_identifier][clf_name].extend([mean_accuracy, mean_precision, mean_recall, mean_f1_score])

            # Generate the message with all metrics
            accuracy_message = (f"Metrics for {clf_name} on subject {base_subject_identifier}: "
                                f"Accuracy: {mean_accuracy * 100:.2f}%, "
                                f"Precision: {mean_precision * 100:.2f}%, "
                                f"Recall: {mean_recall * 100:.2f}%, "
                                f"F1 Score: {mean_f1_score * 100:.2f}%")
            accuracy_mat_knn.append(accuracy_message)  # Store the message in the list
            print(accuracy_message)
        return accuracy_mat_knn,trial_labels_sentences_array       
    
def mat_modeling_cnn(subject_identifier,features_df,labels):
    subject_scores = defaultdict(lambda: defaultdict(list))
    subject_data = {}
    accuracy_mat_cnn=[]
    trial_labels_sentences_array = []  # Array to store labels for each trial


    if subject_identifier not in subject_data:
            subject_data[subject_identifier] = {'features': pd.DataFrame(), 'labels': []}
    # Append features and labels to the subject's data
    subject_data[subject_identifier]['features'] = pd.concat([subject_data[subject_identifier]['features'], features_df], ignore_index=True)
    subject_data[subject_identifier]['labels'].extend(labels) 
    trial_labels = subject_data[subject_identifier]['labels']
    trial_labels_sentences = [label_descriptions[label] for label in trial_labels]
    labels_message = f"Labels for each trial for subject {subject_identifier}: {trial_labels_sentences}"
    trial_labels_sentences_array.append(labels_message)
    
    
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
                'CNN': cnn_classifier
            }

            # Reshape the input data for CNN
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

        scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1_score': make_scorer(f1_score, average='weighted', zero_division=0),
    }
        
        for clf_name, clf in classifiers.items():
            if clf_name == 'CNN':
                clf.fit(X_reshaped, y)
                model_path = os.path.join(model_directory_mat, f'{clf_name.lower()}_model.h5')
                clf.model_.save(model_path)

            # Perform cross-validation and collect accuracy for each classifier
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        for clf_name, clf in classifiers.items():
            if clf_name=='CNN':
                cv_results = cross_validate(clf, X_reshaped, y, cv=skf, scoring=scoring_metrics)    
                
            # Calculate mean of each metric
            mean_accuracy = np.mean(cv_results['test_accuracy'])
            mean_precision = np.mean(cv_results['test_precision'])
            mean_recall = np.mean(cv_results['test_recall'])
            mean_f1_score = np.mean(cv_results['test_f1_score'])

            # Get the base subject identifier without session (e.g., "sub-001")
            base_subject_identifier = subject_identifier.rsplit('_', 1)[0]
            subject_scores[base_subject_identifier][clf_name].extend([mean_accuracy, mean_precision, mean_recall, mean_f1_score])

            # Generate the message with all metrics
            accuracy_message = (f"Metrics for {clf_name} on subject {base_subject_identifier}: "
                                f"Accuracy: {mean_accuracy * 100:.2f}%, "
                                f"Precision: {mean_precision * 100:.2f}%, "
                                f"Recall: {mean_recall * 100:.2f}%, "
                                f"F1 Score: {mean_f1_score * 100:.2f}%")
            accuracy_mat_cnn.append(accuracy_message)  # Store the message in the list
            print(accuracy_message)
        return accuracy_mat_cnn,trial_labels_sentences_array   
 
def get_label_text():
    global label_conditions
    for file_path in file_paths:
        if file_path.endswith('.txt'):
            txt_files = glob(os.path.join(file_path, '*.txt'))
            print("Txt files:", txt_files) 
            print("text entered")
            # Assuming there's only one TXT file in the directory
            label_conditions = read_label_conditions(file_path) 

results_array = []


def predict_movement(features_df,subject_identifier):

    
    # Read, preprocess, and extract features just like before
    
    X_test = features_df.iloc[:, :-1]  # Features
    y_test = features_df.iloc[:, -1]  # Labels

    # Standardize the features using a scaler fitted on the training data
    scaler = StandardScaler().fit(X_test)
    X_test_scaled = scaler.transform(X_test)

    # Prepare a label encoder for converting string labels to integers, if necessary
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)

    model_directory_mat = 'saved_models_mat'

    # List of models to predict with
    model_files = [
        'randomforest_model.joblib',
        'svc_model.joblib',
        'logistic_regression_model.joblib',
        'knn_model.joblib',
        'cnn_model.h5'  # Keras model has a different file extension
    ]

    predictions_output = {}

    # Iterate over each model file, load the model, and make predictions
    for model_file in model_files:
        model_path = os.path.join(model_directory_mat, model_file)
        model_name = model_file.replace('_model.joblib', '').replace('.h5', '')

        # Condition to check if the model is Keras model or not
        if model_file.endswith('.h5'):  # Keras model
            model = load_model(model_path)
            # Reshape the data for CNN
            X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
            predictions = model.predict(X_test_reshaped)
            predicted_classes = np.argmax(predictions, axis=1)
        else:  # Sklearn models
            model = load(model_path)
            predicted_classes = model.predict(X_test_scaled)

        # Convert numeric predictions back to original labels if necessary
        predicted_labels = label_encoder.inverse_transform(predicted_classes)

        # Convert numeric predictions to descriptive sentences
        predicted_sentences = [label_descriptions.get(label, "Unknown movement") for label in predicted_labels]

        # Store the predictions for the current model as sentences
        predictions_output[model_name] = predicted_sentences

    all_predictions = {subject_identifier: predictions_output}

    # Initialize an array to hold the results
    prediction_results = []

    # Process the predictions to generate a structured result
    for subject_id, models_dict in all_predictions.items():
        subject_result = {'Subject Identifier': subject_id, 'Models': []}
        for model_name, predictions in models_dict.items():
            model_result = {'Model Name': model_name, 'Predictions': []}
            for i, pred in enumerate(predictions, 1):
                prediction_text = f"Trial {i}: {pred}"
                model_result['Predictions'].append(prediction_text)
            subject_result['Models'].append(model_result)
        prediction_results.append(subject_result)

    return prediction_results

def predict_movement_svc(features_df,subject_identifier):

    
    # Read, preprocess, and extract features just like before
    
    X_test = features_df.iloc[:, :-1]  # Features
    y_test = features_df.iloc[:, -1]  # Labels

    # Standardize the features using a scaler fitted on the training data
    scaler = StandardScaler().fit(X_test)
    X_test_scaled = scaler.transform(X_test)

    # Prepare a label encoder for converting string labels to integers, if necessary
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)

    model_directory_mat = 'saved_models_mat'

    # List of models to predict with
    model_files = [
        'svc_model.joblib',

    ]

    predictions_output = {}
    metrics_results={}

    # Iterate over each model file, load the model, and make predictions
    for model_file in model_files:
        model_path = os.path.join(model_directory_mat, model_file)
        model_name = model_file.replace('_model.joblib', '').replace('.h5', '')

        # Condition to check if the model is Keras model or not
        if model_file.endswith('.h5'):  # Keras model
            model = load_model(model_path)
            # Reshape the data for CNN
            X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
            predictions = model.predict(X_test_reshaped)
            predicted_classes = np.argmax(predictions, axis=1)
        else:  # Sklearn models
            model = load(model_path)
            predicted_classes = model.predict(X_test_scaled)

        # Convert numeric predictions back to original labels if necessary
        predicted_labels = label_encoder.inverse_transform(predicted_classes)

        accuracy = accuracy_score(y_test_encoded, predicted_classes)*100
        precision = precision_score(y_test_encoded, predicted_classes, average='weighted')*100
        recall = recall_score(y_test_encoded, predicted_classes, average='weighted')*100
        f1 = f1_score(y_test_encoded, predicted_classes, average='weighted')*100
        conf_matrix = confusion_matrix(y_test_encoded, predicted_classes)

        print(f"Model: {model_name} - Accuracy: {accuracy}")

        # Store metrics in the dictionary
        metrics_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist()  # Convert numpy array to list for JSON serialization
        }

        # Convert numeric predictions to descriptive sentences
        predicted_sentences = [label_descriptions.get(label, "Unknown movement") for label in predicted_labels]

        # Store the predictions for the current model as sentences
        predictions_output[model_name] = predicted_sentences

    all_predictions = {subject_identifier: predictions_output}

    # Initialize an array to hold the results
    prediction_results = []

    # Process the predictions to generate a structured result
    for subject_id, models_dict in all_predictions.items():
        subject_result = {'Subject Identifier': subject_id, 'Models': []}
        for model_name, predictions in models_dict.items():
            model_result = {'Model Name': model_name, 'Predictions': []}
            for i, pred in enumerate(predictions, 1):
                prediction_text = f"Trial {i}: {pred}"
                model_result['Predictions'].append(prediction_text)
            subject_result['Models'].append(model_result)
        prediction_results.append(subject_result)

    return prediction_results,metrics_results   

def predict_movement_random(features_df,subject_identifier):

    
    # Read, preprocess, and extract features just like before
    
    X_test = features_df.iloc[:, :-1]  # Features
    y_test = features_df.iloc[:, -1]  # Labels

    # Standardize the features using a scaler fitted on the training data
    scaler = StandardScaler().fit(X_test)
    X_test_scaled = scaler.transform(X_test)

    # Prepare a label encoder for converting string labels to integers, if necessary
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)

    model_directory_mat = 'saved_models_mat'

    # List of models to predict with
    model_files = [
        'randomforest_model.joblib',
    ]

    predictions_output = {}
    metrics_results={}

    # Iterate over each model file, load the model, and make predictions
    for model_file in model_files:
        model_path = os.path.join(model_directory_mat, model_file)
        model_name = model_file.replace('_model.joblib', '').replace('.h5', '')

        # Condition to check if the model is Keras model or not
        if model_file.endswith('.h5'):  # Keras model
            model = load_model(model_path)
            # Reshape the data for CNN
            X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
            predictions = model.predict(X_test_reshaped)
            predicted_classes = np.argmax(predictions, axis=1)
        else:  # Sklearn models
            model = load(model_path)
            predicted_classes = model.predict(X_test_scaled)

        # Convert numeric predictions back to original labels if necessary
        predicted_labels = label_encoder.inverse_transform(predicted_classes)

        accuracy = accuracy_score(y_test_encoded, predicted_classes)*100
        precision = precision_score(y_test_encoded, predicted_classes, average='weighted')*100
        recall = recall_score(y_test_encoded, predicted_classes, average='weighted')*100
        f1 = f1_score(y_test_encoded, predicted_classes, average='weighted')*100
        conf_matrix = confusion_matrix(y_test_encoded, predicted_classes)

        # Store metrics in the dictionary
        metrics_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist()  # Convert numpy array to list for JSON serialization
        }

        print(f"Model: {model_name} - Accuracy: {accuracy}")

        # Convert numeric predictions to descriptive sentences
        predicted_sentences = [label_descriptions.get(label, "Unknown movement") for label in predicted_labels]

        # Store the predictions for the current model as sentences
        predictions_output[model_name] = predicted_sentences

    all_predictions = {subject_identifier: predictions_output}

    # Initialize an array to hold the results
    prediction_results = []

    # Process the predictions to generate a structured result
    for subject_id, models_dict in all_predictions.items():
        subject_result = {'Subject Identifier': subject_id, 'Models': []}
        for model_name, predictions in models_dict.items():
            model_result = {'Model Name': model_name, 'Predictions': []}
            for i, pred in enumerate(predictions, 1):
                prediction_text = f"Trial {i}: {pred}"
                model_result['Predictions'].append(prediction_text)
            subject_result['Models'].append(model_result)
        prediction_results.append(subject_result)

    return prediction_results,metrics_results 

def predict_movement_logistic(features_df,subject_identifier):

    
    # Read, preprocess, and extract features just like before
    
    X_test = features_df.iloc[:, :-1]  # Features
    y_test = features_df.iloc[:, -1]  # Labels

    # Standardize the features using a scaler fitted on the training data
    scaler = StandardScaler().fit(X_test)
    X_test_scaled = scaler.transform(X_test)

    # Prepare a label encoder for converting string labels to integers, if necessary
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)

    model_directory_mat = 'saved_models_mat'

    # List of models to predict with
    model_files = [
        'logistic_regression_model.joblib',
    ]

    predictions_output = {}
    metrics_results={}


    # Iterate over each model file, load the model, and make predictions
    for model_file in model_files:
        model_path = os.path.join(model_directory_mat, model_file)
        model_name = model_file.replace('_model.joblib', '').replace('.h5', '')

        # Condition to check if the model is Keras model or not
        if model_file.endswith('.h5'):  # Keras model
            model = load_model(model_path)
            # Reshape the data for CNN
            X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
            predictions = model.predict(X_test_reshaped)
            predicted_classes = np.argmax(predictions, axis=1)
        else:  # Sklearn models
            model = load(model_path)
            predicted_classes = model.predict(X_test_scaled)

        # Convert numeric predictions back to original labels if necessary
        predicted_labels = label_encoder.inverse_transform(predicted_classes)

        accuracy = accuracy_score(y_test_encoded, predicted_classes)*100
        precision = precision_score(y_test_encoded, predicted_classes, average='weighted')*100
        recall = recall_score(y_test_encoded, predicted_classes, average='weighted')*100
        f1 = f1_score(y_test_encoded, predicted_classes, average='weighted')*100
        conf_matrix = confusion_matrix(y_test_encoded, predicted_classes)

        # Store metrics in the dictionary
        metrics_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist()  # Convert numpy array to list for JSON serialization
        }

        print(f"Model: {model_name} - Accuracy: {accuracy}")

        # Convert numeric predictions to descriptive sentences
        predicted_sentences = [label_descriptions.get(label, "Unknown movement") for label in predicted_labels]

        # Store the predictions for the current model as sentences
        predictions_output[model_name] = predicted_sentences

    all_predictions = {subject_identifier: predictions_output}

    # Initialize an array to hold the results
    prediction_results = []

    # Process the predictions to generate a structured result
    for subject_id, models_dict in all_predictions.items():
        subject_result = {'Subject Identifier': subject_id, 'Models': []}
        for model_name, predictions in models_dict.items():
            model_result = {'Model Name': model_name, 'Predictions': []}
            for i, pred in enumerate(predictions, 1):
                prediction_text = f"Trial {i}: {pred}"
                model_result['Predictions'].append(prediction_text)
            subject_result['Models'].append(model_result)
        prediction_results.append(subject_result)

    return prediction_results,metrics_results 

def predict_movement_knn(features_df,subject_identifier):

    
    # Read, preprocess, and extract features just like before
    
    X_test = features_df.iloc[:, :-1]  # Features
    y_test = features_df.iloc[:, -1]  # Labels

    # Standardize the features using a scaler fitted on the training data
    scaler = StandardScaler().fit(X_test)
    X_test_scaled = scaler.transform(X_test)

    # Prepare a label encoder for converting string labels to integers, if necessary
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)

    model_directory_mat = 'saved_models_mat'

    # List of models to predict with
    model_files = [
        'knn_model.joblib',
    ]

    predictions_output = {}
    metrics_results={}

    # Iterate over each model file, load the model, and make predictions
    for model_file in model_files:
        model_path = os.path.join(model_directory_mat, model_file)
        model_name = model_file.replace('_model.joblib', '').replace('.h5', '')

        # Condition to check if the model is Keras model or not
        if model_file.endswith('.h5'):  # Keras model
            model = load_model(model_path)
            # Reshape the data for CNN
            X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
            predictions = model.predict(X_test_reshaped)
            predicted_classes = np.argmax(predictions, axis=1)
        else:  # Sklearn models
            model = load(model_path)
            predicted_classes = model.predict(X_test_scaled)

        # Convert numeric predictions back to original labels if necessary
        predicted_labels = label_encoder.inverse_transform(predicted_classes)

        accuracy = accuracy_score(y_test_encoded, predicted_classes)*100
        precision = precision_score(y_test_encoded, predicted_classes, average='weighted')*100
        recall = recall_score(y_test_encoded, predicted_classes, average='weighted')*100
        f1 = f1_score(y_test_encoded, predicted_classes, average='weighted')*100
        conf_matrix = confusion_matrix(y_test_encoded, predicted_classes)

        # Store metrics in the dictionary
        metrics_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist()  # Convert numpy array to list for JSON serialization
        }

        print(f"Model: {model_name} - Accuracy: {accuracy}")

        # Convert numeric predictions to descriptive sentences
        predicted_sentences = [label_descriptions.get(label, "Unknown movement") for label in predicted_labels]

        # Store the predictions for the current model as sentences
        predictions_output[model_name] = predicted_sentences

    all_predictions = {subject_identifier: predictions_output}

    # Initialize an array to hold the results
    prediction_results = []

    # Process the predictions to generate a structured result
    for subject_id, models_dict in all_predictions.items():
        subject_result = {'Subject Identifier': subject_id, 'Models': []}
        for model_name, predictions in models_dict.items():
            model_result = {'Model Name': model_name, 'Predictions': []}
            for i, pred in enumerate(predictions, 1):
                prediction_text = f"Trial {i}: {pred}"
                model_result['Predictions'].append(prediction_text)
            subject_result['Models'].append(model_result)
        prediction_results.append(subject_result)

    return prediction_results,metrics_results 

def predict_movement_cnn(features_df,subject_identifier):

    
    # Read, preprocess, and extract features just like before
    
    X_test = features_df.iloc[:, :-1]  # Features
    y_test = features_df.iloc[:, -1]  # Labels

    # Standardize the features using a scaler fitted on the training data
    scaler = StandardScaler().fit(X_test)
    X_test_scaled = scaler.transform(X_test)

    # Prepare a label encoder for converting string labels to integers, if necessary
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)

    model_directory_mat = 'saved_models_mat'

    # List of models to predict with
    model_files = [
        'cnn_model.h5'  # Keras model has a different file extension
    ]

    predictions_output = {}
    metrics_results={}

    # Iterate over each model file, load the model, and make predictions
    for model_file in model_files:
        model_path = os.path.join(model_directory_mat, model_file)
        model_name = model_file.replace('_model.joblib', '').replace('.h5', '')

        # Condition to check if the model is Keras model or not
        if model_file.endswith('.h5'):  # Keras model
            model = load_model(model_path)
            # Reshape the data for CNN
            X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
            predictions = model.predict(X_test_reshaped)
            predicted_classes = np.argmax(predictions, axis=1)
        else:  # Sklearn models
            model = load(model_path)
            predicted_classes = model.predict(X_test_scaled)

        # Convert numeric predictions back to original labels if necessary
        predicted_labels = label_encoder.inverse_transform(predicted_classes)

        accuracy = accuracy_score(y_test_encoded, predicted_classes)*100
        precision = precision_score(y_test_encoded, predicted_classes, average='weighted')*100
        recall = recall_score(y_test_encoded, predicted_classes, average='weighted')*100
        f1 = f1_score(y_test_encoded, predicted_classes, average='weighted')*100
        conf_matrix = confusion_matrix(y_test_encoded, predicted_classes)

        # Store metrics in the dictionary
        metrics_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist()  # Convert numpy array to list for JSON serialization
        }

        print(f"Model: {model_name} - Accuracy: {accuracy}")

        # Convert numeric predictions to descriptive sentences
        predicted_sentences = [label_descriptions.get(label, "Unknown movement") for label in predicted_labels]

        # Store the predictions for the current model as sentences
        predictions_output[model_name] = predicted_sentences

    all_predictions = {subject_identifier: predictions_output}

    # Initialize an array to hold the results
    prediction_results = []

    # Process the predictions to generate a structured result
    for subject_id, models_dict in all_predictions.items():
        subject_result = {'Subject Identifier': subject_id, 'Models': []}
        for model_name, predictions in models_dict.items():
            model_result = {'Model Name': model_name, 'Predictions': []}
            for i, pred in enumerate(predictions, 1):
                prediction_text = f"Trial {i}: {pred}"
                model_result['Predictions'].append(prediction_text)
            subject_result['Models'].append(model_result)
        prediction_results.append(subject_result)

    return prediction_results,metrics_results  

def main():

    global all_features
    global labels_list
    
    all_features = pd.DataFrame()  # Initialize an empty DataFrame to hold all features
    labels_list=[]
    

    all_true_labels = []
    all_predictions = []
    
    

    process_with_builtin_functions = False   #Toggling
    proces_with_builtin_accuracy= False
    csv_only= False
    edf_only=False
    csv_test=False
    all_predictions = defaultdict(lambda: defaultdict(list))

    for file_path in file_paths:
        print("Folder found:",file_paths)
        print(f"File found: {file_path}, with extension: {os.path.splitext(file_path)[1]}")

        # Determine the type of file and handle it accordingly
        if file_path.lower().endswith(('.csv', '.xls', '.xlsx', '.xlsm', '.xlsb','.txt')):
            
            csv_only= True
            csv_test=True
            raw_data, sfreq,cleanliness_messages = read_eeg_file(file_path)
            # csv_features(raw_data)
            #get_label_text()
            messages,csv_only=csv_identification(file_paths,processed_data_keywords)
           

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
                raw, sfreq, labels,data_info = read_mat_eeg(file_path)
                preprocessed_raw, preprocessing_steps = preprocess_raw_eeg(raw, 250, subject_identifier)
                features_message, features_df = extract_features_csp(preprocessed_raw, sfreq, labels)

                
                # Get predictions for the current file
                predictions = predict_movement_svc(features_df,subject_identifier)
                #print("Predictions test",predictions)
                
                # Append predictions to the corresponding subject and session
                # for model_name, model_predictions in predictions.items():
                #     all_predictions[subject_identifier][model_name].extend(model_predictions)
            
           
            else:
                raw_data, sfreq, labels,data_info= read_mat_eeg(file_path) # Handling .mat file 

                
                preprocessed_raw ,preprocessing_steps= preprocess_raw_eeg(raw_data, 250,subject_identifier)

                # for step in preprocessing_steps:
                #     print(step)


                # #features_df = extract_features_mat(preprocessed_raw, sfreq, labels,epoch_length=1.0)
                message,features_df = extract_features_csp(preprocessed_raw, sfreq, labels, epoch_length=1.0)

                # print(message)
                
                accuracy_test=mat_modeling_svc(subject_identifier,features_df,labels)
                # predictions_info = train_test_split_models(subject_identifier, features_df,model_name='svc' ,labels=labels)
                # print(predictions_info)
                # for info in predictions_info:
                #     print(info)
                #print(f"Test accuracy: {test_accuracy * 100:.2f}%")
                #print(label_sentence)
                # if subject_identifier not in subject_data:
                #         subject_data[subject_identifier] = {'features': pd.DataFrame(), 'labels': []}
                # # Append features and labels to the subject's data
                # subject_data[subject_identifier]['features'] = pd.concat([subject_data[subject_identifier]['features'], features_mat], ignore_index=True)
                # subject_data[subject_identifier]['labels'].extend(labels_mat)     
               
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
    
    


    # else:        
    # # This part should be outside (below) the for loop that goes through file_paths

    #     # Loop over your subject_data to perform cross-validation for each subject
    #     for subject_identifier, data in subject_data.items():
    #         X = data['features'].iloc[:, :-1]  # Features: all columns except the last
    #         y = LabelEncoder().fit_transform(data['features'].iloc[:, -1])  # Labels: the last column
    #         X_scaled = StandardScaler().fit_transform(X)  # Standardize features

    #        # Assuming X_scaled.shape is (n_samples, n_features)
    #         n_samples, n_features = X_scaled.shape

    #         # Check if n_features is less than the desired kernel_size
    #         if n_features < 3:
    #             # Code to handle the situation, e.g., reducing kernel_size or feature engineering
    #             # For example, reducing kernel_size:
    #             kernel_size = n_features  # Set kernel size equal to the number of features
    #         else:
    #             kernel_size = 3  # Or any other value greater than or equal to n_features

    #         # Reshape input data for the CNN
    #         X_reshaped = X_scaled.reshape((n_samples, n_features, 1))

    #         # Now define the input shape for the CNN model
    #         input_shape = (n_features, 1)  # Revised input shape
           
    #         cnn_classifier = KerasClassifier(build_fn=create_cnn_model, input_shape=input_shape, epochs=10, batch_size=10, verbose=0)

    #         # Define classifiers
    #         classifiers = {
    #             'RandomForest': RandomForestClassifier(),
    #             'SVC': SVC(),
    #             'Logistic Regression': LogisticRegression(),
    #             'KNN':KNeighborsClassifier(n_neighbors=5) ,
    #             'CNN': cnn_classifier
    #         }

    #         # Reshape the input data for CNN
    #         X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    #         # Perform cross-validation and collect accuracy for each classifier
    #         skf = StratifiedKFold(n_splits=5, shuffle=True)
    #         for clf_name, clf in classifiers.items():
    #             if clf_name=='CNN':
    #                  cv_scores = cross_val_score(clf, X_reshaped, y, cv=skf)
    #             else:
    #                 cv_scores = cross_val_score(clf, X_scaled, y, cv=skf)     
                
    #             # Get the base subject identifier without session (e.g., "sub-001")
    #             base_subject_identifier = subject_identifier.rsplit('_', 1)[0]
    #             subject_scores[base_subject_identifier][clf_name].extend(cv_scores)

    #     # Now calculate the overall accuracy for each subject by taking the mean of the aggregated scores
    #     overall_accuracy_dict = defaultdict(dict)
    #     for base_subject_identifier, clf_scores in subject_scores.items():
    #         for clf_name, scores in clf_scores.items():
    #             overall_accuracy = np.mean(scores)
    #             overall_accuracy_dict[base_subject_identifier][clf_name] = overall_accuracy
    #             print(f"Overall accuracy for {clf_name} on subject {base_subject_identifier}: {overall_accuracy * 100:.2f}%")

    #     # for subject_identifier, data in subject_data.items():
    #     #     label_encoder = LabelEncoder()
            
    #     #     X = data['features'].iloc[:, :-1]  # features: all columns except the last
    #     #     y_binned = label_encoder.fit_transform(data['features'].iloc[:, -1])  # labels: the last column

    #     #     # Perform a train-test split
    #     #     X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2)

    #     #     # Standardize the features
    #     #     scaler = StandardScaler()
    #     #     X_train = scaler.fit_transform(X_train)
    #     #     X_test = scaler.transform(X_test)

    #     #     # Fit the model (you can choose SVC, RandomForestClassifier, or GradientBoostingClassifier as shown in your code)
    #     #     clf_rnd = RandomForestClassifier()
    #     #     clf_rnd.fit(X_train, y_train)

    #     #     # Predict on the test data
    #     #     y_pred_rnd = clf_rnd.predict(X_test)

    #     #     # Calculate the accuracy
    #     #     acc_rnd = accuracy_score(y_test, y_pred_rnd)
    #     #     print(f"Accuracy RandomForest for subject {subject_identifier}: {acc_rnd * 100:.2f}%")
    #     #     accuracy_dict_rnd[subject_identifier].append(acc_rnd)

    #     #     # Fit the model (you can choose SVC, RandomForestClassifier, or GradientBoostingClassifier as shown in your code)
    #     #     clf_SVC = SVC()
    #     #     clf_SVC.fit(X_train, y_train)

    #     #     # Predict on the test data
    #     #     y_pred_SVC = clf_SVC.predict(X_test)

    #     #     # Calculate the accuracy
    #     #     acc_svc = accuracy_score(y_test, y_pred_SVC)
    #     #     print(f"Accuracy SVC for subject {subject_identifier}: {acc_svc * 100:.2f}%")  
    #     #     accuracy_dict_svc[subject_identifier].append(acc_svc)

    #     #     clf_gbc = GradientBoostingClassifier()
    #     #     clf_gbc.fit(X_train, y_train)

    #     #     # Predict on the test data
    #     #     y_pred_gbc = clf_gbc.predict(X_test)

    #     #     # Calculate the accuracy
    #     #     acc_gbc = accuracy_score(y_test, y_pred_gbc)
    #     #     print(f"Accuracy Gradient for subject {subject_identifier}: {acc_gbc * 100:.2f}%")
    #     #     accuracy_dict_gbc[subject_identifier].append(acc_gbc)

    #     #     clf_knn = MLPClassifier(hidden_layer_sizes=(40,), max_iter=500, alpha=0.001,
    #     #             solver='sgd', tol=0.000000001)
    #     #     clf_knn.fit(X_train, y_train)

    #     #     # Predict on the test data
    #     #     y_pred_knn = clf_knn.predict(X_test)

    #     #     # Calculate the accuracy
    #     #     acc_knn = accuracy_score(y_test, y_pred_knn)
    #     #     print(f"Accuracy Gradient for subject {subject_identifier}: {acc_knn * 100:.2f}%")
    #     #     accuracy_dict_knn[subject_identifier].append(acc_knn)  
        
    #     # for subject_session in accuracy_dict_rnd:
            
    #     #     # Extract the subject identifier without the session
    #     #     subject_identifier = subject_session.rsplit('_', 1)[0]
    #     #     # Sum the accuracies for RandomForest
    #     #     total_accuracy_dict_rnd[subject_identifier] = total_accuracy_dict_rnd.get(subject_identifier, 0) + sum(accuracy_dict_rnd[subject_session])
    #     #     # Sum the accuracies for SVC
    #     #     total_accuracy_dict_svc[subject_identifier] = total_accuracy_dict_svc.get(subject_identifier, 0) + sum(accuracy_dict_svc[subject_session])
    #     #     # Sum the accuracies for GradientBoosting
    #     #     total_accuracy_dict_gbc[subject_identifier] = total_accuracy_dict_gbc.get(subject_identifier, 0) + sum(accuracy_dict_gbc[subject_session])

    #     #     total_accuracy_dict_knn[subject_identifier] = total_accuracy_dict_knn.get(subject_identifier, 0) + sum(accuracy_dict_knn[subject_session])

    #     #     # Count the number of sessions for each subject
    #     #     subject_session_counts[subject_identifier] = subject_session_counts.get(subject_identifier, 0) + len(accuracy_dict_rnd[subject_session])

    #     # # Now calculate the average accuracy for each classifier for each subject
    #     # average_accuracy_dict_rnd = {subject: total_accuracy / subject_session_counts[subject] for subject, total_accuracy in total_accuracy_dict_rnd.items()}
    #     # average_accuracy_dict_svc = {subject: total_accuracy / subject_session_counts[subject] for subject, total_accuracy in total_accuracy_dict_svc.items()}
    #     # average_accuracy_dict_gbc = {subject: total_accuracy / subject_session_counts[subject] for subject, total_accuracy in total_accuracy_dict_gbc.items()}
    #     # average_accuracy_dict_knn = {subject: total_accuracy / subject_session_counts[subject] for subject, total_accuracy in total_accuracy_dict_knn.items()}


    #     # # Print the average accuracy for each subject
    #     # for subject in average_accuracy_dict_rnd.keys():
    #     #     print(f"Subject {subject} Accuracy RandomForest: {average_accuracy_dict_rnd[subject] * 100:.2f}%")
    #     #     print(f"Subject {subject} Accuracy SVC: {average_accuracy_dict_svc[subject] * 100:.2f}%")
    #     #     print(f"Subject {subject} Accuracy GradientBoosting: {average_accuracy_dict_gbc[subject] * 100:.2f}%")
    #     #     print(f"Subject {subject} Accuracy Knn: {average_accuracy_dict_knn[subject] * 100:.2f}%")


    # # if process_with_builtin_functions and :
    # #     print("Length of all_features: ", len(all_predictions))
    # #     print("Length of labels_list: ", len(all_true_labels))

    # #     overall_accuracy = accuracy_score(all_true_labels, all_predictions)
    # #     print(f'All model accuracy across all files: {overall_accuracy * 100:.2f}%')

    # else:
    if csv_only | edf_only:        
       #csv_modeling()
        csv_logistic_model(label_conditions_path)
        #label_conditions = read_label_conditions(label_conditions_path)
        # predictions_on_test = predict_on_training_data('Model_CNN',label_conditions)
        # for message in predictions_on_test:
        #     print("Message:", message)
    # else:
    #     label_conditions = read_label_conditions(conditions_file)
    #     conditions = load_and_predict_random(all_features, label_conditions)
    #     # for condition in conditions:
    #     #     print(condition)   

   


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
