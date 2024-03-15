from glob import glob
import os    #reading file paths or directories.
import mne   # For working with EEG data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mne.io import RawArray
from scipy.stats import entropy
from scipy.fft import rfft
from pyprep.find_noisy_channels import NoisyChannels
from mne.preprocessing import ICA



downloads_folder = os.path.expanduser("~/Downloads")  # Get the path to the "Downloads" directory
dataverse_files_folder = os.path.join(downloads_folder, "seizure")  # Combine with the folder name

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
            return None, None,None
        
        print(f"Number of samples (time points) in the recording: {len(df)}")
        labels = df.iloc[:, -1].copy()  # Assume the last column contains labels
        df = df.iloc[:, :-1] 


        if check_processed_from_columns(df, processed_data_keywords):
            print("The data appears to be processed.")
            return df,labels, None  # Return the DataFrame and None for sfreq

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

        return df, labels, sfreq

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None, None,None


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
    ica = ICA(n_components=10, random_state=97, max_iter=800)
    ica.fit(raw)
    # Apply ICA to the raw data to remove the bad components
    raw = ica.apply(raw)

    return raw 

def extract_features(raw, epoch_length=1.0):
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

def create_epochs_from_preprocessed_features(df, epoch_length=1.0, sfreq=None):
    if sfreq is None:
        raise ValueError("Sampling frequency must be provided if not inherent in the DataFrame index.")

    # Calculate the number of samples per epoch
    samples_per_epoch = int(sfreq * epoch_length)
    print(f'Samples per epoch: {samples_per_epoch}')

    # Calculate the number of complete epochs that can be formed
    num_complete_epochs = len(df) // samples_per_epoch
    print(f'Number of complete epochs: {num_complete_epochs}')

    if num_complete_epochs == 0:
        print("No complete epochs could be formed.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Extract only the part of the DataFrame that can be evenly divided into epochs
    complete_epoch_data = df.iloc[0:num_complete_epochs * samples_per_epoch]
    print(f'Length of complete epoch data: {len(complete_epoch_data)}')

    # Reshape the data to have an index for epochs and a second index for samples within those epochs
    reshaped_data = complete_epoch_data.values.reshape(num_complete_epochs, samples_per_epoch, -1)
    print(f'Reshaped data shape: {reshaped_data.shape}')

    # Initialize a list to hold features from all epochs
    all_features = []

    # Iterate over the epochs to calculate mean features
    for epoch_data in reshaped_data:
        # Calculate features for the current epoch if needed or just collect the existing features
        epoch_features = epoch_data.mean(axis=0).tolist()  # Example feature calculation

        # Append the features of the current epoch to the list
        all_features.append(epoch_features)

    # Create a DataFrame from the features list
    feature_df = pd.DataFrame(all_features, columns=df.columns)

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
    features_df = extract_features(raw)  # Should now be a DataFrame
    return features_df



all_features = pd.DataFrame()  # Initialize an empty DataFrame to hold all features
labels_list=[]




# Iterate through each file path in the list of file paths
for file_path in file_paths:
    if file_path.endswith('.csv',):
        
        # Read the CSV file and determine if it's processed
        df_or_raw, labels, sfreq = read_eeg_file(file_path)
        
        if isinstance(df_or_raw, pd.DataFrame):
            print("The data appears to be processed and features are already extracted.")
            identified_methods = identify_feature_extraction_methods(df_or_raw, processed_data_keywords)
            if identified_methods:
                print(f"The following feature extraction methods were identified: {', '.join(identified_methods)}")
                
                # Identify calculation methods for each band of interest
                for band in ['theta', 'alpha', 'beta', 'gamma']:
                    band_columns = [col for col in df_or_raw.columns if band in col.lower()]
                    for col in band_columns:
                        method = identify_calculation_method(col, processed_data_keywords)
                        print(f"The {band} band feature '{col}' is calculated using: {method}")
            
                epochs_df = create_epochs_from_preprocessed_features(df_or_raw, epoch_length=1.0, sfreq=500)
                print("Epochs DataFrame created with preprocessed features.")
                print(epochs_df.head()) 
                all_features = pd.concat([all_features, epochs_df], ignore_index=True)
                print("All features shape:", all_features.shape)
                     
            else:
                print("No known feature extraction methods were identified.")
        elif df_or_raw is not None:
            # Handle the raw data
            print("The data appears to be raw.")
            print(f"CSV file {file_path} read successfully with sampling frequency {sfreq} Hz.")
            first_five_channels = df_or_raw.ch_names[:10]  # Get the names of the first five channels
            plot_raw_eeg(df_or_raw, title=f'Raw EEG Data: {file_path}', picks=first_five_channels, separate_channels=True)
            raw_preprocessed = preprocess_raw_eeg(df_or_raw, sfreq)
            plot_raw_eeg(raw_preprocessed, title=f'Preprocessed EEG Data: {file_path}', picks=first_five_channels, separate_channels=True)
            if raw_preprocessed is not None:
                fft_features = extract_features(raw_preprocessed)
                print(fft_features)
                features_df = extract_features_as_df(raw_preprocessed)
                all_features = pd.concat([all_features, features_df], ignore_index=True)
                labels_list.append(labels)
                print("Labels list:", labels_list)
                print("All features shape:", all_features.shape)


    elif file_path.lower().endswith('.edf'):
        raw, sfreq = read_edf_eeg(file_path)
        
        if raw is not None:
            print(f"EDF file {file_path} read successfully with sampling frequency {sfreq} Hz.")
            raw_preprocessed = preprocess_raw_eeg(raw, sfreq)
            first_five_channels = raw.ch_names[:10]  # Get the names of the first five channels
           # plot_raw_eeg(raw_preprocessed, title=f'Preprocessed EEG Data: {file_path}', picks=first_five_channels, separate_channels=True)
            if raw_preprocessed is not None:
                fft_features = extract_features(raw_preprocessed)
                print(fft_features)
                features_df = extract_features_as_df(raw_preprocessed)
                all_features = pd.concat([all_features, features_df], ignore_index=True)
                print("All features shape:", all_features.shape)

    elif file_path.lower().endswith(('.xls', '.xlsx', '.xlsm', '.xlsb')):
        df_or_raw, sfreq = read_eeg_file(file_path)
        if isinstance(df_or_raw, pd.DataFrame): 
            print("The data appears to be processed and features are already extracted.")
            identified_methods = identify_feature_extraction_methods(df_or_raw, processed_data_keywords)
            if identified_methods:
                print(f"The following feature extraction methods were identified: {', '.join(identified_methods)}")
                
                # Identify calculation methods for each band of interest
                for band in ['theta', 'alpha', 'beta', 'gamma']:
                    band_columns = [col for col in df_or_raw.columns if band in col.lower()]
                    for col in band_columns:
                        method = identify_calculation_method(col, processed_data_keywords)
                        print(f"The {band} band feature '{col}' is calculated using: {method}")
            
                epochs_df = create_epochs_from_preprocessed_features(df_or_raw, epoch_length=1.0, sfreq=500)
                print("Epochs DataFrame created with preprocessed features.")
                print(epochs_df.head()) 
                all_features = pd.concat([all_features, epochs_df], ignore_index=True)
                print("All features shape:", all_features.shape)
                     
            else:
                print("No known feature extraction methods were identified.")
        elif df_or_raw is not None:
            # Handle the raw data
            print("The data appears to be raw.")
            print(f"CSV file {file_path} read successfully with sampling frequency {sfreq} Hz.")
            first_five_channels = df_or_raw.ch_names[:10]  # Get the names of the first five channels
            plot_raw_eeg(df_or_raw, title=f'Raw EEG Data: {file_path}', picks=first_five_channels, separate_channels=True)
            raw_preprocessed = preprocess_raw_eeg(df_or_raw, sfreq)
            plot_raw_eeg(raw_preprocessed, title=f'Preprocessed EEG Data: {file_path}', picks=first_five_channels, separate_channels=True)
            if raw_preprocessed is not None:
                fft_features = extract_features(raw_preprocessed)
                print(fft_features)
                features_df = extract_features_as_df(raw_preprocessed)
                all_features = pd.concat([all_features, features_df], ignore_index=True)
                print("All features shape:", all_features.shape)           
   
    else:
        print(f"File {file_path} is not a recognized EEG file type.")





# label_encoder = LabelEncoder()
# all_labels = pd.concat(labels_list, ignore_index=True)


# X = all_features.iloc[:, :-1]  # features: all columns except the last
# y_binned = label_encoder.fit_transform(all_labels)     # labels: the last column
# print("Y binned:", y_binned)
# print("Last column: ",all_features.iloc[:, -1])


# threshold = np.percentile(y_binned, 50)  # This is essentially the same as the median
# y = pd.cut(y_binned, bins=[-np.inf, threshold, np.inf], labels=[0, 1])

# print("Y head:",y)  # To see the first few entries
# print("Y unique:",y.unique())  # To see the unique values

# # Perform a train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test= scaler.transform(X_test)

# clf = SVC()
# clf.fit(X_train, y_train)
# y_pred_svc = clf.predict(X_test)
# acc_svc = round(clf.score(X_train, y_train) * 100, 2)
# print("SVM Accuracy is:",(str(acc_svc)+'%'))

# # Initialize the classifier
# clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# # Fit the classifier to the training data
# clf_rf.fit(X_train, y_train)

# # Predict on the test data
# y_pred_rf = clf_rf.predict(X_test)

# # Calculate the accuracy
# acc_rf = round(clf_rf.score(X_train, y_train) * 100, 2)
# print("Random Forest accuracy is:", (str(acc_rf) + '%'))

# clf_gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)

# # Fit the classifier to the training data
# clf_gbc.fit(X_train, y_train)

# # Predict on the test data
# y_pred_gbc = clf_gbc.predict(X_test)

# # Calculate the accuracy on the training set
# acc_gbc = round(clf_gbc.score(X_train, y_train) * 100, 2)
# print("Gradient Boosting Classifier accuracy is:", (str(acc_gbc) + '%'))



