import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import tkinter as tk
from tkinter import filedialog 
from tkinter import messagebox as msg 
import pyedflib
import mne
import scipy
from scipy.fft import fft
from scipy import stats
import os
import threading
import queue
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from statsmodels.tsa.ar_model import AutoReg
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def loadData(fileName):

    if fileName.endswith('mat'):
        try:
            mat = scipy.io.loadmat(fileName)
            eegData = mat['data']
            label = mat['labels']
            return eegData, label
        except ImportError:
            print('Library Error', 'scipy is required to load MAT data.')
            return None
        except Exception as e:
            print('Error', f'Error in loading mat file: {str(e)}')
            return None
        
    else:
        print('Wrong file type')

def use_csp(eegData, labels):
    print('csp started')
    n_channels, n_samples, n_trials = eegData.shape
    eegData = eegData.transpose(1, 0, 2).reshape(n_samples, n_channels * n_trials)

    ch_names = [f'ch{i+1}' for i in range(n_channels)] 
    info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')
    raw = mne.io.RawArray(eegData, info)
    
    raw.notch_filter(60, picks='eeg')  # Remove powerline noise

    events = np.array([[i, 0, label] for i, label in enumerate(labels)])
    epochs = mne.Epochs(raw, events, tmin=0, tmax=raw.times[-1], baseline=None, detrend=1, preload=True)
    epochs = epochs.pick_types(eeg=True)
    epochs_train = epochs.copy().crop(tmin=0, tmax=0.5)
    csp = mne.decoding.CSP(n_components=4)
    csp.fit(epochs_train)
    filtereddata = csp.transform(epochs.get_data())
    print('ended')
    return filtereddata

def stat(channeldata, L, channel, headers, sample):
    min_val = np.min(channeldata)
    max_val = np.max(channeldata)
    mean_val = np.mean(channeldata)
    std_val = np.std(channeldata)
    rms_val = np.sqrt(np.mean(channeldata**2))
    var_val = np.var(channeldata)
    power_val = np.mean(channeldata**2)
    peak_val = np.max(np.abs(channeldata))
    p2p_val = np.ptp(channeldata)
    crest_factor_val = peak_val / rms_val
    skew_val = stats.skew(channeldata)
    kurtosis_val = stats.kurtosis(channeldata)

    L.extend([min_val, max_val, mean_val, std_val, rms_val, var_val,
                power_val, peak_val, p2p_val, crest_factor_val, skew_val,
                kurtosis_val])
    if sample == 0:
        headers.extend([f'Channel_{channel+1}_min', f'Channel_{channel+1}_max',
                    f'Channel_{channel+1}_mean', f'Channel_{channel+1}_std',
                    f'Channel_{channel+1}_rms', f'Channel_{channel+1}_var',
                    f'Channel_{channel+1}_power', f'Channel_{channel+1}_peak',
                    f'Channel_{channel+1}_p2p', f'Channel_{channel+1}_crestfactor',
                    f'Channel_{channel+1}_skew', f'Channel_{channel+1}_kurtosis'])
    return L, headers

def ar(channeldata, L, channel, headers, sample):
    order = 3
    ar_model = AutoReg(channeldata, lags=order)
    ar_model_fit = ar_model.fit()
    ar_coeffs = ar_model_fit.params[1:]

    L.extend(ar_coeffs.tolist())
    if sample == 0:
        for lag in range(1, 4):
            headers.append(f'Channel_{channel+1}_ar{lag}')

    return L, headers

def fft_features(channeldata, L, channel, headers, sample):
    fft_val = fft(channeldata)

    mean_fft = np.mean(np.abs(fft_val))

    L.extend([mean_fft])
    if sample == 0:
        headers.extend([f'Channel_{channel+1}_fft']) 

    return L, headers

def psd(channeldata, L, channel, headers, sample):
    fft_val = fft(channeldata)
    psd_val = np.abs(fft_val)**2 / len(channeldata)

    mean_psd = np.mean(np.abs(psd_val))

    L.extend([mean_psd])
    if sample == 0:
        headers.extend([f'Channel_{channel+1}_psd']) 

    return L, headers

def fdjt(channeldata, L, channel, headers, sample):
    fdjt_val = fft(channeldata)
    fdjt_val = np.abs(fdjt_val) / len(channeldata)

    mean_fdjt = np.mean(np.abs(fdjt_val))

    L.extend([mean_fdjt])
    if sample == 0:
        headers.extend([f'Channel_{channel+1}_fdjt']) 

    return L, headers

def features(eegData, labels):
    method = 'fft'
    FS = []
    headers = []
    samples = eegData[:,0,0]
    channels = eegData[0,:,0]
    for sample in range(samples.size):
        L=[]
        si=eegData[sample,:,:]
        for channel in range(channels.size):
            channeldata = si[channel,:]

            if method == 'stat':
                L, headers = stat(channeldata, L, channel, headers, sample)
            elif method == 'ar':
                L, headers = ar(channeldata, L, channel, headers, sample)
            elif method == 'fft':
                L, headers = fft_features(channeldata, L, channel, headers, sample)
            elif method == 'psd':
                L, headers = psd(channeldata, L, channel, headers, sample)
            elif method == 'fdjt':
                L, headers = fdjt(channeldata, L, channel, headers, sample)

        L.append(labels[0,sample])
        FS.append(L)
    headers.append('Label')
    return FS, headers

csvpaths = []
def allfeatures(FS, headers, csv_path):
    dffeatures = pd.concat(FS, ignore_index=True)
    dffeatures.columns = headers
    current_dir = os.path.dirname(os.path.realpath(__file__))
    csvpath = os.path.join(current_dir, csv_path)
    dffeatures.to_csv(csvpath, index=False)
    print(csvpath, " done")
    csvpaths.append(csvpath)
    return csvpath

subbands = [(0, 100), (0.5, 4), (4, 8), (8, 12), (12, 30), (30, 100)]
best = [0,0,0,0,0,0]
subfeatures = [[] for _ in range(6)]
subheaders = [[] for _ in range(6)]

def filtering(data, fs, lowcut, highcut, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    
    samples, channels, trials = data.shape
    data_2d = np.reshape(data, (samples * channels, trials))

    filtered_data_2d = sosfiltfilt(sos, data_2d, axis=-1)

    filtered_data = np.reshape(filtered_data_2d, (samples, channels, trials))
    
    return filtered_data

def extract_freq(band_idx, lowcut, highcut, eegData, labels):
    print('freq', band_idx, 'start')
    if band_idx == 0:
        filtered_data = eegData
    else:
        filtered_data = filtering(eegData, fs=250, lowcut=lowcut, highcut=highcut)
    subband_features, subband_headers = features(filtered_data, labels)
    subfeatures[band_idx].append(pd.DataFrame(subband_features))
    subheaders[band_idx] = subband_headers

def extract_features(file_paths):
    threads = []
    
    for path in file_paths:
        eegData, labels = loadData(path)
        for band_idx, (lowcut, highcut) in enumerate(subbands, start=0):
            name = 'freqthread_', band_idx, path
            name = threading.Thread(target=extract_freq, args=(band_idx, lowcut, highcut, eegData, labels))
            name.start()
            threads.append(name)
        for thread in threads:
            thread.join()

    for band_idx in range(len(subbands)):
        csv_path = f'subband_{band_idx}_features.csv'
        allfeatures(subfeatures[band_idx], subheaders[band_idx], csv_path)

    for index in range(0,6):
        name = 'thread_', index
        name = threading.Thread(target=classificationrf, args=(index,))
        name.start()
        threads.append(name)

    for thread in threads:
        thread.join()

    max = best[0]
    bestpath = csvpaths[0]
    for i in range(1,6):
        if best[i]>max:
            max = best[i]
            bestpath = csvpaths[i]
    print(max, " for ", bestpath)
    return bestpath

def classificationsvm(index):
    print('start', index)
    df = pd.read_csv(csvpaths[index])

    X = df.drop(columns=['Label']) 
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_classifier = SVC(kernel='linear') 
    svm_classifier.fit(X_train, y_train)

    y_pred = svm_classifier.predict(X_test)
    best[index] = accuracy_score(y_test, y_pred)
    print("Accuracy of " , csvpaths[index], " :", best[index])

def classificationrf(index):
    print('start', index)
    df = pd.read_csv(csvpaths[index])

    X = df.drop(columns=['Label']) 
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)
    best[index] = accuracy_score(y_test, y_pred)
    print("Accuracy of ", csvpaths[index], " :", best[index])