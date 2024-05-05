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
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

def use_csp(eegData):
    #write code to take raw eeg data from the loadData function and use csp on it and then extract features and filter using subbands
    pass

def features(eegData, labels):
    FS = []
    headers = []
    samples = eegData[:,0,0]
    channels = eegData[0,:,0]
    for sample in range(samples.size):
        L=[]
        si=eegData[sample,:,:]
        for channel in range(channels.size):
            channeldata = si[channel,:]

            #Time Domain Features
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
            '''
            #Frequency Domain Features
            fft_val = np.fft.fft(channeldata)
            psd_val = np.abs(fft_val)**2 / len(channeldata)
            fdjt_val = np.fft.fft(channeldata)
            fdjt_val = np.abs(fdjt_val) / len(channeldata)

            L.extend([fft_val, psd_val, fdjt_val])

            if sample == 0:
                headers.extend([f'Channel_{channel+1}_fft', f'Channel_{channel+1}_psd',
                                f'Channel_{channel+1}_fdjt'])
            '''    
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

subbands = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 100)]

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

def extract_features(file_paths):
    allf = []
    feat = []
    for path in file_paths:
        headers = []
        eegData, labels = loadData(path)
        feat, headers = features(eegData, labels)
        allf.append(pd.DataFrame(feat))
    for band_idx, (lowcut, highcut) in enumerate(subbands, start=1):
        subfeatures = []
        for path in file_paths:
            eegData, labels = loadData(path)
            filtered_data = filtering(eegData, fs=500, lowcut=lowcut, highcut=highcut)
            subband_features, subband_headers = features(filtered_data, labels)
            subfeatures.append(pd.DataFrame(subband_features))
        csv_path = f'subband_{band_idx}_features.csv'
        allfeatures(subfeatures, headers, csv_path)

    allfeatures(allf, headers, 'allfeatures.csv')
    choosebest()
    return csv_path

def choosebest():
    best = -99999999
    for path in csvpaths:
        accuracy = classification(path)
        if accuracy > best:
            best = accuracy
            bestpath = path
    print(best)
    print(bestpath)
    return bestpath

def classification(csvpath):
    df = pd.read_csv(csvpath)

    X = df.drop(columns=['Label']) 
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_classifier = SVC(kernel='linear') 
    svm_classifier.fit(X_train, y_train)

    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of " , csvpath, " :", accuracy)
    return accuracy