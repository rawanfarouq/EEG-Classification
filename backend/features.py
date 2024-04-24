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
#from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC
#from sklearn.metrics import accuracy_score

def files(fileName, isTime):
    subid = fileName.split('_')[0]
    subjectfiles = glob.glob(subid + '*')
    subjectdata = []
    for file in subjectfiles:
        df, labels = loadData(file)
        subjectdata.append(df)

    #filtereddata = [noiseCancellation(df) for df in subjectdata]
    extract(subjectdata, isTime, labels)


def loadData(fileName):

    if fileName.endswith('edf') or fileName.endswith('edf+'):
        try:
            f = pyedflib.EdfReader(fileName)
            numsignals = f.signals_in_file
            signalslabels = f.getSignalLabels()
            samples = f.getNSamples()[0]
            data = np.zeros((samples, numsignals))
            for index in range(numsignals):
                data[:,index] = f.readSignal(index)
            #fs = f.getSampleFrequency(0)
            df = pd.DataFrame(data, columns=signalslabels)
            df.index = range(samples)
            #noiseCancellation(df)
            return df
        except ImportError:
            print('Library Error', 'pyedflib library is required to load EDF/EDF+ data.')
            return None
        except Exception as e:
            print('Error', f'Error in loading EDF/EDF+ data: {str(e)}')
            return None
        
    elif fileName.endswith('mat'):
        try:
            mat = scipy.io.loadmat(fileName)
            eegData = mat['data']

            if len(eegData.shape) == 3:
                trials, channels, samples = eegData.shape
                eegData = np.concatenate(eegData, axis=1)
                eegData = eegData.reshape(channels, trials * samples)

            channelnames = ['EEG' + str(i)  for i in range(channels)]
            channeltypes = ['eeg'] * channels

            info = mne.create_info(ch_names=channelnames,sfreq=500,ch_types=channeltypes)
            raw = mne.io.RawArray(eegData, info=info)
            labels = mat['labels']
            df = pd.DataFrame(eegData.T)
            #noiseCancellation(df)
            return df, labels
        except ImportError:
            print('Library Error', 'scipy is required to load MAT data.')
            return None
        except Exception as e:
            print('Error', f'Error in loading mat file: {str(e)}')
            return None
    else:
        print('Wrong file type')
            
    '''
    elif fileName.endswith('vhdr'):
        try:
            raw = mne.io.read_raw_brainvision(fileName, preload=True)
            data = raw.get_data().T
            channels = raw.ch_names
            times = raw.times
            df = pd.DataFrame(data, columns=channels)
            df['Time'] = times
            return df
        except ImportError:
            msg.showerror('Library Error', 'mne is required to load BV/BVR data.')
            return None
        except Exception as e:
            msg.showerror('Error', f'Error in loading vhdr file: {str(e)}')
            return None
            
    elif fileName.endswith('cnt'):
        try:
            file=mne.io.read_raw_cnt(fileName)
            df=file.to_data_frame()
            return df
        except ImportError:
            msg.showerror('Library Error', 'mne is required to load CNT data.')
            return None
        except Exception as e:
            msg.showerror('Error', f'Error in loading cnt file: {str(e)}')
            return None
            
    elif fileName.endswith('gdf'):
        try:
            file=mne.io.read_raw_gdf(fileName)
            df=file.to_data_frame()
            return df
        except ImportError:
            msg.showerror('Library Error', 'mne is required to load GDF data.')
            return None
        except Exception as e:
            msg.showerror('Error', f'Error in loading gdf file: {str(e)}')
            return None
    '''
def noiseCancellation(df):
    filtered_df = df.copy()
    '''
    if not df.empty:
        window_size = min(self.window_size, len(df))
        #samplesnum = len(df)
        #duration = samplesnum / self.fs
            
        for column in filtered_df.columns:
            filtered_df[column] = df[column].rolling(window=window_size, min_periods=1).mean()
            #filtered_df[column] = df[column].rolling(window=window_size, min_periods=duration).mean()
    '''
    #threading.Thread(target=plotChannels, args=(filtered_df,)).start()
    #channelextraction(filtered_df)
    return filtered_df
    pass

'''
def plotChannels(df):
    q = queue.Queue()

    def plotInMainThread():
        for channel in df.columns:
            fig, ax = plt.subplots()
            ax.plot(df.index, df[channel])
            ax.set_title(f'Channel: {channel}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.grid(True)
            canvas = FigureCanvasTkAgg(fig, master=root) 
            canvas.draw()
            q.put(canvas.get_tk_widget())

    plt_thread = threading.Thread(target=plotInMainThread)
    plt_thread.start()

    canvas_widget = q.get()
    canvas_widget.pack()

def plot(self, dfs, filtereds):
    #Same Figure
    #plt.figure(figsize=(10,6))
    #for i, df in enumerate(dfs, start=1):
        #col = df.columns[0] if len(df.columns) > 0 else 'Data'
        #print(f"DataFrame for Session {i}:")
        #print(df.head())
        #plt.plot(df.index, df[col], label=f'Session {i}')
    #plt.title('Subject EEG Data')
    #plt.xlabel('Time')
    #plt.ylabel('Amplitude')
    #plt.legend()
    #for df in dfs: df.plot()
        
    #Same Window Different Figures
    num_sessions = len(dfs)
    fig, axes = plt.subplots(num_sessions, 2, figsize=(15, 6*num_sessions))
    plt.subplots_adjust(hspace=0.75)
    for i, (df,filtered,ax) in enumerate(zip(dfs,filtereds,axes), start=1):
        col = df.columns[0] if len(df.columns) > 0 else 'Data'
        filteredcol = filtered.columns[0] if len(filtered.columns) > 0 else 'Data'
        ax[0].plot(df.index, df[col], color='black')
        ax[0].set_title(f'Unfiltered Session {i}')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Amplitude')
        ax[1].plot(filtered.index, filtered[filteredcol], color='blue')
        ax[1].set_title(f'Filtered Session {i}')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Amplitude')
        #print(f"Unfiltered DataFrame for Session {i}:")
        #print(df.head())
        #print(f"Filtered DataFrame for Session {i}:")
        #print(filtered.head())
    pass
'''
    
def extract(dfs, isTime, labels):
    allfeatures = []
    
    for df in dfs:
        if(isTime):
            channelfeatures, featuresheader = channelextractiontime(df, labels)
        else:
            channelfeatures, featuresheader = channelextractionfreq(df, labels)
        #channelfeatures, featuresheader = channelextraction(df)
        allfeatures.append(pd.DataFrame(data=channelfeatures, columns=featuresheader))

    dffeatures = pd.concat(allfeatures, ignore_index=False) 
    current_dir = os.path.dirname(os.path.realpath(__file__))
    csvpath = os.path.join(current_dir, 'features.csv')
    dffeatures.to_csv(csvpath, index=False)
    return csvpath
            
    #allv = np.concatenate([df.values.ravel() for df in dfs])
    #mean = np.mean(allv)
    #rms = np.sqrt(np.mean(allv**2))
    #variance = np.var(allv)
    #std = np.std(allv)
    #zeroCrossings = np.where(np.diff(np.sign(allv)))[0]
    #zcr = len(zeroCrossings) / (len(allv) - 1)

    #print("Mean Amplitude:", mean)
    #print("Root Mean Square:", rms)
    #print("Variance:", variance)
    #print("Zero-Crossing Rate:", zcr)
    #print("Standard Deviation:", std)

def channelextraction(df, labels):
    features = []

    featuresheader = ['Channel', 'Minimum', 'Maximum', 'Mean', 'RMS', 'Variance', 'Standard Deviation',
                        'Power', 'Peak', 'Peak-to-Peak', 'Crest Factor', 'Skew', 'Kurtosis',
                        'Max_F', 'Sum_F', 'Mean_F', 'Variance_F', 'Peak_F', 'Skew_F', 'Kurtosis_F', 'Label']
        
    for channel in df.columns: 
        channeldata = df[channel]
            
        #Time Domain
        min_val = np.min(channeldata)
        max_val = np.max(channeldata)
        mean_val = np.mean(channeldata)
        rms_val = np.sqrt(np.mean(channeldata**2))
        var_val = np.var(channeldata)
        std_val = np.std(channeldata)
        power_val = np.mean(channeldata**2)
        peak_val = np.max(np.abs(channeldata))
        p2p_val = np.ptp(channeldata)
        crest_factor_val = peak_val / rms_val
        skew_val = stats.skew(channeldata)
        kurtosis_val = stats.kurtosis(channeldata)
            
        #Frequency Domain
        ft = fft(channeldata.values)
        S = np.abs(ft)**2 / len(channeldata)
        maxf_val = np.max(S)
        sumf_val = np.sum(S)
        meanf_val = np.mean(S)
        varf_val = np.var(S)
        peakf_val = np.max(np.abs(S))
        skewf_val = stats.skew(S)
        kurtosisf_val = stats.kurtosis(S)
            
        features.append([channel, min_val, max_val, mean_val, rms_val, var_val, std_val, power_val, 
                        peak_val, p2p_val, crest_factor_val, skew_val, kurtosis_val, maxf_val, 
                        sumf_val, meanf_val, varf_val, peakf_val, skewf_val, kurtosisf_val, labels])
            
    #dffeatures = pd.DataFrame(data=features, columns=featuresheader)
    #current_dir = os.path.dirname(os.path.realpath(__file__))
    #csvpath = os.path.join(current_dir, 'features.csv')
    #dffeatures.to_csv(csvpath, index=False)
    #return csvpath
    return features, featuresheader

def channelextractiontime(df, labels):
    features = []

    featuresheader = ['Channel', 'Minimum', 'Maximum', 'Mean', 'RMS', 'Variance', 'Standard Deviation',
                        'Power', 'Peak', 'Peak-to-Peak', 'Crest Factor', 'Skew', 'Kurtosis', 'Label']
        
    for channel in df.columns: 
        channeldata = df[channel]
            
        #Time Domain
        min_val = np.min(channeldata)
        max_val = np.max(channeldata)
        mean_val = np.mean(channeldata)
        rms_val = np.sqrt(np.mean(channeldata**2))
        var_val = np.var(channeldata)
        std_val = np.std(channeldata)
        power_val = np.mean(channeldata**2)
        peak_val = np.max(np.abs(channeldata))
        p2p_val = np.ptp(channeldata)
        crest_factor_val = peak_val / rms_val
        skew_val = stats.skew(channeldata)
        kurtosis_val = stats.kurtosis(channeldata)
            
        features.append([channel, min_val, max_val, mean_val, rms_val, var_val, std_val, power_val, 
                        peak_val, p2p_val, crest_factor_val, skew_val, kurtosis_val, labels])
        
    return features, featuresheader

def channelextractionfreq(df, labels):
    features = []

    featuresheader = ['Channel', 'Maximum', 'Sum', 'Mean', 'Variance', 'Peak', 'Skew', 'Kurtosis', 'Label']
        
    for channel in df.columns: 
        channeldata = df[channel]
            
        #Frequency Domain
        ft = fft(channeldata.values)
        S = np.abs(ft)**2 / len(channeldata)
        maxf_val = np.max(S)
        sumf_val = np.sum(S)
        meanf_val = np.mean(S)
        varf_val = np.var(S)
        peakf_val = np.max(np.abs(S))
        skewf_val = stats.skew(S)
        kurtosisf_val = stats.kurtosis(S)
            
        features.append([channel, maxf_val, sumf_val, meanf_val, 
                         varf_val, peakf_val, skewf_val, kurtosisf_val, labels])
        
    return features, featuresheader