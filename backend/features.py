import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import mne
import scipy
from scipy import stats
from scipy.fft import fft
from scipy.signal import butter, sosfiltfilt
import os
import threading
from statsmodels.tsa.ar_model import AutoReg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def plot_channels(eegData, path):
    output_dir = os.path.join("frontend", "static", "plots")
    os.makedirs(output_dir, exist_ok=True)
    for channel in range(eegData.shape[1]):
        plt.figure(figsize=(10, 5))
        plt.plot(eegData[:, channel, :])
        plt.title(f"Channel {channel+1}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        file_name = os.path.basename(path)
        file_name = os.path.splitext(file_name)[0]
        os.path.join(file_name, "backend")
        os.path.join(file_name, "plots")
        os.makedirs(file_name, exist_ok=True)
        plot_file_path = os.path.join(output_dir, f"plot_{file_name}_channel{channel+1}.png")
        plt.savefig(plot_file_path)
        plt.close()

def loadData(fileName):

    if fileName.endswith("mat"):
        try:
            mat = scipy.io.loadmat(fileName)
            eegData = mat["data"]
            label = mat["labels"]
            return eegData, label
        except ImportError:
            print("Library Error", "scipy is required to load MAT data.")
            return None
        except Exception as e:
            print("Error", f"Error in loading mat file: {str(e)}")
            return None
        
    else:
        print("Wrong file type")

def use_csp(eegData, labels):
    n_samples, n_channels, n_trials = eegData.shape
    eegData = np.concatenate(eegData, axis = 1)
    eegData_reshaped = eegData.reshape(n_channels, n_samples * n_trials)

    ch_names = [f"ch{i+1}" for i in range(n_channels)] 
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types="eeg")
    raw = mne.io.RawArray(eegData_reshaped, info)

    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, preload=True, 
                                          reject_by_annotation=False)
    epochs_data = epochs.get_data()

    labels = labels.flatten()
    
    min_len = min(len(epochs_data), len(labels))
    epochs_data = epochs_data[:min_len]
    labels = labels[:min_len]

    csp = mne.decoding.CSP(n_components=32, reg=None, log=True, norm_trace=False)
    csp.fit(epochs_data, labels)
    filtered_data = csp.transform(epochs_data)
    filtered_data = np.reshape(filtered_data, (n_samples, 32, 1))

    return filtered_data

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
        headers.extend([f"Channel_{channel+1}_min", f"Channel_{channel+1}_max",
                    f"Channel_{channel+1}_mean", f"Channel_{channel+1}_std",
                    f"Channel_{channel+1}_rms", f"Channel_{channel+1}_var",
                    f"Channel_{channel+1}_power", f"Channel_{channel+1}_peak",
                    f"Channel_{channel+1}_p2p", f"Channel_{channel+1}_crestfactor",
                    f"Channel_{channel+1}_skew", f"Channel_{channel+1}_kurtosis"])
    return L, headers

def ar(channeldata, L, channel, headers, sample):
    order = 3
    ar_model = AutoReg(channeldata, lags=order)
    ar_model_fit = ar_model.fit()
    ar_coeffs = ar_model_fit.params[1:]

    L.extend(ar_coeffs.tolist())
    if sample == 0:
        for lag in range(1, 4):
            headers.append(f"Channel_{channel+1}_ar{lag}")

    return L, headers

def fourier(channeldata, L, channel, headers, sample):
    fft_val = fft(channeldata)

    mean_fft = np.mean(np.abs(fft_val))

    L.extend([mean_fft])
    if sample == 0:
        headers.extend([f"Channel_{channel+1}_fft"]) 

    return L, headers

def psd(channeldata, L, channel, headers, sample):
    fft_val = fft(channeldata)
    psd_val = np.abs(fft_val)**2 / len(channeldata)

    mean_psd = np.mean(np.abs(psd_val))

    L.extend([mean_psd])
    if sample == 0:
        headers.extend([f"Channel_{channel+1}_psd"]) 

    return L, headers

def fdjt(channeldata, L, channel, headers, sample):
    fdjt_val = fft(channeldata)
    fdjt_val = np.abs(fdjt_val) / len(channeldata)

    mean_fdjt = np.mean(np.abs(fdjt_val))

    L.extend([mean_fdjt])
    if sample == 0:
        headers.extend([f"Channel_{channel+1}_fdjt"]) 

    return L, headers

def features(eegData, labels, method):
    FS = []
    headers = []
    samples = eegData[:,0,0]
    channels = eegData[0,:,0]
    for sample in range(samples.size):
        L=[]
        si=eegData[sample,:,:]
        for channel in range(channels.size):
            channeldata = si[channel,:]

            if method == "stat":
                L, headers = stat(channeldata, L, channel, headers, sample)
            elif method == "ar":
                L, headers = ar(channeldata, L, channel, headers, sample)
            elif method == "fft":
                L, headers = fourier(channeldata, L, channel, headers, sample)
            elif method == "psd":
                L, headers = psd(channeldata, L, channel, headers, sample)
            elif method == "fdjt":
                L, headers = fdjt(channeldata, L, channel, headers, sample)

        L.append(labels[0,sample])
        FS.append(L)
    headers.append("Label")
    return FS, headers

def allfeatures(FS, headers, csv_path, method, csvpaths):
    dffeatures = pd.concat(FS, ignore_index=True)
    dffeatures.columns = headers
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if method == "stat":
        current_dir = os.path.join(current_dir, "statistical")
    elif method == "ar":
        current_dir = os.path.join(current_dir, "autoregression")
    elif method == "fft":
        current_dir = os.path.join(current_dir, "fourier")
    elif method == "psd":
        current_dir = os.path.join(current_dir, "psd")
    elif method == "fdjt":
        current_dir = os.path.join(current_dir, "fdjt")
    
    os.makedirs(current_dir, exist_ok=True)
    csvpath = os.path.join(current_dir, csv_path)
    dffeatures.to_csv(csvpath, index=False)
    csvpaths.append(csvpath)
    return csvpath

subbands = [(0, 100), (0.5, 4), (4, 8), (8, 12), (12, 30), (30, 100)]

def filtering(data, fs, lowcut, highcut, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], analog=False, btype="band", output="sos")
    
    samples, channels, trials = data.shape
    data_2d = np.reshape(data, (channels, samples * trials))

    filtered_data_2d = sosfiltfilt(sos, data_2d, axis=-1)

    filtered_data = np.reshape(filtered_data_2d, (samples, channels, trials))
    
    return filtered_data

def extract_freq(band_idx, lowcut, highcut, eegData, labels, subfeatures, subheaders, method):
    if band_idx == 0:
        filtered_data = eegData
    else:
        filtered_data = filtering(eegData, fs=250, lowcut=lowcut, highcut=highcut)
    subband_features, subband_headers = features(filtered_data, labels, method)
    subfeatures[band_idx].append(pd.DataFrame(subband_features))
    subheaders[band_idx] = subband_headers

def extract_features(file_paths, method, best, subfeatures, subheaders, csvpaths):
    threads = []
    global max_value, bestpath, loadeddata, loadedlabels
    for index in range(len(file_paths)):
        for band_idx, (lowcut, highcut) in enumerate(subbands, start=0):
            name = "freqthread_", band_idx, file_paths[index]
            name = threading.Thread(target=extract_freq, 
                    args=(band_idx, lowcut, highcut, loadeddata[index], loadedlabels[index], subfeatures, subheaders, method))
            name.start()
            threads.append(name)
        for thread in threads:
            thread.join()

    for band_idx in range(len(subbands)):
        csv_path = f"subband_{band_idx}_features.csv"
        allfeatures(subfeatures[band_idx], subheaders[band_idx], csv_path, method, csvpaths)

    for index in range(0,6):
        name = "thread_", index
        name = threading.Thread(target=classificationrf, args=(index, best, csvpaths))
        name.start()
        threads.append(name)

    for thread in threads:
        thread.join()

def classificationrf(index, best, csvpaths):
    global max_value, bestpath

    df = pd.read_csv(csvpaths[index])
    X = df.drop(columns=["Label"]) 
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                        y, test_size=0.2, random_state=42)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)
    best[index] = accuracy_score(y_test, y_pred) * 100
    print("rf done", csvpaths[index], best[index])
    if best[index] > max_value:
            max_value = best[index]
            bestpath = csvpaths[index]

def compare_methods(paths):
    global loadeddata, loadedlabels
    for index in range(len(paths)):
        path = paths[index]
        eegData, labels = loadData(path)
        filtereddata = use_csp(eegData, labels)
        loadeddata.append(filtereddata)
        loadedlabels.append(labels)
        #plot_channels(eegData, path)
    methods = ["ar", "fft", "psd", "fdjt", "stat"]
    for method in methods:
        best = [0,0,0,0,0,0]
        csvpaths = []
        subfeatures = [[] for _ in range(6)]
        subheaders = [[] for _ in range(6)]
        extract_features(paths, method, best, subfeatures, subheaders, csvpaths)
        print(max_value, "% :", bestpath)
max_value = 0
bestpath = ""
loadeddata = []
loadedlabels = []
def main():
    paths = [r"D:\GUC\Bachelor\Datasets\CrossSessionVariability\mat\subject1\sub-001_ses-01_task_motorimagery_eeg.mat", 
             r"D:\GUC\Bachelor\Datasets\CrossSessionVariability\mat\subject1\sub-001_ses-02_task_motorimagery_eeg.mat", 
             r"D:\GUC\Bachelor\Datasets\CrossSessionVariability\mat\subject1\sub-001_ses-03_task_motorimagery_eeg.mat", 
             r"D:\GUC\Bachelor\Datasets\CrossSessionVariability\mat\subject1\sub-001_ses-04_task_motorimagery_eeg.mat",
             r"D:\GUC\Bachelor\Datasets\CrossSessionVariability\mat\subject1\sub-001_ses-05_task_motorimagery_eeg.mat"]
    compare_methods(paths)
if __name__ == "__main__":
    main()