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



def process_folder(folder_path):
    downloads_folder = os.path.expanduser("~/Downloads")  # Get the path to the "Downloads" directory
    dataverse_files_folder = os.path.join(downloads_folder, "seizure")  # Combine with the folder name

# Use glob to get a list of file paths matching a specific pattern
    file_paths = glob(os.path.join(dataverse_files_folder, '*.*'))  #Getting all files with any extension
    return "Folder processed successfully."