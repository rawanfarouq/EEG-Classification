import scipy
import numpy as np
import mne 
import pandas as pd
import os

mat = scipy.io.loadmat(r'C:\Users\HP\Downloads\mat_sub-023_ses-03_task_motorimagery_eeg.mat')

eegData = mat['data']
sample=eegData[:,0,0]
#print(eegData.shape)
print(sample.size)
labels =  mat['labels']     #mat.get('labels', None)
#print(labels.shape)
channels = eegData[0,:,0]
#print(channels.size)
FS=[]
for i in range(0,99):
    L=[]
    si=eegData[i,:,:]
    for j in range(0,32):
        f=np.mean(si[j,:])
        f1=np.std(si[j,:])
        L.append(f)
        L.append(f1)
    L.append(labels[0,i])
    FS.append(L)
#allf = pd.concat(FS, ignore_index=False)
dffeatures = pd.DataFrame(FS)
current_dir = os.path.dirname(os.path.realpath(__file__))
csvpath = os.path.join(current_dir, 'featurestest1.csv')
dffeatures.to_csv(csvpath, index=True)
#print(len(FS[0]))
        



if len(eegData.shape) == 3:
    trials, channels, samples = eegData.shape
    eegData = np.concatenate(eegData, axis=1)
    eegData = eegData.reshape(channels, trials * samples)

channelnames = ['EEG' + str(i)  for i in range(channels)]
channeltypes = ['eeg'] * channels

info = mne.create_info(ch_names=channelnames,sfreq=500,ch_types=channeltypes)
raw = mne.io.RawArray(eegData, info=info)
labels =  mat['labels']     #mat.get('labels', None)
if labels is not None:
    labels = labels.flatten()
print(labels.shape)    
df = pd.DataFrame(eegData.T)
print()