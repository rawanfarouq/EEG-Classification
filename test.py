from glob import glob
import os    #reading file paths or directories.
import mne   # For working with EEG data
import numpy as np
import pandas
import matplotlib.pyplot as pltp
from keras.models import Sequential
from keras.layers import Conv1D ,BatchNormalization,LeakyReLU ,MaxPooling1D # Importing layers suitable for 1D data
from keras.layers import Flatten ,GlobalAveragePooling1D,Dense,Dropout,AveragePooling1D
from keras.backend import clear_session
import sys
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold,GridSearchCV, LeaveOneGroupOut
  

downloads_folder = os.path.expanduser("~/Downloads")  # Get the path to the "Downloads" directory
dataverse_files_folder = os.path.join(downloads_folder, "dataverse_files")  # Combine with the folder name

# Use glob to get a list of file paths matching a specific pattern
file_paths = glob(os.path.join(dataverse_files_folder, '*.*'))  #Getting all files with any extension

# Print the list of file paths with folder names prefixed
for file_path in file_paths:
    folder_name, file_name = os.path.split(file_path)
    print(f"'{folder_name}|{file_name}'")


healthy_file_path=[]
patient_file_path=[]

for file_path in file_paths:
    folder_name,file_name=os.path.split(file_path)
    if(file_name.startswith('h')):
        healthy_file_path.append(file_path)
    elif(file_name.startswith('s')):
        patient_file_path.append(file_path)   

print(len(healthy_file_path),len(patient_file_path))     

def read_data(file_paths):
    data=mne.io.read_raw_edf(file_paths,preload=True)
    data.set_eeg_reference()  # helps to remove common noise sources and artifacts from the EEG data
    data.filter(l_freq=0.5,h_freq=45)
    epochs=mne.make_fixed_length_epochs(data,duration=5,overlap=1)  #divides continuous data into smaller segments called epochs
    array=epochs.get_data()
    return array

sample_data=read_data(healthy_file_path[0])
print(sample_data.shape)  #no of epochs, channels, length of signal

with open(os.devnull, "w") as f:
    sys.stdout = f  # Redirect stdout to null device

    control_epochs_array = [read_data(i) for i in healthy_file_path]
    patient_epochs_array = [read_data(i) for i in patient_file_path]

    sys.stdout = sys.__stdout__  # Reset stdout

# print(control_epochs_array[0].shape,control_epochs_array[1].shape)

control_epochs_labels=[len(i)*[0] for i in control_epochs_array]
patient_epochs_labels=[len(i)*[1] for i in patient_epochs_array]
print(len(control_epochs_labels),len(patient_epochs_labels))

epochs_array=control_epochs_array+patient_epochs_array
epochs_labels=control_epochs_labels+patient_epochs_labels

group_list=[[i]*len(j) for i,j in enumerate(epochs_array)]

data_array=np.vstack(epochs_array)
label_array=np.hstack(epochs_labels)  
group_array=np.hstack(group_list)

# print(data_array.shape,label_array.shape,group_array.shape)

# EXTRACT FEATURES FROM DATA

def mean(x):
    return np.mean(x,axis=-1)

def std(x):
     return np.std(x,axis=-1)

def ptp(x):
    return np.ptp(x,axis=-1)

def var(x):
    return np.var(x,axis=-1)

def minim(x):
    return np.min(x,axis=-1)

def maxim(x):
    return np.max(x,axis=-1)

def argminim(x):
    return np.argmin(x,axis=-1)

def argmaxim(x):
    return np.argmax(x,axis=-1)

def rms(x):
    return np.sqrt(np.mean(x**2,axis=-1))

def abs_diff_signal(x):
    return np.sum(np.abs(np.diff(x,axis=-1)),axis=-1)

def skewness(x):
    return stats.skew(x,axis=-1)

def kurtosis(x):
    return stats.kurtosis(x,axis=-1)

def concatenate_features(x):
    return np.concatenate((mean(x), std(x), ptp(x), var(x), minim(x), maxim(x),
                           argminim(x), argmaxim(x), rms(x), abs_diff_signal(x),
                           skewness(x), kurtosis(x)), axis=-1)


features=[]
for d in data_array:
    features.append(concatenate_features(d))

features_array=np.array(features)
print(features_array.shape) 

clf= LogisticRegression()
gkf=GroupKFold(5)
pipe=Pipeline([('scaler',StandardScaler()),('clf',clf)])
param_grid={'clf__C':[0.1,0.5,0.7,1,3,5,7]}
gscv=GridSearchCV(pipe,param_grid,cv=gkf,n_jobs=12)
gscv.fit(features_array,label_array,groups=group_array)
            
print(gscv.best_score_)


# DEEP LEARNING CNN

epochs_array=np.vstack(epochs_array)
epochs_labels=np.hstack(epochs_labels)
group_array=np.hstack(group_array)

print(epochs_array.shape,epochs_labels.shape, group_array.shape)

#(no of echops, channels, length) moving the channels to the end

epochs_array=np.moveaxis(epochs_array,1,2)
print(epochs_array.shape)

def cnnmodel():
    clear_session()
    model=Sequential()
    model.add(Conv1D(filters=5,kernel_size=3,strides=1,input_shape=(1250,19)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling1D(pool_size=2,strides=2))
    model.add(Conv1D(filters=5,kernel_size=3,strides=1))
    model.add(LeakyReLU())
    model.add(MaxPooling1D(pool_size=2,strides=2))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=5,kernel_size=3,strides=1))
    model.add(LeakyReLU())
    model.add(AveragePooling1D(pool_size=2,strides=2))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=5,kernel_size=3,strides=1))
    model.add(LeakyReLU())
    model.add(AveragePooling1D(pool_size=2,strides=2))
    model.add(Conv1D(filters=5,kernel_size=3,strides=1))
    model.add(LeakyReLU())
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1,activation='sigmoid'))

    model.compile('adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

model=cnnmodel()
model.summary()

gkf=GroupKFold()

accuracy=[]
for train_index, val_index in gkf.split(epochs_array, epochs_labels, groups=group_array):
    train_features, train_labels = epochs_array[train_index], epochs_labels[train_index]
    val_features, val_labels = epochs_array[val_index], epochs_labels[val_index]
    
    # Fit the StandardScaler on the training data
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features.reshape(-1, train_features.shape[-1])).reshape(train_features.shape)
    
    # Transform both the training and validation data
    val_features = scaler.transform(val_features.reshape(-1, val_features.shape[-1])).reshape(val_features.shape)
    
    model = cnnmodel()
    model.fit(train_features, train_labels, epochs=10, batch_size=1024, validation_data=(val_features, val_labels))
    accuracy.append(model.evaluate(val_features, val_labels)[1])

    break

print(train_features.shape)















