#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a function-based pipeline to preprocess the Audio-MNIST dataset for 
audio denoising applications. An extra entry is generated by adding noise to 
one of the audio files, and the spectrograms are calculated and stored as 
numpy arrays to be used as input.

@author: nk
"""

#%% Dependencies:
import numpy as np
import os
from os import walk
import librosa

#%%
PATH_TO_DATASET = "path/to/dataset/"
PATH_TO_CLEAN = "path/to/clean/features/"
PATH_TO_NOISY = "path/to/noisy/features/"

#%%
def get_filepaths(dataset_path):
    '''
    Generates a list of the paths to all audio files in the dataset.
    args:
        dataset_path : string indicating path to dataset
    returns:
        filepaths : list of paths to the audio files
    '''
    filepaths = []
    
    # Append list with the path to each file:
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for name in files:
            filepaths.append(os.path.join(root, name))
    
    # Keep in mind: The last file in the Kaggle dataset is a txt and should be
    # discarded from the filepaths list (uncomment the following line):
    
    #filepaths = filepaths[:-1]
    
    print(len(filepaths), 'audio files in the dataset.')
    
    return filepaths

#%%
def get_all_spectrograms(filepaths, noise_gain = 1, sample_rate = 22050, duration_s = 1):
    '''
    Function to generate features for audio de-noising:
    - Reads from paths to audio files specified in "filepaths" list. 
    - Sets to a fixed length.
    - Generates spectrogram.
    - Adds noise and generates corresponding spectrogram.
    - Outputs clean and noisy spectrogram features.
    args:
        filepaths : list of paths to the audio files
        noise_gain : gain for the noise to be applied to the audio samples
        sample_rate : sample rate to load the audio
        duration_s : duration in seconds to set for all samples
    returns:
        X_clean : array of spectrograms for the clean audio
        X_noisy : array of spectrograms for the noisy audio
    '''
    X_clean = []
    X_noisy = []
    
    # Get duration in samples:
    duration = int(sample_rate * duration_s)
    
    for filepath in filepaths:
        
        # Read from audio file:
        data, _ = librosa.load(filepath, sr = sample_rate)
        
        # Pad/Cut to appropriate length:
        if len(data) < sample_rate:
            max_offset = np.abs(len(data) - duration)
            offset = np.random.randint(max_offset)
            data = np.pad(data, (offset, duration-len(data)-offset), "constant")
        
        elif len(data) > sample_rate:
            max_offset = np.abs(len(data) - duration)
            offset = np.random.randint(max_offset)
            data = data[offset:len(data)-max_offset+offset]
        
        else:
            offset = 0
   
        # Clean spectrogram:
        S = np.abs(librosa.stft(data))[:-1,:] # Hacky fix to keep the size reconstructable
        # Clean feature array:
        X_clean.append(S)
        
        # Generate noise at appropriate level:
        RMS=np.sqrt(np.mean(np.abs(data**2)))
        noise=np.random.normal(0, RMS, data.shape[0])
        # Create noisy waveform:
        data_noisy = data + noise_gain * noise
        # Noisy spectrogram:
        S_noisy = np.abs(librosa.stft(data_noisy))[:-1,:] # Hacky fix to keep the size reconstructable
        # Noisy feature array:
        X_noisy.append(S_noisy)
    
    # Convert lists to numpy ndarrays:
    X_clean = np.array(X_clean)
    X_noisy = np.array(X_noisy)

    # Expand dimensions to be used as input tensors:
    X_clean = np.expand_dims(X_clean, -1)
    X_noisy = np.expand_dims(X_noisy, -1)
      
    print('Shape of clean feature tensor:', X_clean.shape)
    print('Shape of noisy feature tensor:', X_noisy.shape)
    
    return X_clean, X_noisy

#%%
def zero_one_normalize(data_to_transform, data_to_fit):
    '''
    Normalizes data#1 according to data#2. If the same data is used as args,
    the normalization is in the range 0, 1 (or -1, 1 if negative values).
    args:
        data_to_transform : array to normalize
        data_to_fit : reference array
    returns:
        normalized_data : transformed array
    '''    
    fitting_constant = np.max(np.abs(data_to_fit))
    
    normalized_data = data_to_transform / fitting_constant
    
    return normalized_data

#%% 
def main():
    '''
    Main function: Runs all functions to preprocess data and store features
    '''
    # Get paths to audio files:
    FILEPATHS = get_filepaths(PATH_TO_DATASET)
    # Get spectrograms:
    # NOTE: DUE TO LARGE FILE SIZE IT IS RECOMMENDED TO CHECK WITH FEWER
    # SAMPLES. EACH SPEAKER HAS 500 SAMPLES.
    X_clean, X_noisy = get_all_spectrograms(FILEPATHS[:500])
    # Normalize clean according to noisy (noisy has higher max value):
    X_clean_n = zero_one_normalize(X_clean, X_noisy)
    # Normalize noisy within 0,1:
    X_noisy_n = zero_one_normalize(X_noisy, X_noisy)
    # Save as arrays for input:
    np.save(PATH_TO_CLEAN, X_clean_n)
    np.save(PATH_TO_NOISY, X_noisy_n)

#%%
if __name__ == "__main__":
    main()