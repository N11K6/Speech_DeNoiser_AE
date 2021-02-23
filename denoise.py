#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program to denoise a short speech sample using a pre-trained autoencoder.

PATH_TO_TRAINED_MODEL : path to the pre-trained model (.h5)
PATH_TO_AUDIO : path to the noisy audio file (.wav)
PATH_TO_SAVE : path to save the denoised audio output (.wav)

@author: nk
"""
#%% Dependencies
import numpy as np
import librosa
import soundfile
from tensorflow import keras
#%%
PATH_TO_TRAINED_MODEL = "./trained_models/audio_denoise_AE.h5"
PATH_TO_AUDIO = "./audio_files/test_noisy.wav"
PATH_TO_SAVE = "./audio_files/new_denoised.wav"

#%%
class _Denoise_AE:
    '''
    Singleton class for denoising short audio samples of spoken words.
    '''
    
    model = None
    
    _instance = None
    
    # This is the fitting constant, saved from the training session!
    fitting_constant = 7.259422170994068
    # This is the sample rate that the model is configured to work with.
    SAMPLE_RATE = 22050
    
    def preprocess(self, path_to_audio):
        '''
        Preprocesses audio file located at specified path.
        - Fixes length to 1s
        - Extracts spectrogram
        '''
        
        data, _ = librosa.load(path_to_audio, sr = self.SAMPLE_RATE)
        
        duration = self.SAMPLE_RATE
        
        # Pad to appropriate length...
        if len(data) < duration:
            max_offset = np.abs(len(data) - duration)
            offset = np.random.randint(max_offset)
            data = np.pad(data, (offset, duration-len(data)-offset), "constant")
        # ... or cut to appropriate length...
        elif len(data) > duration:
            max_offset = np.abs(len(data) - duration)
            offset = np.random.randint(max_offset)
            data = data[offset:len(data)-max_offset+offset]
        # ... or leave as is.
        else:
            offset = 0
        # Spectrogram
        S = np.abs(librosa.stft(data))[:-1,:]
        
        return S
    
    def denoise(self, path_to_audio):
        '''
        Denoises input with autoencoder.
        '''
        # Load spectrogram
        S = self.preprocess(path_to_audio)
        # Get dimensions
        dim_1 = S.shape[0]
        dim_2 = S.shape[1]
        # Reshape as input tensor
        S = np.reshape(S, (1, dim_1, dim_2, 1))
        S /= self.fitting_constant
        # Get denoised spectrogram from autoencoder
        S_denoised = self.model.predict(S).reshape((dim_1, dim_2))
        
        # Convert denoised spectrogram to time series waveform
        denoised = librosa.griffinlim(S_denoised) * self.fitting_constant
        
        return denoised
#%%
def Denoise_AE():
    # Ensure single instance of AE
    if _Denoise_AE()._instance is None:
        _Denoise_AE._instance = _Denoise_AE()
        _Denoise_AE.model = keras.models.load_model(PATH_TO_TRAINED_MODEL)
    return _Denoise_AE._instance
#%%
if __name__ == "__main__":
    
    dnae = Denoise_AE()
    dnae2 = Denoise_AE()
    
    assert dnae is dnae2
    
    denoised = dnae.denoise(PATH_TO_AUDIO)
    
    soundfile.write(PATH_TO_SAVE, denoised, dnae.SAMPLE_RATE) 