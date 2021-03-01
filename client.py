#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the client for a basic Flask REST API implementation of the audio
denoising autoencoder. 

It sends an audio file  to the server, where the autoencoder is called to
perform denoising. The server returns the denoised data in the form of a list
in a dictionary, which is turned to a numpy array and then written into an
audio file by the client.

@author: nk
"""
#%% Dependencies:
import requests
import soundfile
import numpy as np

#%%
URL = "http://127.0.0.1:5000/"
PATH_TO_TEST_AUDIO = "./audio_files/test_noisy.wav"
PATH_TO_DENOISED_AUDIO = "./audio_files/flask_denoised.wav"
SAMPLE_RATE = 22050

if __name__ == "__main__":
    
    # Read from audio file:
    audio_file = open(PATH_TO_TEST_AUDIO, "rb")
    values = {"file": (PATH_TO_TEST_AUDIO, audio_file, "audio/wav")}
    
    # Make request to server:
    response = requests.post(URL, files = values)
    
    # Read the data from the response:
    data = response.json()
    # Write data into an audio file:
    soundfile.write(PATH_TO_DENOISED_AUDIO, np.array(data["data"]), SAMPLE_RATE)
