#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the server for a basic Flask REST API implementation of the audio
denoisng autoencoder. 

The server receives an audio file from the client, and calls the autoencoder
to perform the denoising. The denoised time series (numpy array) is returned 
in a dictionary as a list, which can then be written into an audio file by
the client script.

@author: nk
"""
#%% Dependencies
from flask import Flask, request
from denoise_ae import Denoise_AE
import os
import random

#%%
app = Flask(__name__)

@app.route("/", methods = ["POST"])

def denoise():
    
    # get audio file and save it:
    audio_file = request.files["file"] # request file
    file_name = str(random.randint(0,100000)) # assign a random file name
    audio_file.save(file_name) # store the audio file under file name
    
    # invoke denoising AE
    dae = Denoise_AE()
    
    # denoise audio
    denoised_audio = dae.denoise(file_name)
    
    # remove input audio file
    os.remove(file_name)

    # send denoised audio file data
    data = {"data" : denoised_audio.tolist()}
    return data
    
#%%
if __name__ == "__main__":
    app.run(debug = False)