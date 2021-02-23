#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a program to train an audio denoising autoencoder using the audio-MNIST
dataset of spoken digits.

@author: nk
"""

#%% Dependencies
import numpy as np
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split

#%%

PATH_TO_CLEAN = "path/to/clean/features/"
PATH_TO_NOISY = "path/to/noisy/features/"
PATH_TO_TRAINED_MODEL = "path/to/trained/model.h5"
#%%
def build_autoencoder(input_shape, learning_rate = 0.001):
    '''
    Builds the convolutional autoencoder:
        args:
            input_shape : tuple indicating the shape of the input tensor
            learning_rate : starting learning rate
        returns:
            autoencoder : compiled model of the autoencoder
    '''    
    # Input layer:
    encoder_input = keras.Input(shape=input_shape)
    # Encoder stage:
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoder_output = layers.MaxPooling2D((2, 2), padding='same')(x)
    # Decoder stage:
    decoder_input = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_output)
    x = layers.UpSampling2D((2, 2))(decoder_input)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    decoder_output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    # Autoencoder:
    autoencoder = keras.Model(encoder_input, decoder_output)
    autoencoder.summary()
    # Optimizer, learning rate:
    opt = keras.optimizers.Adam(learning_rate = learning_rate)
    # Compile Autoencoder w/ optimizer and binary crossentropy loss:
    autoencoder.compile(optimizer=opt, loss='binary_crossentropy')
    return autoencoder

#%%
def train(autoencoder, X_train, y_train, X_validation, y_validation,
          epochs = 20, batch_size = 24, patience = 3):
    
    '''
    Trains the autoencoder:
        args:
            autoencoder : The compiled autoencoder
            X_train : Training inputs
            y_train : Training targets
            X_validation : Validation inputs
            y_validation : Validation targets
            epochs : Number of epochs
            batch_size : Batch size in samples
            patience : Patience for early stopping
        returns:
            history : Training history
    '''
    
    my_callbacks = [keras.callbacks.EarlyStopping(patience = patience), 
                keras.callbacks.ReduceLROnPlateau(monitor="loss", patience=3),]
    
    history = autoencoder.fit(X_train, y_train,
                              validation_data = (X_validation, y_validation),
                              epochs = epochs,
                              batch_size = batch_size,
                              callbacks = my_callbacks,
                              shuffle = True,
                              verbose = 2)
    return history

#%%
def main():
    
    # Load clean and noisy spectrogram tensors:
    Clean = np.load(PATH_TO_CLEAN)
    Noisy= np.load(PATH_TO_NOISY)
    
    # Perform train-test split on data: 
    # (Inputs are noisy, targets are clean)
    x_train, x_val, y_train, y_val = train_test_split(Noisy,
                                                        Clean,
                                                        test_size = 0.2)
    
    INPUT_SHAPE = (Noisy.shape[1], Noisy.shape[2], Noisy.shape[3])
    
    AUTOENCODER = build_autoencoder(INPUT_SHAPE)
    
    HISTORY = train(AUTOENCODER,
                    x_train, y_train,
                    x_val, y_val)
    
    AUTOENCODER.save(PATH_TO_TRAINED_MODEL)

#%%
if __name__ == "__main__":
    main()