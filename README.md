# De-Noising Speech segments with a Convolutional Autoencoder

In this project, a basic speech denoising model is developed around a convolutional autoencoder. The model is trained using short audio samples of digits being spoken. Noise is artificially added to the base samples, and the originals are used as the targets. Upon completion of training, the model is tested using similar audio samples from a different speaker. Results show that the model can successfully reconstruct decipherable clean audio regardless of the identity of the speaker.

## The Dataset

The Audio MNIST dataset was created to constitute an audio counterpart to the famous MNIST hand-written digit dataset, for the purpose of training neural networks and other machine learning models to classify audio [1]. The repository for the dataset is in (https://github.com/soerenab/AudioMNIST), though for this work was used through a Kaggle upload (https://www.kaggle.com/sripaadsrinivasan/audio-mnist). 

## Approach

Using the audio samples in the Audio MNIST dataset, a secondary dataset is created for the task of training a de-noising model. 
This is done by creating a "noisy" counterpart to every "clean" file, by simply adding RMS weighted white noise to the original.
Due to storage and cache memory constraints, the files in the original datasete were reshuffled, and the first 1000 were chosen as the training dataset.

Spectrograms are calculated for the audio data, and after reshaped into the appropriate tensor form, are used as input (noisy) and targets (clean). The spectrograms are also normalized within the 0-1 range, which ensures compliance with the activations and loss function used in the neural network, as well stability during the training process.

The model architecture chosen is a relatively simple Convolutional Autoencoder, with two convolutional stages and a max pooling layer in the encoder part, and a symmetrical structure of two deconvolutions and an up sampling layer for the decoder. A sigmoid activation is used for the output layer, and the binary crossentropy loss function was found to be the most effective for the denoising task.

![alt text](https://github.com/N11K6/Speech_DeNoiser_AE/blob/main/images/autoencoder_schem.png?raw=true)

It should be noted, that although more complex autoencoder architectures were tested, including Dense layer latent space representations, the minimalistic architecture above proved to be more efficient given the small size of the dataset [2], and the relatively basic separation task.

## Results

* The autoencoder is trained for 25 epochs, which appear to be enough to reach its capability for loss minimization. Depicted below is an example from the training data given to the model. On the left, the spectrogram of the original clean file that was used as a target, in the middle is the input given with the added noise. On the right is the de-noised spectrogram as output by the autoencoder.

  ![alt text](https://github.com/N11K6/Speech_DeNoiser_AE/blob/main/images/train_spectrograms.png?raw=true)

  Just from the visual, one can see that it is not a straight forward task to make out the original form from the noisy spectrogram. The model succeeds in reconstructing the basic forms, however, some amount of information is lost to the noise. Nevertheless, the result remains decipherable in its spoken content, and the noise is almost entirely eliminated.

  *The sound files for the above spectrograms can be found in the "audio_files" directory as "clean.wav", "noisy.wav" and "denoised.wav".*

* To test the practical effectiveness of the denoising model, an input is chosen from the audio in the original dataset that was not included in the training samples. This is an utterance of the word "five" by an entirely different speaker. From the images below, it can be seen that the autoencoder still manages to reconstruct the original spectrogram to a significant degree, while substantially reducing the surrounding noise.

  ![alt text](https://github.com/N11K6/Speech_DeNoiser_AE/blob/main/images/test_spectrograms.png?raw=true)
  
  *The sound files for the above spectrograms can be found in the "audio_files" directory as "test_clean.wav", "test_noisy.wav" and "test_denoised.wav".*
  
  Overall, the autoencoder-based denoiser appears adequate in its task, especially considering its simplicity. Further development could begin from expansion of the training dataset, using more files, and a more diverse implementation of noise to the clean samples (different "colors" of noise, other recordings superimposed). 
Given this, the actual capabilities of more advanced Autoencoder architectures could be taken advantage of.

## Flask API

A very bare-bones API implementation based around the interaction between a server and a client is included. The server needs to be instantiated first by running the corresponding program. By calling the client script, a request is sent to the server to read from a specified audio file and denoise it using the autoencoder. The server then returns the denoised data in a list. The client script in turn receives this list and writes it into a new audio file (*"flask_denoised.wav"* in the *"audio_files"* directory). As I gain more experience with Flask and REST APIs I aim to make a more substantial and user-friendly implementation.

## Repository Contents

* *Notebooks* : Directory for the *.ipynb* file accompanying the code. This notebook outlines the feature generation process and methodology of the project.
* *audio_files* : Directory for the audio files used in the above notebook and in the demonstration of the model.
* *images* : Directory for the images used in this presentation.
* *trained_models* : Directory for storing the trained models.
* **preprocess.py** : Preprocessing pipeline to assemble a training dataset by reading in from the original audio samples and generating clean and noisy spectrograms.
* **train.py** : Training pipeline for the denoising autoencoder.
* **denoise.py** : Program utilizing the denoiser object to perform denoising on an audio sample, using the pre-trained model.
* **server.py** : Server for the Flask implementation.
* **client.py** : Client for the Flask implementation.

## References

> [1] Becker, Sören & Ackermann, Marcel & Lapuschkin, Sebastian & Müller, Klaus-Robert & Samek, Wojciech. (2018). Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals. 

> [2] Huajian Fang, Guillaume Carbajal, Stefan Wermter, Timo Gerkmann. (2021). Variational Autoencoder for Speech Enhancement with a Noise-Aware Encoder. arXiv:2102.08706
