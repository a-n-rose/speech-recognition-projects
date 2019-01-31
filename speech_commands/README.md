 
## Extract Speech Features and Train Deep Neural Networks

These scripts allow for the extraction of 2 popular sets of speech features for machine and deep learning: mel filterbank energies (FBANK) and mel frequency cepstral coefficients (MFCC). 

The features can then be fed to a convolutional neural network (CNN), long short-term memory network (LSTM), or stacked CNN+LSTM. The data generator organizes the data to the correct dimensions for each. 


## Default Settings

The default settings for the scripts are listed below. To change them, you'll have to go into their corresponding functions. Pretty easy to do though. :)

### Defaults for feature extraction:

* MFCC or FBANK features (and STFT values) extracted at windows of 25ms with 10ms shifts

* sampling rate = 16000

### Defaults for model architecture:

Loosely based on this paper: 

Kim, Myungjong & Cao, Beiming & An, Kwanghoon & Wang, Jun. (2018). Dysarthric Speech Recognition Using Convolutional LSTM Neural Network. 10.21437/interspeech.2018-2250. (available <a href="https://www.researchgate.net/publication/327350843_Dysarthric_Speech_Recognition_Using_Convolutional_LSTM_Neural_Network/related">here</a>)

* CNN: 1 CNN, 1 MaxPooling, 1 Hidden Dense Layer

* LSTM: 2 LSTMs stacked

* CNN+LSTM: 1 CNN, 1 MaxPooling, 2 LSTMs stacked

## Data

Compatible with (at least) the following datasets:


<a href="https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html">Speech Commands Dataset</a> (to download directly, click <a href="download.tensorflow.org/data/speech_commands_v0.01.tar.gz">here</a>)


## Script Functionality

### 'coll_save_features_npy.py'

Duration: to process entire speech commands dataset (with balanced classes), my computer took appx. 2 hours.

1) collects data and labels from subdirectories and subdirectory names.

* Note: it expects the subdirectories to contain waves corresponding to the subdirectory name (i.e. subdirectory 'bird' should have wavefiles with recordings of people saying "bird")

2) organizes data into balanced train, validation, and test datasets (if you don't want balanced datasets, you'll have to adjust a few things... sorry. I might add functionality for that as well someday.)

3) extracts speech features and saves to .npy files

### 'train_models_CNN_LSTM_CNNLSTM.py'

Duration: the shortest processing time is with the CNN. The LSTM models depend on the number of units/cells. These run on my CPU just fine. When I trained the CNN+LSTM with the entire (balanced) speech commands dataset, early stopping took place at epoch 18 out of 50, which took 1 hour and 45 minutes. Accuracy on test data was 64.8% (For reference, the winners of the Kaggle competition scored around 92%) 

1) loads data from .npy files

2) trains CNN, LSTM, or CNN+LSTM models with data via a data generator


## Requirements

Will be updated soon...

