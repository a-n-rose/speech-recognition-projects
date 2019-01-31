 
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

Saarbrücker Voice Database. Directions for downloading male and female speech can be found <a href="https://a-n-rose.github.io/2019/01/31/small-female-male-speech-data.html">here</a>.


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


## Set Up

I suggest using a virtual environemt. This allows you to use any package versions without them interfering with other programs on your system. 

You can set up a virtual environment different ways. One way is with Python3.6-venv.

### Python3.6-venv

To install, enter into the command line:

```
sudo apt install python3.6-venv
```

In folder of cloned directory (i.e. "deep_learning_acoustics"), write in the command-line:

```
python3 -m venv env
```

This will create a folder 'env'.

Then type into the command-line:

```
source env/bin/activate
```

and your virtual envioronment will be activated. Here you can install packages that will not interfere with any other programs.

To deactivate this environment, simply type in the command line:

```
deactivate
```
 
## Prerequisites

1) Computer with CPU or GPU and sound recording capabilities, ideally via headset

2) A way to examine SQL tables (I use <a href="https://sqlitebrowser.org/">DB Browser for SQLite</a>)

3) Python 3.6

To check your version type the following into the command-line (Linux):

```
python3 --version
```

To install all the python packages we will use, first start up your virtual environment (in the folder "deep_learning_acoustics"):

```
source env/bin/activate
```

4) In your virtual environment, run 'requirements.txt' to install all necessary packages via pip. 

```
(env)..$ pip install -r requirements.txt
```

## Download the Data

Download the speech commands dataset <a href="download.tensorflow.org/data/speech_commands_v0.01.tar.gz">here</a>

Note: you can explore other kinds of speech data, as long as the wavefiles are saved in directories named by the label the wavefile belongs to. i.e. in your .data/ directory, there should be subdirectories with names such as 'healthy' and 'clinical', 'female' and 'male', or, as in the speech commands dataset, all the words the neural network should be able to recognize, 'bed', 'bird', etc. 

If you would like to explore male vs female speech, I have directions for how to do that <a href="https://a-n-rose.github.io/2019/01/31/small-female-male-speech-data.html">here</a>.

Note: the speech commands dataset includes a folder with background noises. If you'd like to include a background noise, you'll have to set that up.

## Extract Features

Open the file 'coll_save_features_npy.py'. Set the variables you'd like (i.e. which features, etc.)

Run the script.

```
(env)..$ python3 coll_save_features_npy.py
```

## Train the model

Find the path of the newly created .npy files. Enter these into the script 'train_models_CNN_LSTM_CNNLSTM.py'. Check the script for variables. 

Run the script.

```
(env)..$ python3 train_models_CNN_LSTM_CNNLSTM.py
```

Depending on the number of wavefiles, labels, features, and model settings, the duration may take from a few minutes to several hours. This runs with out a problem on a CPU.  
