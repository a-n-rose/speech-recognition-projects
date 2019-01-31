 
## Extract Speech Features and Train Deep Neural Networks

These scripts allow for the extraction of 2 popular sets of speech features for machine and deep learning: mel filterbank energies (FBANK) and mel frequency cepstral coefficients (MFCC). 

The features can then be fed to a convolutional neural network (CNN), long short-term memory network (LSTM), or stacked CNN+LSTM. The data generator organizes the data to the correct dimensions for each. 


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

2) Downloaded speech data (instructions below)

3) Python 3.6

To check your version type the following into the command-line (Linux):

```
python3 --version
```

4) Install required pacakges:

To install all the python packages we will use, first start up your virtual environment:

```
source env/bin/activate
```
 
In your virtual environment, run 'requirements.txt' to install all necessary packages via pip.

```
(env)..$ pip install -r requirements.txt
```

## Download the Data

Download the speech commands dataset <a href="download.tensorflow.org/data/speech_commands_v0.01.tar.gz">here</a>. Save the zip folder in a subdirectory called 'data' in your working directory. Extract the zipfile.

Note: you can explore other kinds of speech data, as long as the wavefiles are saved in directories named by the label the wavefile belongs to. i.e. in your .data/ directory, there should be subdirectories with names such as 'healthy' and 'clinical'; 'female' and 'male'; or, as in the speech commands dataset, all the words the neural network should be able to recognize: 'bed', 'bird', etc. In each of those folders should be recordings beloning to that class/label.

If you would like to explore male vs female speech, I have directions for how download some <a href="https://a-n-rose.github.io/2019/01/31/small-female-male-speech-data.html">here</a>. With that database, you could also explore classifying different speech disorders.

Note: the speech commands dataset includes a folder with background noises that can be used in training. If you'd like to include a background noise, you'll have to either record background noise yourself and input that filename or find some other background noise. It's not necessary though.

## Default Settings

The default settings for the scripts are listed below. To change them, you'll have to go into their corresponding functions. Pretty easy to do though. :)

### Defaults for feature extraction:

* MFCC or FBANK features (and STFT values) extracted at windows of 25ms with 10ms shifts

* sampling rate = 16000

* As of now, you will extract features from *ALL* wavefiles (balanced across classes, meaning all classes have the same number of wavefiles). If you want to change that, there is functionality for that in the variable: 'limit'

### Defaults for model architecture:

Loosely based on this paper: 

Kim, Myungjong & Cao, Beiming & An, Kwanghoon & Wang, Jun. (2018). Dysarthric Speech Recognition Using Convolutional LSTM Neural Network. 10.21437/interspeech.2018-2250. (available <a href="https://www.researchgate.net/publication/327350843_Dysarthric_Speech_Recognition_Using_Convolutional_LSTM_Neural_Network/related">here</a>)

* CNN: 1 CNN, 1 MaxPooling, 1 Hidden Dense Layer

* LSTM: 2 LSTMs stacked

* CNN+LSTM: 1 CNN, 1 MaxPooling, 2 LSTMs stacked


## Script Functionality

### 'coll_save_features_npy.py'

Duration: to process entire speech commands dataset (with balanced classes), my computer took appx. 2 hours.

I would read through the script before running. In the script you can enter if you would like to get FBANK or MFCC features, how many, also if you would like to get their 1st and 2nd derivatives (i.e. deltas), as well as other options.

In sum, the script performs the following:

1) collects data and labels from subdirectories and subdirectory names.

2) organizes data into balanced train, validation, and test datasets (if you don't want balanced datasets, you'll have to adjust a few things... sorry. I might add functionality for that as well someday.)

3) extracts speech features, and if specified, adds noise and removes beginning silences. The features and labels are then saved to .npy files

### 'train_models_CNN_LSTM_CNNLSTM.py'

Duration: the shortest processing time is with the CNN. The LSTM models depend on the number of units/cells. These run on my CPU just fine. When I trained the CNN+LSTM with the entire (balanced) speech commands dataset, early stopping took place at epoch 18 out of 50, which took 1 hour and 45 minutes. Accuracy on test data was 64.8% with all 30 words (For reference, the *winners* of the Kaggle competition scored around 92%).

Again, I advise reading the script through to understand its components, especially the generator function. You can enter which type of model you would like to train, and ultimately you can also add/remove layers, change model settings, etc. 

This script does the following: 

1) loads data from .npy files.

2) trains CNN, LSTM, or CNN+LSTM models with data via a data generator

## Extract Features

Open the file 'coll_save_features_npy.py'. Set the variables you'd like (i.e. which features, etc.)

Run the script.

```
(env)..$ python3 coll_save_features_npy.py
```

## Train the model

Find the path of the newly created .npy files. Enter these into the script 'train_models_CNN_LSTM_CNNLSTM.py'. Check the script for variables. 

**Important: insert the newly created train, validation, and test .npy filenames!**

```
if __name__=="__main__":

    filename_train = <test filename>
    filename_validation = <validation filename>
    filename_test = <test filename>
```

Run the script.

```
(env)..$ python3 train_models_CNN_LSTM_CNNLSTM.py
```

Depending on the number of wavefiles, labels, features, and model settings, the duration may take from a few minutes to several hours. This runs with out a problem on a CPU. You should be able to use your computer normally (although I personally would avoid running other programs). Note: you might want to turn off any battery saver. If the computer goes into standby, the training can't continue.
