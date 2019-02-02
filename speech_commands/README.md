 
## Extract Speech Features and Train Deep Neural Networks

In this workshop, our goal is to experiment with speech feature extraction and the training of deep neural networks in Python.

Applying deep learning to the speech signal has many uses. We all know that we can speak to Siri and Alexa, but speech can also be used for security purposes, as in <a href="https://arxiv.org/abs/1803.05427">speaker verification</a>, <a href="https://www.dw.com/en/voice-analysis-an-objective-diagnostic-tool-based-on-flawed-algorithms/a-17187057">healthcare contexts</a>, for example, identifying if a person has Parkinson's or if they have Attention Deficit Hyperactivity Disorder, among other applications as well. 

In much of the research I have read, recurring speech features used in machine and deep learning are the <a href="http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/">Mel Frequency Cepstral Coefficients</a> (MFCC) and the Mel Filterbank Energies (FBANK), which are similar but less filtered than the MFCC.

Additionally, deep learning neural networks I see quite often are the convolutional neural network and, for time series analysis, long short-term memory neural networks. 

Python offers libraries for audio analysis, <a href="https://librosa.github.io/">Librosa</a>, as well as for deep learning, <a href="https://keras.io/">Keras</a>. In this workshop, we will explore speech feature extraction using Librosa and the training of neural networks via Keras. 

For Installation instructions, see <a href="https://github.com/a-n-rose/speech-recognition-projects/blob/master/speech_commands/INSTALLATION.md">here</a>.

After you have installed everything, I suggest reading through the scripts:

1) coll_save_features_npy.py

2) generator_speech_CNN_LSTM.py

Commented out lines should give you some information about the script and what we want to do.

To get started, start up your virtual environment:

```
$ source env/bin/activate
```
As it is, the script 'coll_save_features_npy.py' will extract only 5% of all the speech data. If you run the script 'as is', this will take appx. 6 minutes to complete. 

```
(env)..$ python3 coll_save_features_npy.py
```
This should print out something like this when it's done:

```
TO TRAIN A MODEL, COPY AND PASTE THE FOLLOWING INTO THE MODEL TRAINING SCRIPT:


features_and_models_2019y2m2d22h6m19s


```
This is just a folder name with a unique time stamp. Copy this from your command line, by highlighting it and pressing ctrl+shift+C. Open 'train_models_CNN_LSTM_CNNLSTM.py', scroll to the bottom of the script, and where it declares the variable 'project_head_folder', paste it.

```
if __name__ == "__main__":
    
    project_head_folder = "features_and_models_2019y2m2d22h6m19s"
```

Once you've done that, run the script. As is, the script should take appx. 4 minutes. 

Once the scripts are through, you can look through the newly created files. Graphs depicting how well the model trained can be found there. 

In the workshop we will explore these scripts in detail, and change parameters to see how the training might be affected. Note: due to time constraints, we will not train on the entire dataset in the workshop. You are encouraged to do so at home, perhaps leaving the computer on and train through the night. 


## Future Use

You can explore other kinds of speech data, as well. I am collecting a list of free and available speech databases <a href="https://a-n-rose.github.io/2019/01/06/resources-publicly-available-speech-databases.html">here</a>. I offer directions for downloading a small dataset of female and male speech <a href="https://a-n-rose.github.io/2019/01/31/small-female-male-speech-data.html">here</a>. Note: in order to work with these scripts, the wavefiles need to be separated by class within the 'data' folder.
