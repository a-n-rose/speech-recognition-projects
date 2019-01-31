import sys
import os
import numpy as np
from tqdm import tqdm
import librosa
import time
import datetime

import feature_extraction_functions as featfun


#to keep saved files unique
#include their names with a timestamp
def get_date():
    time = datetime.datetime.now()
    time_str = "{}y{}m{}d{}h{}m{}s".format(time.year,time.month,time.day,time.hour,time.minute,time.second)
    return(time_str)


def main(data_path,feature_type,num_filters=None,delta=False,noise=False,vad=False,timesteps=None,context_window=None,noise_path=None):
    if num_filters is None:
        num_filters = 40
    if timesteps is None:
        timesteps = 5
    if context_window is None:
        context_window = 5
    if delta:
        num_features = num_filters * 3
    else:
        num_features = num_filters
    
    session_name = get_date()
    
    #############################################
    ################## LABELS ###################
    '''
    Collect all labels in data:
    Labels should be the subdirectories of the data directory
    Not included:
    Folders/files with names:
    * starting with "_"
    * are typical GitHub files, like LICENSE
    '''
    labels_class = featfun.collect_labels(data_path)
    print(labels_class)
    
    '''
    Create labels-encoding dictionary:
    This helps when saving data later to npy files
    Integer encode the labels and save with feature data as label column
    '''
    labels_sorted = sorted(labels_class)
    dict_labels_encoded = {}
    for i, label in enumerate(labels_sorted):
        dict_labels_encoded[label] = i
    print(dict_labels_encoded)
    
    '''
    Create and save to .csv the encoded labels
    Helpful for implementing the model later.
    Better to know speech categorized as 'bird'
    rather than '1'
    '''
    featfun.save_class_labels(labels_sorted,session_name)
    
    
    
    #############################################
    ############## DATA ORGANIZATION ############

    #collect filenames and labels of each filename
    paths, labels_wavefile = featfun.collect_audio_and_labels(data_path)

    #to balance out the classes, find the label/class w fewest recordings
    max_num_per_class, min_label = featfun.get_min_samples_per_class(labels_class,labels_wavefile)

    '''
    Create dictionary with labels and their indices in the lists: labels_wavefile and paths
    useful in separating the indices into balanced train, validation, and test datasets
    '''
    dict_class_index_list = featfun.make_dict_class_index(labels_class,labels_wavefile)
    
    '''
    Calculate number of recordings for each dataset, 
    keeping the data balanced between classes
    * .8 of max number of samples --> train
    * .1 of max number of samples --> validation
    * .1 of max number of samples --> test
    '''
    max_nums_train_val_test = featfun.get_max_nums_train_val_test(max_num_per_class)

    #randomly assign indices to train, val, test datasets:
    dict_class_dataset_index_list = featfun.assign_indices_train_val_test(labels_class,dict_class_index_list,max_nums_train_val_test)

    #make sure no indices mix between datasets:
    train_indices = []
    test_indices = []
    val_indices = []
    for label in labels_class:
        label_indices = dict_class_dataset_index_list[label]
        train_indices.append(label_indices[0])
        val_indices.append(label_indices[1])
        test_indices.append(label_indices[2])
    print("Checking mix of datasets")
    print(len(train_indices[0]))
    #print(train_indices)
    print(len(val_indices[0]))
    print(len(test_indices[0]))
    for train_index in train_indices[0]:
        if train_index in test_indices or train_index in val_indices:
            print("Training Index {} is in other datasets")
    for val_index in val_indices[0]:
        if val_index in train_indices or val_index in test_indices:
            print("Val Index {} is in other datasets")
    for test_index in test_indices[0]:
        if test_index in train_indices or test_index in val_indices:
            print("Test Index {} is in other datasets")


    #############################################
    ############# FEATURE EXTRACTION ############

    frame_width = context_window*2 + 1
    filename_save_data = "{0}{1}_delta{2}_noise{3}_vad{4}_timestep{5}_framewidth{6}_numlabels{7}_date{8}".format(feature_type,num_filters,delta,noise,vad,timesteps,frame_width,len(labels_class),session_name)
    train_val_test_filenames = []
    train_val_test_directories = []

    #Set up directories where the .npy files will be saved (in a train, validation, and test folder)
    for i in ["train","val","test"]:
        new_path = "./data_ALLwords_shuffled_{}{}_{}/".format(feature_type,num_filters,i)
        train_val_test_filenames.append(new_path+"{}_".format(i)+filename_save_data)
        train_val_test_directories.append(new_path)
        try:
            os.makedirs(new_path)
        except OSError as e:
            print("Directory  ~  {}  ~  already exists".format(new_path))
            pass


    start_feature_extraction = time.time()

    for i in tqdm(range(3)):
        dataset_index = i   # 0 = train, 1 = validation, 2 = test
        
        ##if you want to limit the number of recordings that features are extracted from:
        limit = int(max_nums_train_val_test[dataset_index]*.01)
        
        ##limit = None --> All .wav files in all label folders will be processed
        #limit = None
        
        extraction_completed = featfun.save_feats2npy(labels_class,dict_labels_encoded,train_val_test_filenames[dataset_index],max_nums_train_val_test[dataset_index],dict_class_dataset_index_list,paths,labels_wavefile,feature_type,num_filters,num_features,timesteps,frame_width,limit=limit,delta=delta,noise_wavefile=noise_path,vad=vad,dataset_index=dataset_index)
        
        if extraction_completed:
            print("\nRound {} feature extraction successful.\n".format(i))
        else:
            print("\nRound {} feature extraction was unsuccessful.".format(i))

    end_feature_extraction = time.time()
    print("Duration of feature extraction: {} minutes".format(round((end_feature_extraction-start_feature_extraction)/60,2)))

if __name__=="__main__":

    #variables to set:
    
    #which directory has the data?
    data_path = "./data"
    feature_type = "fbank" # "mfcc", "stft"
    num_filters = 40 # 13, None
    delta = True # Calculate the 1st and 2nd derivatives of features?
    noise = True # Add noise to speech data?
    vad = True #voice activity detection
    timesteps = 5
    context_window = 5
    noise_path = "./data/_background_noise_/doing_the_dishes.wav" # None

    main(
        data_path,feature_type,
        num_filters=num_filters,delta=delta,noise=noise,vad=vad,
        timesteps=timesteps,context_window=context_window,noise_path=noise_path
        )
