import sys
import os
import numpy as np
from tqdm import tqdm
import librosa
import time

import feature_extraction_functions as featfun


current_filename = os.path.basename(__file__)
session_name = get_date()


#variables to set:
feature_type = "fbank" # "mfcc", "stft"
num_filters = 40 # 13, None
delta = False # True
noise = True #False
vad = True #voice activity detection
sampling_rate = 16000
window = 25
shift = 10
timesteps = 5
context_window = 5
frame_width = context_window*2 + 1



#collect labels
data_path = "./data_3words"
labels_class = featfun.collect_labels(data_path)
print(labels_class)

#create labels-encoding dictionary
labels_sorted = sorted(labels_class)
dict_labels_encoded = {}
for i, label in enumerate(labels_sorted):
    dict_labels_encoded[label] = i
print(dict_labels_encoded)
#save the labels for implementing model
featfun.save_class_labels(labels_sorted,session_name)

#collect filenames
paths, labels_wavefile = featfun.collect_audio_and_labels(data_path)
noise_path = "./data/_background_noise_/doing_the_dishes.wav"

#to balance out the classes, find label w fewest recordings
max_num_per_class, min_label = featfun.get_min_samples_per_class(labels_class,labels_wavefile)

#create dictionary with labels and their indices in the lists: labels_wavefile and paths
#useful in separating the indices into balanced train, validation, and test datasets
dict_class_index_list = featfun.make_dict_class_index(labels_class,labels_wavefile)
            
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


filename_save_data = "{0}_{1}_delta{2}_noise{3}_sr{4}_window{5}_shift{6}_timestep{7}_framewidth{8}_date{9}".format(feature_type,num_filters,delta,noise,sampling_rate,window,shift,timesteps,frame_width,session_name)
train_val_test_filenames = []
train_val_test_directories = []

for i in ["train","val","test"]:
    new_path = "./data_3words_shuffled_wnoise_vad{}{}_{}/".format(feature_type,num_filters,i)
    train_val_test_filenames.append(new_path+"{}_".format(i)+filename_save_data)
    train_val_test_directories.append(new_path)
    try:
        os.makedirs(new_path)
    except OSError as e:
        print("Directory  ~  {}  ~  already exists".format(new_path))
        pass


start_feature_extraction = time.time()

for i in tqdm(range(3)):
#extract train data and save to pickel file
    dataset_index = i   # 0 = train, 1 = validation, 2 = test
    #limit = int(max_nums_train_val_test[dataset_index]*.01)
    limit = None
    train_features = featfun.save_feats2npy(labels_class,dict_labels_encoded,train_val_test_filenames[dataset_index],max_nums_train_val_test[dataset_index],dict_class_dataset_index_list,paths,labels_wavefile,feature_type,num_filters,timesteps,frame_width,limit=limit,delta=False,noise_wavefile=noise_path,vad=True,dataset_index=dataset_index)

end_feature_extraction = time.time()
print("Duration of feature extraction: {} minutes".format(round((end_feature_extraction-start_feature_extraction)/60,2)))

