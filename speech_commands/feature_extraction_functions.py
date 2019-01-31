#to work w Windows, Mac, and Linux:
from pathlib import Path, PurePath
#saving labels
import csv

#audio 
import librosa
import librosa.display
import matplotlib.pyplot as plt

#data prep
import numpy as np
import random

#my own speech prep: voice activity detection
import prep_noise as prep_data_vad_noise


def collect_labels(data_path):
    p = Path(data_path)
    labels = list(p.glob('*/'))
    labels = [PurePath(labels[i]) for i in range(len(labels))]
    labels = [x.parts[1] for x in labels if '_' not in x.parts[1]]
    labels = check_4_github_files(labels)

    return labels


def check_4_github_files(labels_list):
    if 'README.md' in labels_list:
        labels_list.remove('README.md')
    if 'LICENSE' in labels_list:
        labels_list.remove('LICENSE')
    return labels_list
    

def collect_audio_and_labels(data_path):
    '''
    expects wavefiles to be in subdirectory: 'data'
    labels are expected to be the names of each subdirectory in 'data'
    speaker ids are expected to be the first section of each wavefile
    '''
    p = Path(data_path)
    waves = list(p.glob('**/*.wav'))
    #remove directories with "_" at the beginning
    paths = [PurePath(waves[i]) for i in range(len(waves)) if waves[i].parts[1][0]!="_"]
    labels = [j.parts[1] for j in paths ]
    
    return paths, labels


def save_class_labels(sorted_labels,session):
    dict_labels = {}
    for i, label in enumerate(sorted_labels):
        dict_labels[i] = label
    filename = 'labels_encoded_date{}.csv'.format(session)
    with open(filename,'w') as f:
        w = csv.writer(f)
        w.writerows(dict_labels.items())
    return None

def get_class_distribution(class_labels,labels_list): 
    dict_class_distribution = {}
    for label in class_labels:
        count = 0
        for label_item in labels_list:
            if label == label_item:
                count+=1
            dict_class_distribution[label] = count
    return dict_class_distribution
        

def get_min_samples_per_class(class_labels, labels_list):
    dict_class_distribution = get_class_distribution(class_labels,labels_list)
    min_val = (1000000, None)
    for key, value in dict_class_distribution.items():
        if value < min_val[0]:
            min_val = (value, key)
    return min_val


def get_max_nums_train_val_test(max_num_per_class):
    max_train = int(max_num_per_class*.8)
    max_val = int(max_num_per_class*.1)
    max_test = int(max_num_per_class*.1)
    sum_max_nums = max_train + max_val + max_test
    if max_num_per_class > sum_max_nums:
        diff = max_num_per_class - sum_max_nums
        max_train += diff
    
    return max_train, max_val, max_test


def get_train_val_test_indices(list_length):
    indices_ran = list(range(list_length))
    random.shuffle(indices_ran)
    train_len = int(list_length*.8)
    val_len = int(list_length*.1)
    test_len = int(list_length*.1)
    sum_indices = train_len + val_len + test_len
    if sum_indices != list_length:
        diff = list_length - sum_indices
        train_len += diff
    train_indices = []
    val_indices = []
    test_indices = []
    for i, item in enumerate(indices_ran):
        if i < train_len:
            train_indices.append(item)
        elif i >= train_len and i < train_len+val_len:
            val_indices.append(item)
        elif i >= train_len + val_len and i < list_length:
            test_indices.append(item)
    return train_indices, val_indices, test_indices


def make_dict_class_index(class_labels,labels_list):
    dict_class_index_list = {}
    for label in class_labels:
        dict_class_index_list[label] = []
        for i, label_item in enumerate(labels_list):
            if label == label_item:
                dict_class_index_list[label].append(i)
    return dict_class_index_list


def assign_indices_train_val_test(class_labels,dict_class_index,max_nums_train_val_test):
    dict_class_dataset_index_list = {}
    for label in class_labels:
        tot_indices = dict_class_index[label]
        tot_indices_copy = tot_indices.copy()
        random.shuffle(tot_indices_copy)
        train_indices = tot_indices_copy[:max_nums_train_val_test[0]]
        val_indices = tot_indices_copy[max_nums_train_val_test[0]:max_nums_train_val_test[0]+max_nums_train_val_test[1]]
        test_indices = tot_indices_copy[max_nums_train_val_test[0]+max_nums_train_val_test[1]:max_nums_train_val_test[0]+max_nums_train_val_test[1]+max_nums_train_val_test[2]]
        dict_class_dataset_index_list[label] = [train_indices,val_indices,test_indices]
    return dict_class_dataset_index_list
 
 
def get_change_acceleration_rate(spectro_data):
    #first derivative = delta (rate of change)
    delta = librosa.feature.delta(spectro_data)
    #second derivative = delta delta (acceleration changes)
    delta_delta = librosa.feature.delta(spectro_data,order=2)

    return delta, delta_delta


def apply_noise(y,sr,wavefile):
    #at random apply varying amounts of environment noise
    rand_scale = random.choice([0.0,0.25,0.5,0.75])
    if rand_scale > 0.0:
        total_length = len(y)/sr
        y_noise,sr = librosa.load(wavefile,sr=16000)
        envnoise_normalized = prep_data_vad_noise.normalize(y_noise)
        envnoise_scaled = prep_data_vad_noise.scale_noise(envnoise_normalized,rand_scale)
        envnoise_matched = prep_data_vad_noise.match_length(envnoise_scaled,sr,total_length)
        if len(envnoise_matched) != len(y):
            diff = int(len(y) - len(envnoise_matched))
            if diff < 0:
                envnoise_matched = envnoise_matched[:diff]
            else:
                envnoise_matched = np.append(envnoise_matched,np.zeros(diff,))
        y += envnoise_matched

    return y


def get_feats(wavefile,feature_type,num_features,delta=False,noise_wavefile = None,vad = False):
    y, sr = get_samps(wavefile)
    if vad:
        y = prep_data_vad_noise.get_speech_samples(y,sr)
    if noise_wavefile:
        y = apply_noise(y,sr,noise_wavefile)
    if delta:
        num_feature_columns = num_features*3
    else:
        num_feature_columns = num_features
        
    extracted = []
    if "mfcc" in feature_type.lower():
        extracted.append("mfcc")
        features = get_mfcc(y,sr,num_mfcc=num_features)
        features -= (np.mean(features, axis=0) + 1e-8)
        if "delta" in feature_type.lower():
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    elif "fbank" in feature_type.lower():
        extracted.append("fbank")
        features = get_mel_spectrogram(y,sr,num_mels = num_features)
        features -= (np.mean(features, axis=0) + 1e-8)
        if "delta" in feature_type.lower():
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    ###!!!!!!! Need to Debug..
    #elif "stft" in feature_type.lower():
        #extracted.append("stft")
        #features = get_stft(y,sr)
        #features -= (np.mean(features, axis=0) + 1e-8)
        #if "delta" in feature_type.lower():
            #delta, delta_delta = get_change_acceleration_rate(features)
            #features = np.concatenate((features,delta,delta_delta),axis=1)
    if features.shape[1] != num_feature_columns: 
        raise FeatureExtractionError("The file '{}' results in the incorrect  number of columns (should be {} columns): shape {}".format(wavefile,num_features,features.shape))
    return features
    

def get_samps(wavefile,sr=None,high_quality=None):
    if sr is None:
        sr = 16000
    if high_quality:
        quality = "kaiser_high"
    else:
        quality = "kaiser_fast"
    y, sr = librosa.load(wavefile,sr=sr,res_type=quality) 
    
    return y, sr


def get_mfcc(y,sr,num_mfcc=None,window_size=None, window_shift=None):
    '''
    set values: default for MFCCs extraction:
    - 40 MFCCs
    - windows of 25ms 
    - window shifts of 10ms
    '''
    if num_mfcc is None:
        num_mfcc = 40
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
    mfccs = librosa.feature.mfcc(y,sr,n_mfcc=num_mfcc,hop_length=hop_length,n_fft=n_fft)
    mfccs = np.transpose(mfccs)
    
    return mfccs


def get_mel_spectrogram(y,sr,num_mels = None,window_size=None, window_shift=None):
    '''
    set values: default for mel spectrogram calculation (FBANK)
    - windows of 25ms 
    - window shifts of 10ms
    '''
    if num_mels is None:
        num_mels = 40
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
        
    fbank = librosa.feature.melspectrogram(y,sr,n_fft=n_fft,hop_length=hop_length,n_mels=num_mels)
    fbank = np.transpose(fbank)
    
    return fbank


def get_stft(y,sr,window_size=None, window_shift=None):
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
    stft = np.abs(librosa.feature.stft(y,sr,n_fft=n_fft,hop_length=hop_length)) #comes in complex numbers.. have to take absolute value
    stft = np.transpose(stft)
    
    return stft


def get_domfreq(y,sr):
    frequencies, magnitudes = get_freq_mag(y,sr)
    #select only frequencies with largest magnitude, i.e. dominant frequency
    dom_freq_index = [np.argmax(item) for item in magnitudes]
    dom_freq = [frequencies[i][item] for i,item in enumerate(dom_freq_index)]
    
    return dom_freq


def get_freq_mag(y,sr,window_size=None, window_shift=None):
    '''
    default values:
    - windows of 25ms 
    - window shifts of 10ms
    '''
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
    #collect frequencies present and their magnitudes
    frequencies,magnitudes = librosa.piptrack(y,sr,hop_length=hop_length,n_fft=n_fft)
    frequencies = np.transpose(frequencies)
    magnitudes = np.transpose(magnitudes)
    
    return frequencies, magnitudes


def save_feats2npy(class_labels,dict_labels_encoded,data_filename4saving,max_num_samples,dict_class_dataset_index_list,paths_list,labels_list,feature_type,num_filters,time_step,frame_width,limit=None,delta=False,noise_wavefile=None,vad=False,dataset_index=0):
    msg = "\nExtracting features from {} samples. \nFeatures will be saved in the file {}".format(max_num_samples,data_filename4saving)
    print(msg)

    #create empty array to fill with values
    expected_rows = max_num_samples*len(class_labels)*frame_width*time_step
    feats_matrix = np.zeros((expected_rows,num_filters+1)) # +1 for the label
    
    #go through all data in dataset and fill in the matrix
    row = 0
    paths_labels_list_dataset = []
    for i, label in enumerate(class_labels):
        #labels_list_dataset = []
        train_val_test_index_list = dict_class_dataset_index_list[label]
        #print(train_val_test_index_list[dataset_index])
        for k in train_val_test_index_list[dataset_index]:
            paths_labels_list_dataset.append((paths_list[k],labels_list[k]))
    
    #shuffle indices:
    random.shuffle(paths_labels_list_dataset)
    
    for j, wav_label in enumerate(paths_labels_list_dataset):

        if limit and j > limit:
            break
        else:
            wav_curr = wav_label[0]
            label_curr = wav_label[1]
            label_encoded = dict_labels_encoded[label_curr]
            feats = coll_feats_manage_timestep(time_step,frame_width,wav_curr,feature_type,num_filters,delta=False, noise_wavefile=noise_wavefile,vad = True)
            #add label column:
            label_col = np.full((feats.shape[0],1),label_encoded)
            feats = np.concatenate((feats,label_col),axis=1)
            
            feats_matrix[row:row+feats.shape[0]] = feats
            row += feats.shape[0]
    np.save(data_filename4saving+".npy",feats_matrix)
    
    return True
    
    
def coll_feats_manage_timestep(time_step,frame_width,wav,feature_type,num_filters,delta=False,noise_wavefile=None,vad = True):
    feats = get_feats(wav,feature_type,num_filters,delta=False,noise_wavefile=noise_wavefile,vad = True)
    max_len = frame_width*time_step
    if feats.shape[0] < max_len:
        diff = max_len - feats.shape[0]
        feats = np.concatenate((feats,np.zeros((diff,feats.shape[1]))),axis=0)
    else:
        feats = feats[:max_len,:]
    
    return feats
        
