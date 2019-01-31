'''
Model architectures are inspired from the (conference) paper:

Kim, Myungjong & Cao, Beiming & An, Kwanghoon & Wang, Jun. (2018). Dysarthric Speech Recognition Using Convolutional LSTM Neural Network. 10.21437/interspeech.2018-2250. 

'''

import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt


# for building and training models
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, LSTM, MaxPooling2D, Dropout, TimeDistributed, ConvLSTM2D
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,CSVLogger,ModelCheckpoint



#feed data to models
class Generator:
    def __init__(self,model_type,data,timesteps,frame_width):
        '''
        This generator pulls data out in sections (i.e. batch sizes).
        
        It then prepares that batched data to be the right shape for the 
        following models:
        * CNN or convolutional neural network
        * LSTM or long short-term memory neural network
        * CNN+LSTM, the LSTM stacked ontop of the CNN
        
        These operations performed on small batches allows you to train an 
        algorithm with quite a lot of data without much memory cost. 
        ~ trying to adjust a full dataset will certainly stall most computers ~
        
        This generator pulls data out from just one recording/word at a time.
        We define this when we extracted the features: each recording/word
        gets 5 sets (timesteps) of 11 frames (framewidth).
        
        The batch size is calculated by multiplying the frame width (e.g. 11) with 
        the timestep (e.g. 5). This is the total number of samples alloted 
        each recording/word.
        
        The label of the recording/word is in the last column of the data.
        
        Once the features and the labels are separated, each are shaped 
        according to the model they will be fed to.
        
        CNNs are great with images. Images have heights, widghts, color scheme.
        This is the shape CNNs need: (h,w,c)
        h = number of pixels up
        w = number of pixels wide
        c = whether it is grayscale (i.e. 1), rgb (i.e. 3), or rgba (i.e.4)
        * because our data is not really in color, 1 is just fine.
        
        LSTMs deal with series of data. They want to have things in timesteps.
        (timestep,....) and whatever other data you wanna put in there. We 
        have the number of frames with number of features:
        (timestep, num_frames, num_features)
        Notice here no color scheme number is needed.
        
        CNN+LSTM is a mixture of both. So a mixture of the dimensions is necessary.
        When the data is fed to the network, first it is fed to a "Time Distribution" 
        module, which feeds the data first to a CNN, and then that 
        output to an LSTM. So the data first needs to meet the needs of the CNN, 
        and then the LSTM.
        (timesteps,num_frames,num_features,color_scheme)
        
        Note: Keras adds a dimension to input to represent the "Tensor" that 
        handles the input. This means that sometimes you have to add a 
        shape of (1,) to the shape of the data. 
        
        All this is done here in the generator!
        '''
        self.model_type = model_type
        self.timesteps = timesteps
        self.frame_width = frame_width
        self.batch_size = timesteps * frame_width
        self.samples_per_epoch = data.shape[0]
        self.number_of_batches = self.samples_per_epoch/self.batch_size
        self.counter=0
        self.dict_classes_encountered = {}
        self.data = data

    def generator(self):
        while 1:
            
            #All models, get the features data
            batch = np.array(self.data[self.batch_size*self.counter:self.batch_size*(self.counter+1),]).astype('float32')
            X_batch = batch[:,:-1]
            
            #Only for the LSTM
            if 'lstm' == self.model_type.lower():
                '''
                desired shape to put into model:
                (1,timestep,frame_width,num_features)
                '''
                X_batch = X_batch.reshape((self.timesteps,self.frame_width,X_batch.shape[1]))
                y_batch = batch[:,-1]
                y_indices = list(range(0,len(y_batch),self.frame_width))
                y_batch = y_batch[y_indices]
                
                #keep track of how many different labels are presented in 
                #training, validation, and test datasets
                labels = y_batch[0]
                if labels in self.dict_classes_encountered:
                    self.dict_classes_encountered[labels] += 1
                else:
                    self.dict_classes_encountered[labels] = 1
            
            else:
                #Only for the CNN
                if 'cnn' == self.model_type.lower():
                    #(1,frame_width,num_features,1)
                    X_batch = X_batch.reshape((X_batch.shape[0],X_batch.shape[1],1))
                    
                #Only for the CNN+LSTM
                elif 'cnnlstm' == self.model_type.lower():
                    #(1,timestep,frame_width,num_features,1)
                    #1st 1 --> keras Tensor
                    #2nd 1 --> color scheme
                    X_batch = X_batch.reshape((self.timesteps,self.frame_width,X_batch.shape[1],1))
        
                #Both CNN and CNN+LSTM:
                X_batch = X_batch.reshape((1,)+X_batch.shape)
                y_batch = batch[0,-1]
                y_batch = y_batch.reshape((1,)+y_batch.shape)

                labels = list(set(y_batch))
                if len(labels) > 1:
                    print("Too many labels assigned to one sample")
                if labels[0] in self.dict_classes_encountered:
                    self.dict_classes_encountered[labels[0]] += 1
                else:
                    self.dict_classes_encountered[labels[0]] = 1
            
            #All data:
            #send the batched and reshaped data to models
            self.counter += 1
            yield X_batch,y_batch

            #restart counter to yeild data in the next epoch as well
            if self.counter >= self.number_of_batches:
                self.counter = 0

#to keep saved files unique
#include their names with a timestamp
def get_date():
    time = datetime.datetime.now()
    time_str = "{}y{}m{}d{}h{}m{}s".format(time.year,time.month,time.day,time.hour,time.minute,time.second)
    return(time_str)

def main(model_type,timesteps,frame_width,num_features,color_scale,num_labels,epochs,optimizer,sparse_targets,path_beginning,path_ending):
    
    #get some orientation
    start = time.time()
    current_filename = os.path.basename(__file__)
    session_name = get_date()
    #make model name unique with timestamp
    modelname = "{}_speech_commands_{}".format(model_type.upper(),session_name)
    if num_labels <= 2:
        loss_type = 'binary_crossentropy'
        activation_output = 'sigmoid' # binary = "sigmoid"; multiple classification = "softmax"
    else:
        loss_type = 'categorical_crossentropy'
        activation_output = 'softmax' # binary = "sigmoid"; multiple classification = "softmax"
    if sparse_targets:
        loss_type = 'sparse_categorical_crossentropy' # if data have mutiple labels which are only integer encoded, *not* one hot encoded.

    
    #create folders to store models, logs, and graphs
    for folder in ["graphs","models","model_log"]:
        try:
            os.makedirs(folder)
        except OSError as e:
            print("Directory  ~  {}  ~  already exists".format(folder))
            pass

    
    #build the model architecture:
    lstm_cells = num_features
    feature_map_filters = 30
    kernel_size = (4,8)
    pool_size = (3,3)
    dense_hidden_units = 60
    
    if 'lstm' == model_type.lower():
        model = Sequential()
        model.add(LSTM(lstm_cells,return_sequences=True,input_shape=(frame_width,num_features))) 
        model.add(LSTM(lstm_cells,return_sequences=True))   
        
    elif 'cnn' == model_type.lower():
        model = Sequential()
        # 4x8 time-frequency filter (goes along both time and frequency axes)
        model.add(Conv2D(feature_map_filters, kernel_size=kernel_size, activation='relu',input_shape=(frame_width*timesteps,num_features,color_scale)))
        #non-overlapping pool_size 3x3
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))
        model.add(Dense(dense_hidden_units))
        
    elif 'cnnlstm' == model_type.lower():
        cnn = Sequential()
        cnn.add(Conv2D(feature_map_filters, kernel_size=kernel_size, activation='relu'))
        #non-overlapping pool_size 3x3
        cnn.add(MaxPooling2D(pool_size=pool_size))
        cnn.add(Dropout(0.25))
        cnn.add(Flatten())

        #prepare stacked LSTM
        model = Sequential()
        model.add(TimeDistributed(cnn,input_shape=(timesteps,frame_width,num_features,color_scale)))
        model.add(LSTM(lstm_cells,return_sequences=True))
        model.add(LSTM(lstm_cells,return_sequences=True))

    model.add(Flatten())
    model.add(Dense(num_labels,activation=activation_output)) 
    print(model.summary())
    model.compile(optimizer=optimizer,loss=loss_type,metrics=['accuracy'])


    #set up "callbacks" which help you keep track of what goes on during training
    model_name = modelname+"_{}epochs".format(epochs)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)
    csv_logging = CSVLogger(filename='./model_log/{}_log.csv'.format(model_name))
    checkpoint_callback = ModelCheckpoint('./models/checkpoint_'+model_name+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')


    #load the data!!
    #pathways to npy files containing train, val, and test data
    train_npy_filename = "./{}_train/train_{}.npy".format(path_beginning,path_ending)
    val_npy_filename = "./{}_val/val_{}.npy".format(path_beginning,path_ending)
    test_npy_filename = "./{}_test/test_{}.npy".format(path_beginning,path_ending)

    train_data = np.load(train_npy_filename)
    val_data = np.load(val_npy_filename)

    #load these into their generators
    train_generator = Generator(model_type,train_data,timesteps,frame_width)
    val_generator = Generator(model_type,val_data,timesteps,frame_width)

    start_training = time.time()
    #train the model and keep the accuracy and loss stored in the variable 'history'
    #helpful in logging/plotting how training and validation goes
    history = model.fit_generator(
            train_generator.generator(),
            steps_per_epoch = train_data.shape[0]/(timesteps*frame_width),
            epochs = epochs,
            callbacks=[early_stopping_callback, checkpoint_callback],
            validation_data = val_generator.generator(), 
            validation_steps = val_data.shape[0]/(timesteps*frame_width)
            )
    end_training = time.time()
    #just to keep track and know for sure...
    print("\nNumber of labels presented to the model \n* Training: {}\n* Validation: {}".format(train_generator.dict_classes_encountered,val_generator.dict_classes_encountered))
    
    print("\nNow testing the model..")
    #now to test the model on brandnew data!
    test_data = np.load(test_npy_filename)
    test_generator = Generator(model_type,test_data,timesteps,frame_width)
    score = model.evaluate_generator(test_generator.generator(), test_data.shape[0]/(timesteps*frame_width))
    loss = round(score[0],2)
    acc = round(score[1]*100,3)

    msg="Model Accuracy on test data: {}%\nModel Loss on test data: {}".format(acc,loss)
    print(msg)

    modelname_final = "{}_{}_{}tr_{}va_{}te_images_{}epochs_{}Optimizer_{}acc".format(modelname,session_name,len(train_data),len(val_data),len(test_data),epochs,optimizer,acc)
    print('Saving Model..')
    model.save('./models/'+modelname_final+'.h5')
    print('Done!')
    print("\nModel saved as:\n{}.h5".format(modelname_final))


    print("Now saving history and plots")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("train vs validation loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train","validation"], loc="upper right")
    plt.savefig("./graphs/{}_LOSS.png".format(modelname_final))

    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title("train vs validation accuracy")
    plt.legend(["train","validation"], loc="upper right")
    plt.savefig("./graphs/{}_ACCURACY.png".format(modelname_final))
    
    end = time.time()
    duration_total = round((end-start)/60,3)
    duration_training = round((end_training-start_training)/60,3)
    print("\nTotal duration = {}\nTraining duration = {}\n".format(duration_total,duration_training))
    
    #document settings used in model
    parameters = []
    if "lstm" in model_type.lower():
        parameters.append(([("lstm cells",lstm_cells)]))
    if "cnn" in model_type.lower():
        parameters.append(([("feature maps",feature_map_filters),("pool size",pool_size),("kernel size",kernel_size),("dense hidden units",dense_hidden_units)]))
    parameters.append(([("test accuracy", acc),("test loss",loss)]))
    parms = []
    for item in parameters:
        for i in item:
            parms.append(i)
    
    dict_parameters={}
    dict_parameters[model_type] = parms
    print(dict_parameters)
    
    with open("./model_log/model_parameters.csv","a") as fd:
        fd.write(str(dict_parameters))
    
    return True



if __name__ == "__main__":
    
    model_type = "cnnlstm" # cnn, lstm, cnnlstm
    timesteps = 5
    frame_width = 11
    num_features = 120 #if delta, num_filters * 3
    color_scale = 1 # 1=gray, 3=color, 4=color+alpha?
    num_labels = 30 
    epochs = 50
    optimizer = 'adam' # 'adam' 'sgd'
    sparse_targets = True
    
    #these should *NOT* include "train_", "val_", "_test", or ".npy" etc.
    #these will be added in the script
    file_path = "data_ALLwords_shuffled_fbank40"
    file_name = "fbank40_deltaTrue_noiseTrue_vadTrue_timestep5_framewidth11_numlabels30_date2019y1m31d16h57m0s"
    
    
    main(
        model_type,
        timesteps,frame_width,num_features,color_scale,
        num_labels,epochs,optimizer,sparse_targets,
        file_path,file_name
        )
