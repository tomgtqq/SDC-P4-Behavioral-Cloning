import numpy as np
import pandas as pd
# import cv2
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras import layers
from keras.regularizers import l2
# from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


def load_data(file_path):
    '''
    Load image and generate data batch
    parameter：
        file_path: the data csv file path
    return：
        train_generator: train dataset batchs generator
        valid_generator: valid dataset batchs generator
    '''
    # read data from csv
    df = pd.read_csv(file_path)
    new_df = pd.DataFrame()
    new_df['features'] = pd.concat([df['center'], df['left'], df['right']], ignore_index=True)
    new_df['labels']  = pd.concat([df['steering'], df['steering']+CORRECTION, df['steering']-CORRECTION], ignore_index=True)
    
    train_df, valid_df = train_test_split(new_df, test_size=0.2, random_state=42)
    
    datagen=ImageDataGenerator() #featurewise_std_normalization=True
    
    train_generator=datagen.flow_from_dataframe(dataframe=train_df, directory=dir_path, x_col="features", y_col="labels", 
                                                class_mode="other", target_size=(160, 320), batch_size=BATCH_SIZE)
    
    valid_generator=datagen.flow_from_dataframe(dataframe=valid_df, directory=dir_path, x_col="features", y_col="labels", 
                                                class_mode="other", target_size=(160, 320), batch_size=BATCH_SIZE)

    return train_generator, valid_generator
    
    

def build_model():
    '''
    Create model base on Nvidia End to End Learning for Self-Driving Cars arXiv:1604.07316v1    
    '''
    model = Sequential([
    layers.Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)),
    layers.Cropping2D(cropping=((70,25), (20,20)), input_shape=(160, 320, 3)),
    layers.Conv2D(24, kernel_size=(5, 5), subsample =(2, 2), activation='relu'),
    layers.Conv2D(36, kernel_size=(5, 5), subsample =(2, 2), activation='relu'),
    layers.Conv2D(48, kernel_size=(5, 5), subsample =(2, 2), activation='relu'),   
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),  #,kernel_regularizer='l2'
    layers.Dense(50, activation='relu' ),  #,kernel_regularizer='l2'
    layers.Dense(10, activation='relu' ),  #,kernel_regularizer='l2'
    layers.Dense(1)
])

    model.summary()
    return model
    
    
def compile_model( model, train_generator, valid_generator):
    '''
    compile model config loss as mse optimizer as Adam
    '''
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4),
        ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    ]
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    
    model.compile(loss='mse', optimizer='Adam') 
    
    history = model.fit_generator(train_generator, steps_per_epoch=STEP_SIZE_TRAIN, 
                                  validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,
                                  epochs=EPOCHS, verbose=1, callbacks=callbacks)

if __name__ == '__main__':
    
    #training parameters
    BATCH_SIZE = 64
    EPOCHS = 100
    
    
    #data parameters
    CORRECTION = 0.2
    dir_path = "./data"
    file_path = dir_path +"/driving_log.csv"

    # load data
    data = load_data(file_path)
    
    # build model
    model = build_model()
    
    # compile model
    compile_model(model, *data)