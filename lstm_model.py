# -*- coding: utf-8 -*-
"""
Created on Sun May 19 20:39:57 2019

@author: aleks
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import os
from attr import dataclass
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential
import json
from numpy import newaxis
import datetime as dt
from sklearn.preprocessing import normalize
from keras.models import load_model

class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, train, test):
        
        if train is None:
            test_only = True
        else:
            test_only = False
        
        if not test_only:
            train_array = np.concatenate((train.X, train.y), axis=1)
            
        test_array = np.concatenate((test.X, test.y), axis=1)
        
        del train, test
        
        self.data_train = train_array if not test_only else None
        self.data_test  = test_array
        self.len_train  = len(self.data_train) if not test_only else None
        self.len_test   = len(self.data_test)
        self.len_train_windows = None


    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        
        return x,y

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        
        x = window[:-1]
        y = window[-1, [0]]
        
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = normalize(window, axis=1, norm='l1')
            normalised_data.append(normalised_window)
            data = np.array(normalised_data)
        return data




class Model():
    
    def __init__(self, configs):
        self.model = Sequential()
        self.configs = configs
        
    def build_model(self, input_dim):

        for layer in self.configs['model']['layers']:
        
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            
            if layer['type'] == 'dense':
            	self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
            	self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
            	self.model.add(Dropout(dropout_rate))
            
        self.model.compile(loss=self.configs['model']['loss'], optimizer=self.configs['model']['optimizer'])
        
        print('[Model] Model Compiled')
        
        
    def train(self, x, y, epochs, batch_size):
        
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		
        save_fname = os.path.join(self.configs["data"]["save_dir"], '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
			EarlyStopping(monitor='val_loss', patience=2),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
		]
        
        
        # try this
        #y = y.reshape((y.shape[0], y.shape[1], 1))
        
        self.model.fit(
			x,
			y,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks
		)
        
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        
        
    def load_h5_model(self, model_name):
        self.model = load_model(model_name)
        
    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch):
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
		
        save_fname = os.path.join(self.configs["data"]["save_dir"], '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
			ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
		]
        
        self.model.fit_generator(
			data_gen,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			callbacks=callbacks,
			workers=1
		)
		
        print('[Model] Training Completed. Model saved as %s' % save_fname)
        
    
    def predict_point_by_point(self, data):
		#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted
    
    
    def predict_sequences_multiple(self, data, window_size, prediction_len):
        #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
                prediction_seqs.append(predicted)
                return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        #Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        return predicted
