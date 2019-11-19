#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:51:18 2019

@author: tacacs1101
"""
import os
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TerminateOnNaN
from keras.models import load_model
import utils.general_utils as ut
from utils.data_preprocessor import Preprocessor

class BaseLM():
    
    def __init__(self, conf):
        self.conf = conf
        self.model = None
        
    
    def __train_test_split(self,filepath):
        val = 0.1
        data_dir = os.path.join(self.dp.data_dir, 'train_test')
        ut.make_safe_dir(data_dir)
        self.num_train_samples = int(self.dp.nsamples*(1.0-val))
        self.num_val_samples = self.dp.nsamples-self.num_train_samples
        train_path = os.path.join(data_dir, 'train.txt')
        val_path = os.path.join(data_dir, 'val.txt')
        curr_count = 0
        fp = open(filepath)
        curr_count = 0
        curr_line = None
        with open(train_path, 'w') as ftrain:
            while curr_count < self.num_train_samples:
                curr_line = fp.readline()
                line = curr_line.strip('\n').strip()
                if line:
                    ftrain.write(line+'\n')
                    curr_count += 1
        with open(val_path, 'w') as fval:
            while (curr_line!=""):
                curr_line = fp.readline()
                line = curr_line.strip('\n').strip()
                if line:
                    fval.write(line+'\n')
        self.train_path = train_path
        self.val_path = val_path
        return
    
     
    def __steps_per_epoch(self):
        self.steps_per_epoch = self.num_train_samples // self.conf.batch_size + 1
        self.val_steps = self.num_val_samples // self.conf.batch_size + 1
        return
    
    def __build_data(self,filepath):
        self.dp = Preprocessor(self.conf)
        self.dp.fit_on_data(filepath)
        self.__train_test_split(filepath)
        self.__steps_per_epoch()

    def _build_model(self):
        '''
        Implement this method in child class while building network.
        '''
        pass
    def __get_sampling_model(self):
        self.sample_model = Model(inputs=self.model.input, outputs=self.model.get_layer('output_layer').output)
        
    
    def fit(self,filepath, train_mode=False, load_from_best=True):
        self.__build_data(filepath)
        if train_mode:
            self._build_model()
        else:
            self.__load_trained_model(load_from_best)
            self.__get_sampling_model()
        
    def train(self, learning_rate=None):
        optimizer = Adam()
        if learning_rate:
            optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        train_gen = self.dp.gen_batch(self.train_path)
        val_gen = self.dp.gen_batch(self.val_path)
        checkpoint_dir = os.path.join(os.getcwd(), 'model_data/checkpoints_dir')
        ut.make_safe_dir(checkpoint_dir)
        ut.clear_directory(checkpoint_dir)
        self.__best_model_path = os.path.join(checkpoint_dir, 'best_model.weight.{epoch:}.hdf5') 
        best_weight_ckpt = ModelCheckpoint(self.__best_model_path, monitor='acc', period=self.conf.epochs//10)
        default_callbacks = [best_weight_ckpt, TerminateOnNaN()]
        self.model.fit_generator(train_gen, self.steps_per_epoch,
                            epochs=self.conf.epochs,
                            validation_data=val_gen, 
                            validation_steps=self.val_steps,
                            callbacks=default_callbacks)
        
        self.__model_path = os.path.join(checkpoint_dir, 'final_model.hdf5')
        self.model.save(self.__model_path)
        return
    
    def __load_trained_model(self, load_from_best=True):
        if load_from_best:
            ckpt_path = ut.get_latest_ckpt('model_data/checkpoints_dir')
        else:
            ckpt_path = 'model_data/final_model.hdf5'
        if not self.model:
            self.model = load_model(ckpt_path)
    
    def __sample_one_line(self, max_words=10):
       
        sos = self.dp.w2idx['<sos>']
        eos = self.dp.w2idx['<eos>']
        inp = np.asarray([[sos]])
        
        output_idx = []
        for i in range(max_words):
            z = np.zeros((1, self.conf.num_word_lstm))
            output, h, c = self.sample_model.predict([inp, z, z])
            probs = output[0,0]
            if np.argmax(probs)==0:
                continue
            probs[0] = 0
            norm_probs = probs/probs.sum()
            idx = np.random.choice(len(probs), p=norm_probs)
            if idx==eos:
                break
            output_idx.append(idx)
            inp[0,0] = idx
        sent = self.dp.idx_to_word(output_idx)
        return sent
            
    def sample_lines(self, nlines):
        batch_output = []
        for n in range(nlines):
            line_out = self.__sample_one_line()
            batch_output.append(line_out)
        return batch_output
            
        


        
        
        
if __name__=='__main__':
    from PARAMETERS import PARAMS 
    from config import Config
    filepath = '/home/tacacs1101/Documents/Rahul/projects/poetry/data/poetry.txt'
    cnf = Config(PARAMS)
    lm = BaseLM(cnf)
    lm.fit(filepath) 
    lm.train()
        
        
        
        