#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:57:04 2019

@author: tacacs1101
"""
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Lambda
from keras.optimizers import Adam
from keras.models import load_model
from models.base_lm import BaseLM


class smallLM(BaseLM):
    
    def __init__(self, config):
        self.config = config
        super().__init__(config)
        
        
    def _build_model(self):
        
        embed_matrix = self.dp.get_embeddings()
        num_words = self.dp.vocab_len
        embed_dim = self.conf.embed_dim
        lstm_units = self.conf.num_word_lstm
        print('Building the model ......')
        
        input_layer = Input((None, ), name='input_layer')
        initial_h = Input((lstm_units, ), name='initial_hidden')
        initial_c = Input((lstm_units, ), name='initial_cell')
        embed = Embedding(num_words, embed_dim, weights=[embed_matrix],trainable=False, name='embedding')
        x = embed(input_layer)
        lstm = LSTM(lstm_units, return_sequences=True, return_state=True, name='lstm')
        x, h, c = lstm(x, initial_state=[initial_h, initial_c])
        dense = Dense(num_words, activation='softmax', name='dense')
        output_layer = Lambda(lambda x: x, name='output_layer')
        logits, h, c = output_layer([dense(x), h, c])
        self.model = Model(input=[input_layer, initial_h, initial_c], output=logits)
        