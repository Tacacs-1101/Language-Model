#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 23:37:23 2019

@author: tacacs1101
"""


class Config():
    
    def __init__(self, params):
        
        self.vocab_size = params['vocab_size']
        self.batch_size = params['batch_size']
        self.use_pretrained_embed = params['use_pretrained_embed']
        self.embed_path = params['word_embed_path']
        self.embed_dim = params['embed_dim']
        self.num_word_lstm = params['num_word_lstm_cells']
        self.epochs = params['epochs']
        
        
    
        
 
