#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:33:45 2019

@author: tacacs1101
"""
import numpy as np
from . import general_utils as ut
from utils.build_data import BuildData
from keras.utils import to_categorical

class Preprocessor():
    
    def __init__(self, config):
        self.config = config
    
    def __initialize_build_data(self, filepath, config):
        bd = ut.check_build_data()
        if bd is None:
            bd = BuildData(filepath, config)
            bd.build_req_data()
        
        self.build_data = bd
    
        
    def fit_on_data(self, filepath):
        self.__initialize_build_data(filepath, self.config)
        self.w2idx = ut.load_index_dict(self.build_data.w2idx_path)
        self.vocab_len = self.build_data.vocab_len
        self.nsamples = self.build_data.nsamples
        self.data_dir = self.build_data.data_dir
            
    def __word_to_int(self, batch_seq):
        int_batch_seq = []
        for seq in batch_seq:
            int_seq = [self.w2idx[w] if w in self.w2idx else self.w2idx['<unk>'] for w in seq]
            int_batch_seq.append(int_seq)
        del batch_seq
        return int_batch_seq
    
    def idx_to_word(self, pred_seq):
        idx2w = ut.load_index_dict(self.build_data.idx2w_path)
        word_seq = [idx2w[p] for p in pred_seq]
        del pred_seq
        return word_seq
    
    def __pad_seq(self, int_batch_seq):
        max_len = max(len(i) for i in int_batch_seq)
        padded_seq = []
        for int_seq in int_batch_seq:
            int_seq = int_seq + [self.w2idx['<pad>']]*(max_len-len(int_seq))
            padded_seq.append(int_seq)
        del int_batch_seq
        padded_seq = np.asarray(padded_seq)
        return padded_seq
    
    def get_embeddings(self):
        if self.config.use_pretrained_embed:
            embed = np.load(self.build_data.embed_matrix_path)
            return embed['embeddings']
        else:
            raise Exception('No pretrained embedding ....')   
            
    def gen_batch(self, filepath):
        while True:
            input_seq = []
            target_seq = []
            index = 0
            fp = open(filepath)
            while index<self.config.batch_size:
                line = fp.readline()
                if line == "":
                    fp.seek(0)
                    line = fp.readline()
                line = line.strip('\n').strip()
                if line:
                    line = "<sos> " + line + " <eos>"
                    inp_words = line.split()
                    input_seq.append(inp_words[:-1])
                    target_seq.append(inp_words[1:])
                    index += 1
            input_seq = self.__word_to_int(input_seq)
            input_seq = self.__pad_seq(input_seq)
            target_seq = self.__word_to_int(target_seq)
            target_seq = self.__pad_seq(target_seq)
            target_one_hot = []
            for t in target_seq:
                t_one_hot = to_categorical(t, len(self.w2idx))
                target_one_hot.append(t_one_hot)
            target_one_hot = np.array(target_one_hot)
            initial_state = np.zeros((input_seq.shape[0], self.config.num_word_lstm))
            yield [input_seq, initial_state, initial_state], target_one_hot
            
    
        
if __name__=='__main__':
    import pprint
    from PARAMETERS import PARAMS 
    from config import Config
    cnf = Config(PARAMS)
    filepath = '/home/tacacs1101/Documents/Rahul/projects/poetry/data/train.txt'
    pr = Preprocessor(filepath, cnf)
    c = 0
    for i,j in pr.gen_batch(filepath):
        print(i.shape)
        print(j.shape)
        #pprint.pprint(i)
        c += 1
        if c>1:
            break
    #w2id = pr.word_to_int()
    #w2id = sorted(w2id.items(),key = lambda x: x[1] )
    #pprint.pprint(w2id)
    #pprint.pprint(vars(pr))