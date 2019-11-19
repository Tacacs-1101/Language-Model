#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 23:36:21 2019

@author: tacacs1101
"""

################# Build vocabulary and index dictionary ######################################################
import os
import pickle as pkl
from collections import Counter
import numpy as np
from . import general_utils as ut

class BuildData():
    
    def __init__(self, filepath, config):
        self.config = config
        self.filepath = filepath
        self.build_dir()
        

    def build_dir(self):
        curr_dir = os.getcwd()
        data_dir = os.path.join(curr_dir, 'model_data')
        tmp_dir = os.path.join(data_dir, 'tmp')
        vocab_dir = os.path.join(data_dir, 'vocab')
        index_dict_dir = os.path.join(data_dir, 'index_dict')
        embed_dir = os.path.join(data_dir, 'embed')
        ut.make_safe_dir(data_dir,tmp_dir, vocab_dir, index_dict_dir, embed_dir)
        self.data_dir = data_dir
        self.vocab_dir = vocab_dir
        self.index_dict_dir = index_dict_dir
        self.embed_dir = embed_dir
        self.tmp_dir = tmp_dir
        return

    def __build_vocab(self):
        nsamples = 0
        word_freq = Counter()
        with open(self.filepath) as f:
            for line in f:
                line = line.strip('\n').strip()
                if line:
                    words = line.split()
                    word_freq.update(words)
                    nsamples += 1
        self.nsamples = nsamples
        most_freq_word = word_freq.most_common(self.config.vocab_size)
        self.vocab_words_path = os.path.join(self.vocab_dir, 'words.txt')
        
        with open(self.vocab_words_path, 'w') as fw:
            for word, freq in most_freq_word:
                fw.write(str(word) + ' ' + str(freq) + '\n')
        return 
    
    def __word2index(self):
        w2idx = {}
        index = 4
        with open(self.vocab_words_path, 'r') as fr:
            for line in fr:
                line = line.strip('\n').strip()
                if line:
                    word, _ = line.split()
                    w2idx[word] = index
                    index += 1
        w2idx['<pad>'] = 0
        w2idx['<unk>'] = 1
        w2idx['<sos>'] = 2
        w2idx['<eos>'] = 3
        self.vocab_len = len(w2idx)
        self.w2idx_path = os.path.join(self.index_dict_dir, 'word2index.pickle')
        with open(self.w2idx_path, 'wb') as handle:
            pkl.dump(w2idx, handle, protocol=pkl.HIGHEST_PROTOCOL)
        del w2idx
        return
        
    def __idx2word(self):
        self.idx2w_path = os.path.join(self.index_dict_dir, 'index2word.pickle')
        with open(self.w2idx_path, 'rb') as fr_handle:
            w2idx = pkl.load(fr_handle)
        idx2w = {i:w for w, i in w2idx.items()}
        with open(self.idx2w_path, 'wb') as fw_handle:
            pkl.dump(idx2w, fw_handle, protocol=pkl.HIGHEST_PROTOCOL)
        del idx2w
        return 
    
    
    def __embedding_matrix(self):
        embedding_matrix = np.zeros((self.vocab_len, self.config.embed_dim))
        embed_dict = {}
        with open(self.config.embed_path, 'r') as f:
            for line in f:
                line = line.strip('\n').strip()
                wv = line.split()
                word, vector = wv[0], wv[1:]
                embed_dict[word] = np.array(vector)
        with open(self.w2idx_path, 'rb') as fr_handle:
            w2idx = pkl.load(fr_handle)
        for w, i in w2idx.items():
            vec = embed_dict.get(w)
            if vec is not None:
                embedding_matrix[i] = vec
        del embed_dict
        del w2idx
        print('shape of the embedding matrix is {}'.format(embedding_matrix.shape))
        self.embed_matrix_path = os.path.join(self.embed_dir, 'embed_matrix.npz')
        np.savez_compressed(self.embed_matrix_path, embeddings=embedding_matrix)
        return
      
            
    def build_req_data(self):
        print('building and writing vocab.....')
        self.__build_vocab()
        print('building and saving index dictionary...')
        self.__word2index()
        self.__idx2word()
        if self.config.use_pretrained_embed:
            print('Building and saving embedding matrix ....')
            self.__embedding_matrix()
        ut.print_dirs(self.data_dir)
        obj_path = os.path.join(self.tmp_dir, 'build_data.pickle')
        with open(obj_path, 'wb') as fw_handle:
            pkl.dump(self, fw_handle, protocol=pkl.HIGHEST_PROTOCOL)
        return
        
        
if __name__=='__main__':
    from PARAMETERS import PARAMS 
    from config import Config
    
    cnf = Config(PARAMS)
    filepath = '/home/tacacs1101/Documents/Rahul/projects/poetry/data/train.txt'
    bd = BuildData(filepath, cnf) 
    bd.build_req_data()