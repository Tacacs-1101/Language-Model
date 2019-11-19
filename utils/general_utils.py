#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 00:26:27 2019

@author: tacacs1101
"""

####################  General utility functions #######################

import os, shutil
import pickle as pkl
import re

def make_safe_dir(*dir_path):
    for path in dir_path:
        if not os.path.exists(path):
            os.mkdir(path)
    return 

def clear_directory(dir_path):
    for the_file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def print_dirs(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
            
def load_index_dict(path):
    with open(path, 'rb') as handle:
        index_dict = pkl.load(handle)
    return index_dict

def check_build_data():
    bd_dir = os.path.join(os.getcwd(), 'model_data/tmp')
    if os.path.exists(bd_dir):
        bd_path = os.path.join(bd_dir, 'build_data.pickle')
        with open(bd_path, 'rb') as handle:
            bd = pkl.load(handle)
            return bd
    return None

def count_nlines(filepath):
    nsamples = 0
    with open(filepath) as f:
            for line in f:
                line = line.strip('\n').strip()
                if line:
                    nsamples += 1
    return nsamples

def get_epoch_from_filename(filename):
    p = r'weight[.]\d+[.]h'
    if re.search(p, filename):
        epoch = re.search(p, filename).group(0)
        epoch = re.search('\d+', epoch).group(0)
        epoch = int(epoch)
        return epoch
    else:
        raise Exception('pattern not matched for getting epoch value')
    

def get_latest_ckpt(ckpt_dir, filename_suffix='best_model.weight'):
    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith(filename_suffix)]
    sorted_ckpts = sorted(ckpts, key=get_epoch_from_filename)
    sorted_ckpts_path = [os.path.join(ckpt_dir, f) for f in sorted_ckpts]
    return sorted_ckpts_path[0]
    
            





