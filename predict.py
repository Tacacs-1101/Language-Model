#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 01:54:14 2019

@author: tacacs1101
"""

################### prediction ##########################

from models.lm1 import smallLM
from configs.config import Config
from configs.parameters import PARAMS as params


def sample_new_lines(model, filepath, nlines):
    model.fit(filepath, train_mode=False)
    new_lines = model.sample_lines(nlines)
    for line in new_lines:
        print(' '.join(line))
        
        
        
        
if __name__=='__main__':
    filepath = '/home/tacacs1101/Documents/Rahul/projects/poetry/data/poetry.txt'
    cnf = Config(params)
    model = smallLM(cnf)
    nlines = 10
    sample_new_lines(model, filepath, nlines)