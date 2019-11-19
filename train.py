#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 22:55:03 2019

@author: tacacs1101
"""

from models.lm1 import smallLM
from configs.config import Config
from configs.parameters import PARAMS as params


def train_lm(model, filepath):
    model.fit(filepath, train_mode=True)
    permission = input("Do you want training to proceed\n [yes/no] : ")
    if permission.lower() == 'yes':
        model.train()
    else:
        print('model training terminated')
    return


if __name__=='__main__':
    
    filepath = '/home/tacacs1101/Documents/Rahul/projects/poetry/data/poetry.txt'
    cnf = Config(params)
    model = smallLM(cnf)
    train_lm(model, filepath)