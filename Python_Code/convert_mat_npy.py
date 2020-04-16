#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 19:15:29 2020

@author: suhas
"""

import numpy as np
import scipy.io
import os

os.chdir('/home/suhas/Desktop/Sem2/audio-classification')

#Data

#db4_all = scipy.io.loadmat('db4_all.mat')
#db4 = db4_all['db4_all']
#sym5_all = scipy.io.loadmat('sym5_all.mat')
#sym5 = sym5_all['sym5_all']


sym5_has = scipy.io.loadmat('updated_db4_hasBird.mat')
db4_has = sym5_has['data_db4_yes']
sym5_no = scipy.io.loadmat('updated_db4_noBird.mat')
db4_no = sym5_no['data_db4_no']
db4 = np.concatenate((db4_has,db4_no),axis=0)
np.save('wavelets_updated_db4.npy',db4)
#Labels

#labels_all = scipy.io.loadmat('wavelets_labels_all.mat')
#labels_all = labels_all['labels_all']