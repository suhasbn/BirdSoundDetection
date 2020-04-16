# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import scipy.io
import os

os.chdir('/home/suhas/Desktop/Sem2/audio-classification/NEW/')
x = np.load('mfbe.npy')
x_reshaped = np.reshape(x,(3870,25974))
scipy.io.savemat('mfbe.mat',{"data":x_reshaped})
#y = np.load('label_mfcc.npy')
#scipy.io.savemat('test_label.mat',{"foo_label":y})
