# coding= UTF-8
#
# Author: Fing
# Date  : 2017-12-03
#

import numpy as np
import scipy
import sys
sys.path.append('/home/suhas/Desktop/Sem2/audio-classification/data_try/libsvm-3.24/python/')
from svmutil import *
import sklearn
from sklearn.model_selection import train_test_split


# Load data from numpy file
X_1 = np.load('db1.npy',allow_pickle = True)
X_1 = X_1[::,1:50:] 
#X = X[::,1:10:1]
X_2 = np.load('db4energy.npy')
X_2 = X_2[::,1:6:1] 
#X_3 = np.load('feat_mel.npy')
X = np.concatenate((X_1,X_2),axis=1)
y =  np.load('labels.npy').ravel()

#X = np.load('db4_6.npy')
#y =  np.load('db4_labels.npy').ravel()

# Split data into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
 
# Read data in LIBSVM format

m = svm_train(y_train, X_train, '-c 4')
p_label, p_acc, p_val = svm_predict(y_test, X_test, m)
svm_save_model('libsvm.model', m)
#m = svm_load_model('libsvm.model')
print(p_acc)
