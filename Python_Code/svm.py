# coding= UTF-8
#
# Author: Fing
# Date  : 2017-12-03
#

import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot
import pickle



# Load data from numpy file
#X_1 = np.load('db1.npy',allow_pickle = True)
#X_1 = X_1[::,1:50:] 
#X = X[::,1:10:1]
X = np.load('db4energy_normalized.npy')
X = X[::,1:4:1]
#X_2 = X_2[::,1:6:1] 
#X_3 = np.load('feat_mel.npy')
#X = np.concatenate((X_1,X_2),axis=1)
y =  np.load('labels.npy').ravel()

# Split data into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Simple SVM
print('fitting...')
c_list = {0.03,0.1,0.5,2,8,32,128,512}
gamma_list = {0.0019,0.007,0.03,0.1,2}
for c in c_list:
	for g in gamma_list:
		clf = SVC(C=c, gamma=g)
		clf.fit(X_train, y_train)


		lr_probs = clf.predict(X_test)
		ns_probs = [0 for _ in range(len(y_test))]
		ns_auc = roc_auc_score(y_test, ns_probs)
		lr_auc = roc_auc_score(y_test, lr_probs)
		print('No Skill: ROC AUC=%.3f' % (ns_auc))
		print(str(c)+'_'+str(g)+ 'SVM_db4energy: ROC AUC=%.3f' % (lr_auc))
		# calculate roc curves
		#***ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
		#***lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
		# plot the roc curve for the model
		#pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill AUC: 0.5')
		#***pyplot.plot(lr_fpr, lr_tpr, marker='.', label='db4energy_'+str(c)+'_'+str(g)+':%.3f' % (lr_auc))
		# axis labels
		#***pyplot.xlabel('False Positive Rate')
		#***pyplot.ylabel('True Positive Rate')
		# show the legend
		#***pyplot.legend()
		# show the plot
		#pyplot.show()
		#with open('svm_model_split_db4_'+str(c)+'_'+str(g)+'_.pkl','wb') as f:
		#    pickle.dump(clf,f)
		#***pyplot.savefig('SVM_Plots/plot_db4energy_'+str(c)+'_'+str(g)+'_.png',bbox_inches='tight',dpi=600)

		# save
		#*** to be removed
		


		# load
		#with open('svm_model_split_spec.pkl', 'rb') as f:
		#    clf2 = pickle.load(f)

		#acc = clf.score(X_test, y_test)
		#print("acc=%0.3f" % acc)
