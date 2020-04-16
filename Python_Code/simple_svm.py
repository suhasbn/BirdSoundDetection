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
from matplotlib import pyplot
import pickle


# Load data from numpy file
X = np.load('db2.npy',allow_pickle = True)
X = X[::,1::30] 
#X = X[::,1:10:1]
#X_2 = np.load('db4energy.npy')
#X_2 = X_2[::,1:6:1] 
#X_3 = np.load('feat_mel.npy')
#X = np.concatenate((X_1,X_2),axis=1)
y =  np.load('labels.npy').ravel()
# Split data into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)


# Simple SVM
print('fitting...')
clf = SVC(C=0.03, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
clf.fit(X_train, y_train)


lr_probs = clf.predict(X_test)
ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('SVM_db2: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill AUC: 0.5')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='SVM_db2 AUC=%.3f' % (lr_auc))
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
#pyplot.show()
pyplot.savefig('plot_db2.png',bbox_inches='tight',dpi=600)

# save
#with open('svm_model_Apr1_sym5.pkl','wb') as f:
#    pickle.dump(clf,f)


# load
#with open('svm_model_split_mfcc.pkl', 'rb') as f:
#    clf2 = pickle.load(f)

#acc = clf.score(X_test, y_test)
#print("acc=%0.3f" % acc)

'''
# Grid search for best parameters
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5],
                      'C': [1, 10 ,20,30,40,50]}], [{'kernel': ['linear'],'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print('')

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print('')
    print(clf.best_params_)
    print('')
    print("Grid scores on development set:")
    print('')
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
       print("%0.3f (+/-%0.03f) for %r"
             % (mean, std * 2, params))
    print('')

    print("Detailed classification report:")
    print('')
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print('')
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print('')
'''
