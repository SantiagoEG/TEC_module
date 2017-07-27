# -*- coding: utf-8 -*-
"""
TAILORED - ENSEMBLE CLASSIFIER


This module implements a version of a ensemble algorithm presented in [1]. In [1] DecisionTrees 
were employed as base estimators to improve Network Traffic Classification predictive models,
but the algorithm implementation allows to provide different base estimators (see test.py). Thus,
this algorithm is called TDTC that stands Tailored- DecisionTree Chain. Additionally, this module
implements some tools to speed-up the tuning process.

This algorithm builds a chain of classifiers in which each classifier acts as a sample filter for
its successor. When unknown samples are identified, the samples are output not reaching the sucessors
classifiers and leading to classification time savings with respect to other ensemble algorithms. 
Please, for more information about this algorithm read [1].

If you use this implementation for your research studies cite us:
[1] - Citar

Contact: santiago.egea@alumnos.uva.es
"""



import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import pandas as pd
from joblib import Parallel, delayed

"""
Standard global function.
"""
def fit_estimator(estimator, x, y):
    estimator.fit(x,y)
    return estimator

"""
This function returns several possibilities to tune the order of the base classifiers. An example
is provided in test.py
"""

def get_orders(x,y, clf_base, retraining = False):
    n_classes = len(np.unique(y))
    criterions = ['errors', 'f1', 'precision', 'recall', 'tn_rate', 'tp_rate',
                  'tp_ratextn_rate', 'tn_rate/tp_rate']
    order = np.zeros(shape = (len(criterions), n_classes))

    try:
        for criteria in criterions:
            x_aux = x.copy()
            y_aux = y.copy()
            cls = list(np.unique(y))
            n_classes = len(cls)
            comp = np.zeros(shape = (n_classes))    
            for j in range(len(cls)):
                
                """
                Base CLF
                """
                if j == 0 or retraining:
                    splitter = StratifiedShuffleSplit(y_aux, 1, test_size = 0.3, random_state = 0)
                    
                    for idx_train, idx_val in splitter:
                        x_train = x_aux[idx_train, :]
                        y_train = y_aux[idx_train]
                        x_val =  x_aux[idx_val, :]
                        y_val = y_aux[idx_val]
                        
                        clf_base.fit(x_train, y_train)
                        y_pred = clf_base.predict(x_val)
                cnfx_matrix_aux = confusion_matrix(y_val, y_pred)
                FP = np.zeros(shape = (n_classes,))
                FN = np.zeros(shape = (n_classes,))
                TP = np.zeros(shape = (n_classes,))
                TN = np.zeros(shape = (n_classes,))
                for i in range(n_classes):
                    comp[i] = np.sum(cnfx_matrix_aux[i, :])
                    FN[i] = np.sum(cnfx_matrix_aux[i, :]) - cnfx_matrix_aux[i, i]
                    FP[i] = np.sum(cnfx_matrix_aux[:, i]) - cnfx_matrix_aux[i, i]
                    TP[i] = cnfx_matrix_aux[i, i]
                    TN[i] = np.sum(comp)-np.sum(cnfx_matrix_aux[i,:])
                TP_rate = TP/(FP+TP)
                TN_rate = TN/(FN+TN)
    
            
                if criteria == 'errors':
                    """
                    %Errors
                    """  
                    rate = (FN+FP)/comp
                    to_del = cls[np.argmin(rate)]
                    order[0,j] = to_del
                    comp = np.delete(comp, cls.index(to_del), 0)
    
    
                elif criteria == 'f1':
                    """
                    F1-Score
                    """
                    rate = f1_score(y_val, y_pred, average = None) 
                    to_del = cls[np.argmax(rate)]
                    order[1,j] = to_del
                    
                elif criteria == 'precision':
                    """
                    Precision
                    """
                    rate = precision_score(y_val, y_pred, average = None) 
                    to_del = cls[np.argmax(rate)]
                    order[2,j] = to_del
                
                elif criteria == 'recall':
                    """
                    Recall
                    """
                    rate = recall_score(y_val, y_pred, average = None) 
                    to_del = cls[np.argmax(rate)]
                    order[3,j] = to_del
    
                elif criteria == 'tn_rate':
                    rate = TN_rate
                    to_del = cls[np.argmax(rate)]
                    order[4,j] = to_del            
    
                elif criteria == 'tp_rate':
                    rate = TP_rate
                    to_del = cls[np.argmax(rate)]
                    order[5,j] = to_del 
                        
                elif criteria == 'tp_ratextn_rate':
                    rate = (TP_rate*TN_rate)
                    to_del = cls[np.argmax(rate)]
                    order[6,j] = to_del 
                        
                elif criteria ==  'tn_rate/tp_rate':
                    rate = TP_rate/TN_rate
                    to_del = cls[np.argmax(rate)]
                    order[7,j] = to_del 
    
                cls.remove(to_del)
                n_classes -= 1
            
        
                if retraining:
                    idx_del = np.where(y_train == to_del)[0]
                    y_aux = np.delete(y_train, idx_del)
                    x_aux = np.delete(x_train, idx_del, 0) 
                else:
                    idx_del = np.where(y_pred == to_del)[0]
                    idx_del = np.hstack((idx_del, np.where(y_val == to_del)[0]))
                    idx_del = np.unique(idx_del)
                    y_pred = np.delete(y_pred, idx_del)
                    y_val = np.delete(y_val, idx_del)   
    except:
        print criteria
        raise
                
    return pd.DataFrame(order, index = criterions)

"""
This function returns the generate datasets for training and validation accoding to the provided
order.
"""

def get_DS(x, y, order):
    marker = -1
    x_list = []
    y_list = []
    for i in range(len(order)-1):            
        x_clf = x.copy()
        y_clf  = y.copy()
        y_clf[y_clf != order[i]] = marker
        x = x[y != order[i],:]
        y = y[y != order[i]]
            
        x_list.append(x_clf)
        y_list.append(y_clf)
    return x_list, y_list

"""
Class Tailored-Ensemble Classifier. In [1] we refered to this algorithm with the name TDCT, as we
employed DecissionTrees as base estimators. Actually, this implementation allows to use different
classifiers in each classification stage. See test.py to see detailed examples.
"""

class TEC(BaseEstimator, ClassifierMixin):    


    marker = -1
    def __init__(self, estimators, features_idx = 'all', order = None, n_jobs = 1):
        self.n_jobs = n_jobs
        self.estimators = []
        self.features_idx = []
        if order != None:
            self.order = order
        else:
            self.order = range(len(estimators))
        self.classes_ = list(self.order)
        self.classes_.sort()
        for i in self.order[:-1]:
            idx = self.classes_.index(i)
            self.estimators.append(estimators[idx])
            self.features_idx.append(features_idx[idx])
        self.n_estimators_ = len(self.estimators)
            
    def fit(self,x,y):
        self.classes_ = list(np.unique(y))
        if self.features_idx == 'all':
            self.features_idx = [range(x.shape[1])]*len(self.classes_)                        
        x_list = []
        y_list = []
        for i in range(self.n_estimators_):            
            x_clf = x[:,self.features_idx[i]].copy()
            y_clf  = y.copy()
            y_clf[y_clf != self.order[i]] = self.marker
            x = x[y != self.order[i],:]
            y = y[y != self.order[i]]
                
            x_list.append(x_clf)
            y_list.append(y_clf)
        
        self.estimators = Parallel(n_jobs = self.n_jobs)(delayed(fit_estimator)(self.estimators[i], 
                                   x_list[i], y_list[i]) for i in range(len(self.estimators)))
        return self
        
    def predict(self, x):
        y_pred = np.zeros(shape = (x.shape[0],), dtype = 'Int64')+self.marker   
        idx_next = np.arange(x.shape[0])
        for i in range(len(self.estimators)): 
                if idx_next.shape[0] > 0:
                    clf = self.estimators[i]
                    x_clf = x[:, self.features_idx[i]]
                    y_pred[idx_next] = clf.predict(x_clf[idx_next, :])    
                    idx_next = np.where(y_pred == self.marker)[0]  
                
        y_pred[y_pred == -1] = self.order[-1]
        return y_pred        








