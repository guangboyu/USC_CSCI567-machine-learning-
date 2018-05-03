#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 15:41:47 2016

@author: guangboyu
"""

import scipy
import numpy as np
from svmutil import *
import time

def process(feature):
    output = []
    for i in feature:
        Row = []
        for j in range(30):
            if j in {1, 6, 7, 13, 14, 25, 28}:
                if i[j] == -1:
                    Row += [1,0,0]
                elif i[j] == 0:
                    Row += [0,1,0]
                else:
                    Row += [0,0,1]
            else:
                if i[j] == -1:
                    Row.append(0)
                else:
                    Row.append(1)
        output.append(Row)
    return output
    
def main():
    train_data = scipy.io.loadmat("phishing-train.mat")
    test_data = scipy.io.loadmat("phishing-test.mat")
    
    train_feature = train_data['features']
    train_label = train_data['label']
    test_feature = test_data['features']
    test_label = test_data['label']
    

    train_n = train_feature.shape[0]
    test_n = test_feature.shape[0]
    train_label = list(train_label)


    train_feature = process(train_feature)
    test_feature = process(test_feature)
    
    y = []
    for i in range(train_n):
        y.append(train_label[0][i])
    test_y = []
    for i in range(test_n):
        test_y.append(test_label[0][i])
        
    train = []
    for i in range(len(train_feature)):
        x = dict()
        for j in range(len(train_feature[1])):
            val = train_feature[i][j]
            index = j+1
            x[index] = val
        train.append(x)
    test = []
    for i in range(len(test_feature)):
        x = dict()
        for j in range(len(test_feature[1])):
            val = test_feature[i][j]
            index = j+1
            x[index] = val
        test.append(x)
    g = 4**(-1)
    c = 4**(7)
    m = svm_train(y, train, '-g %f -t 2 -c %f'%(g,c))
    p = svm_predict(test_y,test,m)
    
if __name__ == "__main__":
	main()
    
