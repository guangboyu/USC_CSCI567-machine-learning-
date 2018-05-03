# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 09:46:06 2016

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
    
    #linear
    print("linear")
    for i in range(9):
        ep_c = i - 6
        c = 4**(ep_c)
        print('c = %d'%(ep_c))
        start = time.clock()
        m = svm_train(y, train, '-v 3 -t 0 -c %f'%c)
        end = time.clock()
        print(end - start)
    print("\n")
    
    
    #polynomial
    print("poly")
    for j in range(3):
        d = j+1
        print("degree = %d"%d)
        for i in range(11):
            ep_c = i - 3
            c = 4**(ep_c)
            print("c = %d, degree = %d"%(ep_c,d))
            start = time.clock()
            m = svm_train(y, train, '-d %d -t 1 -v 3 -c %f'%(d,c))
            end = time.clock()
            print(end-start)
        print('\n')
    print('\n')

    
    ##RBF
    print("RBF")
    for i in range(11):
        ep_c = i-3
        c = 4**(ep_c)
        for j in range(7):
            ep_g = j-7
            g = 4**(ep_g)
            print("c = %d, gamma = %d:"%(ep_c,ep_g))
            start = time.clock()
            m = svm_train(y, train, '-g %f -t 2 -v 3 -c %f'%(g,c))
            end = time.clock()
            print(end-start)
        print('\n')
        
#    print('\n')
        

    
if __name__ == "__main__":
	main()
    
