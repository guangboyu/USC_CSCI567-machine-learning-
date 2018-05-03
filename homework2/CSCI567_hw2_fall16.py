# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
import math

def plothist(x):
    d = x.shape[1]
    print("histograms of all numerical attributes")
    for i in range(d):
        plt.hist(x[:,i],bins = 10)
        plt.title("the %dth feature" %(i+1))
        plt.show()

def crossValidation(x,y,numVal,lamb):
    m = len(y)
    #indexList = range(m)
    wMat = []
    
    for i in range(numVal):
        trainX= []; trainY=[]
        testX = []; testY = []
        #np.random.shuffle(indexList)
        for j in range(m):
            if j % 10 == i:
                testX.append(x[j])
                testY.append(y[j]) 
            else:
                trainX.append(x[j])
                trainY.append(y[j])
        
        testX = np.array(testX)
        k = testX.shape[0]

        omega = ridge_lr(trainX,trainY,lamb)
        output = predict(testX, omega)
        rlr_mse = np.sum([(testY[i] - output[i]) ** 2 for i in range(k)]) / k
        wMat.append(rlr_mse)
    return np.mean(wMat)

def multiply(a,b):
    sum_ab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sum_ab+=temp
    return sum_ab

def cal_pearson(x,y):
    n=len(x)
    sum_x=sum(x)
    sum_y=sum(y)
    sum_xy=multiply(x,y)
    sum_x2 = sum([i ** 2 for i in x])
    sum_y2 = sum([j ** 2 for j in y])
    molecular=sum_xy-(float(sum_x)*float(sum_y)/n)
    denominator=np.sqrt((sum_x2-float(sum_x**2)/n)*(sum_y2-float(sum_y**2)/n))
    return molecular/denominator
    
def normalize(x):
    mean = np.mean(x, axis = 0)
    std = np.std(x, axis = 0)
    normalized = (x - mean) / std
    return normalized
    
def ridge_lr(x, y, lamb):
    x = np.array(x)
    y = np.array(y)
    n = x.shape[0]
    d = x.shape[1]              
    x = np.append(np.ones([n,1]),x,axis = 1)
    eye = np.identity(x.shape[1])
    xTx = np.dot(np.transpose(x), x)
    z = np.linalg.pinv(lamb * eye + xTx)
    t = np.dot(z, np.transpose(x))
    omega = np.dot(t, y)
    omega.shape = (1,d+1)
    omega = np.transpose(omega)
    return omega

def normal_equation(x, y):
    n = x.shape[0]
    d = x.shape[1]              
    x = np.append(np.ones([n,1]),x,axis = 1)
    
    z = np.linalg.pinv(np.dot(np.transpose(x), x))
    omega = np.dot(np.dot(z, np.transpose(x)), y)
    
    omega.shape = (1,d+1)
    omega = np.transpose(omega)
    
    return omega
    
def predict(x, w):
    m = x.shape[0]
    x = np.append(np.ones([m, 1]), x, axis = 1)
    x = np.array(x)    
    predict = np.dot(x, w)
    return predict
            
def main(): 
    #process data       
    boston = load_boston()
    test_data = boston.data[::7, :]
    test_target = boston.target[::7]
    train_data = []
    train_target= []    
    for i in range(0, len(boston.data)):
        if i%7 != 0:
            train_data.append(boston.data[i])
            train_target.append(boston.target[i])           
    train_target = np.array(train_target)
    train_data = np.array(train_data)
    d = train_data.shape[1]
    m = test_data.shape[0]
    
    plothist(train_data)
    #cor
    pearson = []
    for i in range(d):
        c = cal_pearson(train_data[:,i], train_target)
        pearson.append(c)    
    plt.figure(figsize=(9,6))
    plt.title("pearson correlation without aboslute value")
    X = np.arange(d)+1
    plt.bar(X, np.abs(pearson), width = 0.35,facecolor = 'black',edgecolor = 'white')
    print("pearson correlation with absolute value")
    pearson = np.array(pearson)
    pearson.reshape((d,1))
    print(pearson)
    print("\n")
    
    
    #normalize data
    train_data = normalize(train_data)
    test_data = normalize(test_data)

    
    #compute omega for lr
    omega = normal_equation(train_data,train_target)
    #predict test for lr
    lr_output = predict(test_data, omega)
    #lr mse
    lr_mse = np.sum([(test_target[i] - lr_output[i]) ** 2 for i in range(m)]) / m
    print("linear regression's MSE")
    print(lr_mse)
    print("\n")

    
    #compute mse for rlr
    for lamb in [0.01, 0.1, 1]:        
        print("lambda = %.2f, ridge regression's MSE" %lamb)
        omega_r = ridge_lr(train_data, train_target, lamb)
        rlr_output = predict(test_data, omega_r)
        rlr_mse = np.sum([(test_target[i] - rlr_output[i]) ** 2 for i in range(m)]) / m
        print(rlr_mse)
        
    print("\n")
    #compute CV
    for lamb in [0.0001, 0.001, 0.01, 0.1, 1, 10]:
        print("lambda = %.4f, cross validation's value" %lamb)
        cv = crossValidation(train_data,train_target,10,lamb)
        print(cv)
    print("\n")
    
    #Mutual info
    print("3, 6, 11, 13 has the highest correlation")
    print("Mutual infomation's MSE")
    M_train = train_data[:, [2, 5, 10, 12]]
    M_test = test_data[:, [2, 5, 10, 12]]
    omega_m = normal_equation(M_train, train_target)
    m_output = predict(M_test, omega_m)
    m_mse = np.sum([(test_target[i] - m_output[i]) ** 2 for i in range(m)]) / m
    print(m_mse)
    print("\n")
    
    #Brute-force search
    B_mse = []
    for q in range(d):
        for w in range(q):
            for e in range(w):
                for r in range(e):
                    B_train = train_data[:, [q,w,e,r]]
                    B_test = test_data[:, [q,w,e,r]]
                    B_omega = normal_equation(B_train, train_target)
                    B_output = predict(B_test, B_omega)
                    mse = np.sum([(test_target[i] - B_output[i]) ** 2 for i in range(m)]) / m
                    B_mse.append(mse)
    print("Selection with Brute-force search")
    print(np.min(B_mse))
    print("\n")
    
      #poly  
    P_train = train_data
    P_test = test_data
    for i in range(d):
        for j in range(i+1):
            P_train = np.append(P_train, train_data[:, [i]]*train_data[:, [j]],axis = 1)
            P_test = np.append(P_test, test_data[:, [i]]*test_data[:, [j]],axis = 1)
    P_train = normalize(P_train)
    P_test = normalize(P_test)
    omega_p = normal_equation(P_train, train_target)
    p_output = predict(P_test, omega_p)
    p_mse = np.sum([(test_target[i] - p_output[i]) ** 2 for i in range(m)]) / m
    print("3.4 Polynomial feature expansion's MSE with normalization")
    print(p_mse)
    print("\n")

                    
       
if __name__ == "__main__":
	main()

