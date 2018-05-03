#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:53:57 2016

@author: guangboyu
"""

import numpy as np
import scipy as sc
import csv
import random
from scipy.stats import multivariate_normal
import math
import matplotlib.pyplot as plt  


#def gaussian(x, mu, sigma, d):
#    dorm = (2 * np.pi) ** (d/2) * (sigma ** (1/2))
#    numer = np.exp(-1/2 * (x - mu).T * sigma * (x - mu))
def Q_function(k, dataSet, priors, omega, means, covs):
    num = priors.shape[0]
    dim = priors.shape[1]
    Q = 0
    for i in range(num):
        for j in range(k):
            N = multivariate_normal.pdf(dataSet[i], mean = means[j],cov = covs[j,:,:], allow_singular=True)
            Q += priors[i, j] * (np.log(omega[j]) + np.log(N))
    return Q
    
def init(dataSet, k):
    dim = dataSet.shape[1]
    priors = np.zeros(k)
    means = np.zeros((k, dim))
    covs = np.zeros((k, dim, dim))
    for i in range(k):
        priors[i] = 1/float(k)

    for i in range(k):
        index = random.randint(0, k)
        means[i, :] = dataSet[index, :]

    for i in range(k):
        A = sc.random.rand(2, 2)
        B = np.dot(A, A.T)
        covs[i, :, :] = np.array(B)
    return priors, means, covs
    
#output is n*k    
def e_step(dataSet, k, means, covs, omega):
    num = dataSet.shape[0]
    dim = dataSet.shape[1]
    priors_numer = np.zeros((num, k))
    for i in range(k):
        priors_numer[:,i] = multivariate_normal.pdf(dataSet, mean = means[i], 
        cov = covs[i,:,:], allow_singular=True)* omega[i]
    priors_dorm = np.sum(priors_numer, axis = 1) 
    priors = priors_numer / np.tile(priors_dorm, (k, 1)).T
    return priors
    
    
def m_step(dataSet, k, priors):
    num = dataSet.shape[0]
    dim = dataSet.shape[1]
    #omega
    omega = np.zeros(k)
    for i in range(k):
        omega[i] = np.sum(priors[:, i]) / num
    #means
    means = np.zeros((k, dim))
    for i in range(k):
        means_num = np.zeros((num, dim))
        for j in range(num):
            means_num[j,:] = priors[j, i] * dataSet[j]
        means_num = np.sum(means_num, axis = 0)  
        means_dorm = np.sum(priors[:, i])
        means[i, :] = means_num / means_dorm
    #covs
    covs = np.zeros((k, dim, dim))
    for i in range(k):
        covs_num = np.zeros((num, dim, dim))
        for j in range(num):
            diff = dataSet[j,:] - means[i,: ]
            diff = diff.reshape(dim, 1)
            covs_num[j,:,: ] = priors[j, i] * np.dot(diff, diff.T)
        covs_num = np.sum(covs_num, axis = 0)
        covs_dorm = np.sum(priors[:, i], 0)
        covs[i,:,:] = covs_num / covs_dorm
    return omega, means, covs


def log_plot(Q):
    num = Q.shape[0]
    for i in range(num):
        plt.plot(i, Q[i], '.')
    
        
        
def cluster(dataSet, priors):
    num = priors.shape[0]
    clusters = np.zeros(num)
    for i in range(num):
        clusters[i] = np.argmax(priors[i, :])
    return clusters
    

def plotCluster(dataSet, k, clusters):  
    num = dataSet.shape[0]
    mark = ['.', 'o', '^', 'd', 'x']  
    colors = ['r', 'g', 'c', 'b', 'm']     
    for i in range(num):  
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[int(clusters[i])], color = colors[int(clusters[i])])    
    plt.show()      
    
    
def em(dataSet, k, iter):

    omega, means, covs = init(dataSet, k)
    Q = np.zeros(iter)
    for p in range(iter):
        #E-step:
        priors = e_step(dataSet, k, means, covs, omega)
        #M-step:
        omega, means, covs = m_step(dataSet, k, priors)
        q = Q_function(k, dataSet, priors, omega, means, covs)
        Q[p] = q
        if math.isnan(Q[p]) and p > 0:
            break
        if p > 0 and (Q[p] - Q[p-1]) < 1e-10:
            for i in range(p+1, iter):
                Q[i] = Q[p]
            break
    return Q, priors, omega, means, covs
        



def p4():
    dataSet = []  
    fileIn = open('hw5_blob.csv') 
    reader = csv.reader(fileIn)
    for i in reader:  
        dataSet.append([float(i[0]), float(i[1])])
    dataSet = np.array(dataSet)
    k = 3
    iter = 100
    i = 0
    while i < 5:
        Q, priors, omega, means, covs = em(dataSet, k, iter)
        if math.isnan(Q[1]):
            continue
        log_plot(Q)
        i += 1
        print('means')
        print(means)
        print('covs')
        print(covs)
    plt.show()

   
        
def p5():
    dataSet = []  
    fileIn = open('hw5_blob.csv') 
    reader = csv.reader(fileIn)
    for i in reader:  
        dataSet.append([float(i[0]), float(i[1])])
    dataSet = np.array(dataSet)
    k = 3
    iter = 100
    i = 0
    tag = True
    while tag:
        tag = False
        Q, priors, omega, means, covs = em(dataSet, k, iter)
        if math.isnan(Q[1]):
            tag = True
            continue
        if Q[60] < -120:
            tag = True
            continue
        i += 1
        clusters = cluster(dataSet, priors)
        plotCluster(dataSet, k, clusters)
        print('means')
        print(means)
        print('covs')
        print(covs)
    
    
    
def main():
    p4()
    p5()
    
    
if __name__ == '__main__':
    main()