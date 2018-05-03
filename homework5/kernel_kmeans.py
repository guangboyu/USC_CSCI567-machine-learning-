#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:30:34 2016

@author: guangboyu
"""

import numpy as np
import csv
import matplotlib.pyplot as plt  
import random


def RBF(dataSet, sigma):
    num = dataSet.shape[0]
    K = np.zeros((num, num))
    for i in range(num):
        for j in range(i, num):
            if i != j:
                numer = -np.sum(np.power(dataSet[i, :] - dataSet[j, :], 2))
                K[i, j] = np.exp(numer/(2*(sigma**2)))
                K[j, i] = K[i, j]
    return K
    
def Poly(dataSet, c, d):
    num = dataSet.shape[0]
    K = np.zeros((num, num))
    for i in range(num):
        for j in range(i, num):
            if i != j:                
                K[i, j] = (np.dot(dataSet[i, :], dataSet[j, :].T) + c)**d
                K[j, i] = K[i, j]
    return K
    

def compute_k(x1, x2, sigma):
    numer = -np.sum(np.power(x1 - x2, 2))
    result = np.exp(numer/(2*(sigma**2)))
    return result
    

def kernel_kmeans(dataSet, k):
    sigma = 0.17
    converged = False
    num = dataSet.shape[0]
    clusters = kernel_init(dataSet, k)
    
    for p in range(20):
#    while not converged:
        converged = True
        
#        K = RBF(dataSet, sigma)
        K = Poly(dataSet, 0, 8)
        clusters_sum = np.zeros(k)
        for i in range(k):
            for j in range(num):
                if clusters[j] == i:
                    clusters_sum[i] += 1
        
        #compute equation 3
        equation_3 = np.zeros(k)
        for i in range(k):
            for j in range(num):
                if clusters[j] == i:
                    for m in range(j, num):
                        if clusters[m] == i:
#                            equation_3[i] += compute_k(dataSet[j, :], dataSet[m, :], sigma)
                            equation_3[i] += K[j, m]
        
        #i: the point # want to assign, j: # of clusters, m: the point in cluster j   
        for i in range(num):
            min_distance = 1e10
            min_index = -1
            for j in range(k):
                nk = clusters_sum[j]
                sum_1 = 0
                for m in range(num):
                    if clusters[m] == j:
#                        sum_1 += compute_k(dataSet[i, :], dataSet[m, :], sigma)
                        sum_1 += K[i, m]
                    
                distance = -((2/nk) * sum_1) + (1/(nk ** 2)) * equation_3[j]
                if distance < min_distance:
                    min_distance = distance
                    min_index = j
                    
                if clusters[i] != min_index:
                    converged = False
                    clusters[i] = min_index           
    return clusters
    
    
def kernel_init(dataSet, k):
    num = dataSet.shape[0]
    clusters = np.zeros(num)
    for i in range(num):
        clusters[i] = random.randint(0,k-1)
    return clusters
    
    
    
def plotCluster(dataSet, k, clusters):  
    num = dataSet.shape[0]
    dim = dataSet.shape[1]      
    mark = ['.', 'o', '^', 'd', 'x']  
    colors = ['r', 'g', 'c', 'b', 'm']     
    for i in range(num):  
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[int(clusters[i])], color = colors[int(clusters[i])])    
    plt.show()    
    
    
def p3():
    dataSet = []  
    fileIn = open('hw5_circle.csv') 
    reader = csv.reader(fileIn)
    for i in reader:  
        dataSet.append([float(i[0]), float(i[1])])
    dataSet = np.array(dataSet)
    k = 2
    clusters = kernel_kmeans(dataSet, k)
    plotCluster(dataSet, k, clusters)
                

def main():
    for i in range(3):
        p3()
    
    
if __name__ == '__main__':
    main()