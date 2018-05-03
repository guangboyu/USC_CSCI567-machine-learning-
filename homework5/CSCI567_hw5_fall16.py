#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 22:51:05 2016

@author: guangboyu
"""

import numpy as np
import csv
import matplotlib.pyplot as plt  
import random
import kernel_kmeans
import hw5_em


def compute_distance(v1, v2):
    distance = np.sqrt(np.sum(np.power(v1 - v2, 2)))
    return distance
    
    
def init(dataSet, k):
    num = dataSet.shape[0]
    dim = dataSet.shape[1]
    centroids = np.zeros((k, dim))
    for i in range(k):
        index = random.randint(0, num)
        centroids[i, :] = dataSet[index, :]
    return centroids
    

def kmeans(dataSet, k):
    num = dataSet.shape[0]
    dim = dataSet.shape[1]
    centroids = init(dataSet, k)
    converged = False
    clusters = np.zeros(num)
    while not converged:
        converged = True
        #compute distance and closet centroid
        for i in range(num):
            min_distance = 1e10
            min_centroids = 0
            #compute each centroid and closet one
            for j in range(k):
                distance = compute_distance(centroids[j, :], dataSet[i, :])
                if distance < min_distance:
                    min_distance = distance
                    min_centroids = j
            
            #reassign
            if clusters[i] != min_centroids:
                converged = False
                clusters[i] = min_centroids
        
        #change centroids
        centroids = np.zeros((k, dim))
        #loop cluser
        for i in range(k):
            #loop row
            for d in range(dim):
                sum_coordinate = 0
                cluster_num = 0
                #loop dimension
                for r in range(num):
                    if clusters[r] == i:
                        sum_coordinate += dataSet[r, d]
                        cluster_num += 1
                centroids[i, d] = sum_coordinate/cluster_num

    return centroids, clusters
            

def plotCluster(dataSet, k, centroids, clusters):  
    num = dataSet.shape[0]
    dim = dataSet.shape[1]      
    mark = ['.', 'o', '^', 'd', 'x']  
    colors = ['r', 'g', 'c', 'b', 'm']     
    for i in range(num):  
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[int(clusters[i])], color = colors[int(clusters[i])])    
    plt.show()            
            

def p1():
    dataSet = []  
    fileIn = open('hw5_blob.csv') 
    reader = csv.reader(fileIn)
    for i in reader:  
        dataSet.append([float(i[0]), float(i[1])])
    dataSet = np.array(dataSet)
    for k in (2, 3, 5):
        centroids, clusters = kmeans(dataSet, k)
        plotCluster(dataSet, k, centroids, clusters)
 
    
def p2():
    dataSet = []  
    fileIn = open('hw5_circle.csv') 
    reader = csv.reader(fileIn)
    for i in reader:  
        dataSet.append([float(i[0]), float(i[1])])
    dataSet = np.array(dataSet)
    for k in (2, 3, 5):
        centroids, clusters = kmeans(dataSet, k)
        plotCluster(dataSet, k, centroids, clusters)
    

        

def main():
    p1()
    p2()
    kernel_kmeans.main()
    hw5_em.main()
    
    
if __name__ == '__main__':
    main()
    