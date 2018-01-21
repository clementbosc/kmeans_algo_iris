# coding=utf-8
# from evaluation import *
from scipy.spatial import distance
import numpy as np


def squaredDistance(vec1, vec2):
    """retourne la distance quadratique entre vec1 et vec2"""
    return (distance.euclidean(vec1, vec2))**2



def computeSSE(data, centers, clusterID):
    """retourne la Sum of Squarred Errors entre les points et le centre de leur cluster associe"""
    sse = 0
    nData = len(data)
    for i in range(nData):
        c = clusterID[i]
        sse += squaredDistance(data[i], centers[c]) ** 2
    return sse


def updateClusterID(data, centers):
    """attribue un indice de cluster à chaque point de data"""

    nData = len(data)
    nCenter = len(centers)
    clusterID = [0] * nData
    for i in range(nData):
        l = []
        for j in range(nCenter):
            l.append(squaredDistance(data[i], centers[j]))
        clusterID[i] = np.argmin(l)

    return clusterID


def updateCenters(data, clusterID, R):
    """recalcule les coordonnées des centres des clusters"""

    nDim = len(data[0])
    centers = [[0] * nDim for i in range(R)]

    for i in range(R):
        nuageI = []
        for j in range(len(clusterID)):
            if(clusterID[j] == i):
                nuageI.append(data[j])

        for j in range(len(nuageI)): #pour chaque element du nuage i
            for k in range(nDim):
                centers[i][k] += nuageI[j][k]
        for k in range(nDim):
            centers[i][k] = centers[i][k]/len(nuageI)

    return centers


def kmeans(data, centers, groundtruth, corpus, maxIter = 100, tol = 1e-6):
    nData = len(data)

    if nData == 0:
        return [], []

    R = len(centers)

    clusterID = [0] * nData

    if R >= nData:
        for i in range(nData):
            clusterID[i] = i
        return clusterID, centers

    stabilise = False
    iter=0

    listCurDistance = []
    while not stabilise and iter < maxIter:

        clusterID = updateClusterID(data, centers)

        centers = updateCenters(data, clusterID, R)

        listCurDistance.append(computeSSE(data, centers, clusterID))

        if listCurDistance[iter] is not None:
            print ("# it %d SSE = %.2f" %(iter, listCurDistance[iter]))

        if(iter > 1):
            stabilise = (listCurDistance[iter-1] - listCurDistance[iter]) < tol


        iter += 1

    print ("# of iterations: %d"% iter)
    if listCurDistance[iter-1] is not None:
        print ("SSE = %.2f"% listCurDistance[iter-1])
    return clusterID, centers
