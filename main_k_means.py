# coding=utf-8

import sys
from LoadData import *
from k_means import *
from evaluation import *
from utils import *
import random

if __name__ == "__main__":

    # données numériques :
    corpus='test'
    dataFilename = 'data/self_test.data'
    groundtruthFilename = 'data/self_test.ground'

    # données IRIS :
    corpus='iris'
    dataFilename = 'data/iris.data'
    groundtruthFilename = 'data/iris.ground'

    data = loadPoints(dataFilename)
    groundtruth = loadClusters(groundtruthFilename)


    nDim = len(data[0])

    R = 3  # Nombre de clusters a priori

    centers = []
    for i in range(R):
        #centers.append(data[i])
        centers.append(data[random.randrange(len(data))])

    centresInitiaux = centers

    print('centres initiaux aléatoires : ', centers)

    results, centers = kmeans(data, centers, groundtruth, corpus)

    affichage_donnees(data, centers, centresInitiaux,results, corpus)

    res_Purity = purity(groundtruth, results, R)
    res_NMI = NMI(groundtruth, results, R)
    print("Purity = %.2f"% res_Purity)
    print("NMI = %.2f"% res_NMI)
