# coding=utf-8
from math import log, sqrt
from sklearn import metrics


def purity(groundtruthAssignment, algorithmAssignment, R):
    '''calcule la pureté d'une partition'''
    sum = 0
    for i in range(len(algorithmAssignment)):
        if algorithmAssignment[i] == groundtruthAssignment[i]:
            sum += 1

    return sum/float(len(algorithmAssignment))


def NMI(groundtruthAssignment, algorithmAssignment, R):
    return metrics.normalized_mutual_info_score(algorithmAssignment, groundtruthAssignment)

