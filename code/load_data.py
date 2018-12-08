
import numpy as np
from numpy.random import shuffle
import torch
import glob
import json
import os

def readCSV(path,annotNb, seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)

    csv = np.genfromtxt(path,dtype="str")[1:]
    distorNbList = countNbDistor(csv)
    trainSet = csv[:,2:2+annotNb]

    return trainSet.astype(int),distorNbList
def countNbDistor(dataset):
    ''' Count the number of distorted videos for each reference video '''
    distorNbList = []
    currentRefInd = 1
    currentNbdistor = 0
    for elem in dataset:
        if int(elem[0]) != currentRefInd:
            distorNbList.append(currentNbdistor)
            currentNbdistor=1
            currentRefInd = int(elem[0])
        else:
            currentNbdistor += 1
    distorNbList.append(currentNbdistor)
    return distorNbList

def loadData(dataset,annotNb):
    ''' Build three dataloader : one for train, one for validation and one for test

    Args:
        dataset (string): the name of the dataset. Can only be \'IRCCYN\' for now.
        annotNb (int) : the number of annotator to use
    Returns:
        train_loader (torch.Tensor): the matrix for training
        test_loader (torch.Tensor): the matrix for testing

    '''

    if dataset  == "IRCCYN":
        trainSet,distorNbList = readCSV("../data/scores_irccyn.csv",annotNb)
    elif dataset  == "NETFLIX":
        trainSet,distorNbList = readCSV("../data/scores_netflix.csv",annotNb)
    else:
        raise ValueError("Unknown dataset",dataset)

    return torch.tensor(trainSet),distorNbList

if __name__ == '__main__':

    trainSet,distorNbList = loadData("IRCCYN",28)
