
import numpy as np
from numpy.random import shuffle
import torch
import glob
import json
import os

def readCSV(path, seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)

    csv = np.genfromtxt(path,dtype="str")[1:]
    distorNbList = countNbDistor(csv)
    trainSet = csv[:,2:]

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

def loadData(dataset):
    ''' Build three dataloader : one for train, one for validation and one for test

    Args:
        dataset (string): the name of the dataset. Can be \'IRCCYN\', \'NETFLIX\' or \'VQEG\'.
    Returns:
        train_loader (torch.Tensor): the matrix for training
    '''

    if dataset  == "IRCCYN":
        trainSet,distorNbList = readCSV("../data/scores_irccyn.csv")
    elif dataset  == "NETFLIX":
        trainSet,distorNbList = readCSV("../data/scores_netflix.csv")
    elif dataset == "VQEG":
        trainSet,distorNbList = readCSV("../data/scores_vqeg.csv")
    else:
        raise ValueError("Unknown dataset",dataset)

    return torch.tensor(trainSet),distorNbList

if __name__ == '__main__':

    trainSet,distorNbList = loadData("IRCCYN",28)
