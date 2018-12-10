import sys
from args import ArgReader
from args import str2bool
import load_data
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from torch.autograd import Variable
import modelBuilder
import os
import configparser
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import grad
import random

def addLossTerms(loss,model,args):

    hessDiagList,_ = computeHessDiagList(loss,model)

    hessDiagList[2] = hessDiagList[2].view(-1)

    if args.scnd_order_weight > 0:
        loss -= args.scnd_order_weight*torch.cat(hessDiagList,dim=0).sum()

    if args.prior_annot_incons > 0:

        loss += args.prior_annot_incons*torch.pow((model.annot_incons-0.5*torch.ones(model.annot_incons.size())),2).sum()

    return loss

def one_epoch_train(model,optimizer,trainMatrix, epoch, args,lr):
    '''Train a model

    After having run the model on every input of the train set,
    its state is saved in the models/NameOfTheExperience/ folder

    Args:
        model (MLE): a MLE module (as defined in modelBuilder)
        optimizer (torch.optim): the optimizer to train the model
        trainMatrix (torch.tensor): the matrix train
        epoch (int): the current epoch number
        args (Namespace): the namespace containing all the arguments required for training and building the model
    '''
    train_loss = 0

    model.train()

    data = Variable(trainMatrix)

    neg_log_proba = model(data)

    loss = neg_log_proba

    loss = addLossTerms(loss,model,args)

    loss.backward(retain_graph=True)

    if args.optim == "LBFGS":

        def closure():
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            return loss
        optimizer.step(closure)
        optimizer.zero_grad()
    elif args.optim == "NEWTON":

        hessDiagList,gradientList = computeHessDiagList(loss,model)

        annot_bias = model.annot_bias-lr*gradientList[0]/hessDiagList[0]
        annot_incons = model.annot_incons-lr*gradientList[1]/hessDiagList[1]
        video_amb = model.video_amb-lr*gradientList[2]/hessDiagList[2]
        video_scor = model.video_scor-lr*gradientList[3]/hessDiagList[3]

        model.setParams(annot_bias,annot_incons,video_amb,video_scor)

    else:

        optimizer.step()
        optimizer.zero_grad()

    return loss
def writeCSV(args,epoch, log_proba):
    '''Write the loss of a model in a csv file

    Every time a batch is processed during an epoch, this function is called and the results of the batch processing
    printed in the csv file, just after the results from the last batch.

    Args:
        args (Namespace): the namespace containing all the arguments required for training and building the model
        epoch (int): the epoch number
        log_proba (list): the loss of the batch
        phase (str): indicates the phase the model is currently in (can be \'train\', \'validation\' or \'test\')
    '''

    filePath = "../results/"+str(args.exp_id)+"/all_scores_model"+str(args.ind_id)+"_epoch"+str(epoch)+"_train.csv"

    #Writes the log probability of the loss in a csv file
    with open(filePath, "w") as text_file:
        print("#loss",file=text_file)
        print(str(log_proba),end="",file=text_file)

def computeHessDiag(loss,tensor):

    gradient = grad(loss, tensor, create_graph=True)
    gradient = gradient[0]

    hessDiag = torch.zeros_like(tensor)
    for i,first_deriv in enumerate(gradient):
        hessDiag[i] = grad(first_deriv, tensor, create_graph=True)[0][i]
    return hessDiag,gradient

def computeHessDiagList(loss, model):

    hessDiagList = []
    gradientList = []
    for tensor in model.parameters():
        #print("Computing second order derivative for ",tensor.size())

        hessDiag,gradient = computeHessDiag(loss,tensor)
        hessDiagList.append(hessDiag)
        gradientList.append(gradient)

    return hessDiagList,gradientList

def get_OptimConstructor(optimStr,momentum):
    '''Return the apropriate constructor and keyword dictionnary for the choosen optimiser
    Args:
        optimStr (str): the name of the optimiser. Can only be \'GD\', \'LBFGS\' or \'NEWTON\'.
    Returns:
        the constructor of the choosen optimiser and the apropriate keyword dictionnary
    '''

    if optimStr == "GD":
        optimConst = optim.SGD
        kwargs= {'momentum': momentum}

    elif optimStr == "LBFGS":
        optimConst = optim.LBFGS
        kwargs = {"history_size":1000,"max_iter":40}

    elif optimStr=="NEWTON":
        optimConst = None
        kwargs = {}
    else:
        raise ValueError("Unknown optimisation algorithm : {}.".format(args.optim))

    print("Optim is :",optimConst)

    return optimConst,kwargs

def train(model,optimConst,kwargs,trainSet, args):

    #Train and evaluate the model for several epochs
    #print(trainSet)
    model.initParams(trainSet)

    epoch = 1
    dist = args.stop_crit+1
    lrCounter = 0
    while epoch < args.epochs and dist > args.stop_crit:

        if epoch%400==0:
            print("\tEpoch : ",epoch,"dist",float(dist))

        #This condition determines when the learning rate should be updated (to follow the learning rate schedule)
        #The optimiser have to be rebuilt every time the learning rate is updated
        if (epoch-1) % ((args.epochs + 1)//len(args.lr)) == 0 or epoch==1:

            kwargs['lr'] = args.lr[lrCounter]
            #print("Learning rate : ",kwargs['lr'])
            if optimConst:
                optimizer = optimConst(model.parameters(), **kwargs)
            else:
                optimizer = None
            if lrCounter<len(args.lr)-1:
                lrCounter += 1

        old_score = model.state_dict()["video_scor"].clone()
        loss = one_epoch_train(model,optimizer,trainSet,epoch, args,args.lr[lrCounter])


        dist = torch.sqrt(torch.pow(old_score-model.state_dict()["video_scor"],2).sum())
        epoch += 1

    print("\tStopped at epoch ",epoch)
    return loss,epoch

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)


    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    torch.manual_seed(args.seed)
    random.seed(args.seed)


    #The group of class to detect
    np.random.seed(args.seed)

    #The folders where the experience file will be written
    if not (os.path.exists("../vis/{}".format(args.exp_id))):
        os.makedirs("../vis/{}".format(args.exp_id))
    if not (os.path.exists("../results/{}".format(args.exp_id))):
        os.makedirs("../results/{}".format(args.exp_id))
    if not (os.path.exists("../models/{}".format(args.exp_id))):
        os.makedirs("../models/{}".format(args.exp_id))

    #Write the arguments in a config file so the experiment can be re-run
    argreader.writeConfigFile("../models/{}/model{}.ini".format(args.exp_id,args.ind_id))

    #Loading data
    trainSet,distorNbList = load_data.loadData(args.dataset)

    if args.cuda:
        trainSet = trainSet.cuda()

    #Building the model
    model = modelBuilder.modelMaker(trainSet.size(1),len(trainSet),distorNbList,args.poly_deg)
    if args.cuda:
        model = model.cuda()
    torch.save(model.state_dict(), "../models/{}/model{}_epoch0".format(args.exp_id,args.ind_id))

    #Getting the contructor and the kwargs for the choosen optimizer
    optimConst,kwargs = get_OptimConstructor(args.optim,args.momentum)

    #If no learning rate is schedule is indicated (i.e. there's only one learning rate),
    #the args.lr argument will be a float and not a float list.
    #Converting it to a list with one element makes the rest of processing easier
    if type(args.lr) is float:
        args.lr = [args.lr]

    loss,epoch = train(model,optimConst,kwargs,trainSet, args)
    #print(model.video_amb)

    torch.save(model.state_dict(), "../models/{}/model{}_epoch{}".format(args.exp_id,args.ind_id,epoch))

if __name__ == "__main__":
    main()
