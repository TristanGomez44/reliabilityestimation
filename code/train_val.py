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
import processResults
import generateData

from torch.distributions.beta import Beta

def paramsToCsv(loss,model,exp_id,ind_id,epoch,scoresDis,score_min,score_max):

    confInterList = processResults.computeConfInter(loss,model)
    keys = list(model.state_dict().keys())

    for i,tensor in enumerate(model.parameters()):


        if keys[i] == "diffs" or keys[i] == "incons":
            tensor = torch.sigmoid(tensor)

        #if keys[i] ==  "trueScores":
        #    tensor = generateData.betaDenormalize(tensor,score_min,score_max)
        #    tensor = torch.clamp(tensor,score_min,score_max)

        tensor = tensor.cpu().detach().numpy().reshape(-1)[:,np.newaxis]

        confInterv = confInterList[i].cpu().detach().numpy().reshape(-1)[:,np.newaxis]
        concat = np.concatenate((tensor,confInterv),axis=1)
        np.savetxt("../results/{}/model{}_epoch{}_{}.csv".format(exp_id,ind_id,epoch,keys[i]),concat,delimiter="\t")

def addLossTerms(loss,model,weight,normSum,cuda):
    #print("Adding loss terms")
    #print(loss)
    if weight>0:
        loss -= weight*model.prior(loss)
        #print(loss)
    if normSum:

        labels = torch.arange(1,6).unsqueeze(0).unsqueeze(0)

        amb_incon = model.ambInconsMatrix(cuda)
        amb_incon = amb_incon.unsqueeze(2).expand(amb_incon.size(0),amb_incon.size(1),labels.size(2))

        scor_bias = model.trueScoresBiasMatrix()
        scor_bias = scor_bias.unsqueeze(2).expand(scor_bias.size(0),scor_bias.size(1),labels.size(2))

        labels = labels.expand(amb_incon.size(0),amb_incon.size(1),labels.size(2)).float()

        if model.score_dis == "Normal":

            exponents = -torch.pow(labels-scor_bias,2)/(amb_incon)
            normSum = torch.logsumexp(exponents,dim=2).sum()

        elif model.score_dis == "Beta":

            labels = generateData.betaNormalize(labels,model.score_min,model.score_max)

            scor_bias = torch.clamp(scor_bias,model.score_min,model.score_max)
            scor_bias = generateData.betaNormalize(scor_bias,model.score_min,model.score_max)
            a,b = generateData.meanvar_to_alphabeta(scor_bias,amb_incon/model.div_beta_var)

            scoresDis = Beta(1,1)
            logConst = scoresDis._log_normalizer(a,b)
            print(logConst.size())

            #normSum = (torch.log((torch.pow(labels,alpha-1)*torch.pow(1-labels,beta-1)).sum(dim=2))+logConst.sum(dim=2)).sum()
            normSum = logConst.sum()+torch.logsumexp((a-1)*torch.log(labels)+(b-1)*torch.log(1-labels),dim=2).sum()

            #print(normSum.size())
            #print()

            tensor = (a-1)*torch.log(labels)+(b-1)*torch.log(1-labels)

            #for i in range(tensor.size(0)):
            #    for j in range(tensor.size(1)):
            #        print(tensor[i,j])

            #print(normSum.size())
            sys.exit(0)

        else:
            raise ValueError("Unknown score distribution : {}".format(model.score_dis))

        loss += normSum

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

    loss = addLossTerms(loss,model,args.prior_weight,args.norm_sum,trainMatrix.is_cuda)

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

    epoch = 1
    dist = args.stop_crit+1
    lrCounter = 0
    while epoch < args.epochs and dist > args.stop_crit:

        if epoch%args.log_interval==0:

            print("\tEpoch : ",epoch,"dist",float(dist))

        #This condition determines when the learning rate should be updated (to follow the learning rate schedule)
        #The optimiser have to be rebuilt every time the learning rate is updated
        if (epoch-1) % ((args.epochs + 1)//len(args.lr)) == 0 or epoch==1:

            kwargs['lr'] = args.lr[lrCounter]
            #print("Learning rate : ",kwargs['lr'])
            if optimConst:
                optimizer = optimConst((getattr(model,param) for param in args.param_to_opti), **kwargs)
            else:
                optimizer = None
            if lrCounter<len(args.lr)-1:
                lrCounter += 1

        #print(model.state_dict()["diffs"])
        old_score = model.state_dict()["trueScores"].clone()
        loss = one_epoch_train(model,optimizer,trainSet,epoch, args,args.lr[lrCounter])
        #print(model.state_dict()["diffs"])
        #sys.exit(0)



        dist = torch.sqrt(torch.pow(old_score-model.state_dict()["trueScores"],2).sum())
        epoch += 1

        if epoch%args.log_interval==0:
            paramsToCsv(loss,model,args.exp_id,args.ind_id,epoch,args.score_dis,args.score_min,args.score_max)
            torch.save(model.state_dict(), "../models/{}/model{}_epoch{}".format(args.exp_id,args.ind_id,epoch))

    print("\tStopped at epoch ",epoch)
    return loss,epoch

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--only_init',action='store_true',help='To initialise a model without training it.\
                                                                        This still computes the confidence intervals')

    #Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

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
    model = modelBuilder.modelMaker(trainSet.size(1),len(trainSet),distorNbList,args.poly_deg,\
                                    args.score_dis,args.score_min,args.score_max,args.div_beta_var)
    if args.cuda:
        model = model.cuda()

    #Inititialise the model
    if args.init_mode == "init_base":
        model.init_base(trainSet)
    elif args.init_mode == "init_oracle":
        model.init_oracle(trainSet,args.dataset,float(args.perc_noise))
    else:
        raise ValueError("Unknown init method : {}".format(args.init_mode))

    torch.save(model.state_dict(), "../models/{}/model{}_epoch0".format(args.exp_id,args.ind_id))

    #Write the parameters of the model and its confidence interval in a csv file
    loss = model(trainSet)
    paramsToCsv(loss,model,args.exp_id,args.ind_id,epoch=0,scoresDis=args.score_dis,score_min=args.score_min,score_max=args.score_max)

    if not args.only_init:
        #Getting the contructor and the kwargs for the choosen optimizer
        optimConst,kwargs = get_OptimConstructor(args.optim,args.momentum)

        #If no learning rate is schedule is indicated (i.e. there's only one learning rate),
        #the args.lr argument will be a float and not a float list.
        #Converting it to a list with one element makes the rest of processing easier
        if type(args.lr) is float:
            args.lr = [args.lr]

        model.setPrior(args.prior,args.dataset)

        print(model.diffs)
        sys.exit(0)

        #print(model.bias,torch.sigmoid(model.incons),torch.sigmoid(model.diffs),model.trueScores)
        loss,epoch = train(model,optimConst,kwargs,trainSet, args)
        #print(model.bias,torch.sigmoid(model.incons),torch.sigmoid(model.diffs),model.trueScores)

        torch.save(model.state_dict(), "../models/{}/model{}_epoch{}".format(args.exp_id,args.ind_id,epoch))
        paramsToCsv(loss,model,args.exp_id,args.ind_id,epoch,args.score_dis,args.score_min,args.score_max)

if __name__ == "__main__":
    main()
