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
import glob

from torch.distributions.beta import Beta

class GradNoise():
    '''A class to add gaussian noise in weight update

    To be used with a pytorch hook so this function is called every time there is a weight update

    '''

    def __init__(self,ampl=0.1):
        '''
        Args:
            ampl (float): the ratio of the noise norm to the gradient norm
        '''

        self.ampl=ampl

    def __call__(self,grad):
        '''
        Args:
            grad (torch.autograd.variable.Variable): the gradient of the udpate
        Returns:
            The gradient with added noise
        '''

        if self.ampl == 0:
            return grad
        else:
            self.noise = torch.tensor(np.random.normal(size=grad.detach().cpu().numpy().shape))/2
            gradMean = torch.abs(grad).mean()
            noise =self.ampl*gradMean*self.noise

            #print(gradMean,torch.abs(torch.tensor(noise)).mean(),torch.abs(torch.tensor(noise)).mean()/gradMean)

            if grad.is_cuda:
                return grad + noise.double().cuda().type("torch.cuda.FloatTensor")
            else:
                return grad + noise.double()

def paramsToCsv(loss,model,exp_id,ind_id,epoch,scoresDis,score_min,score_max,nb_video_per_content):


    keys = list(model.state_dict().keys())

    for i,key in enumerate(keys):
        tensor = getattr(model,key)

        confInterv = processResults.computeConfInter(loss,tensor).cpu().detach().numpy().reshape(-1)[:,np.newaxis]

        if model.score_dis == "Beta":
            if (keys[i] =="diffs" or keys[i]=="incons"):
                tensor = torch.sigmoid(tensor)

        if keys[i] ==  "trueScores":

            if not nb_video_per_content is None:
                trueScores = torch.zeros(len(tensor)).double()
                vidInds = torch.arange(len(tensor))

                trueScores[vidInds%nb_video_per_content == 0] = score_max
                trueScores[vidInds%nb_video_per_content != 0] = tensor[vidInds%nb_video_per_content != 0]

                tensor = trueScores

            tensor = torch.clamp(tensor,score_min,score_max)

        tensor = tensor.cpu().detach().numpy().reshape(-1)[:,np.newaxis]

        concat = np.concatenate((tensor,confInterv),axis=1)
        np.savetxt("../results/{}/model{}_epoch{}_{}.csv".format(exp_id,ind_id,epoch,keys[i]),concat,delimiter="\t")

def addLossTerms(loss,model,weight):

    if weight>0:
        loss = weight*model.prior(loss)

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

    loss = addLossTerms(loss,model,args.prior_weight)

    loss.backward(retain_graph=True)

    if args.optim == "LBFGS":

        def closure():
            optimizer.zero_grad()
            loss = model(data)
            loss.backward(retain_graph=True)
            return loss
        optimizer.step(closure)
        optimizer.zero_grad()

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
        kwargs = {}

    elif optimStr=="NEWTON":
        optimConst = None
        kwargs = {}
    else:
        raise ValueError("Unknown optimisation algorithm : {}.".format(args.optim))

    print("Optim is :",optimConst)

    return optimConst,kwargs

def train(model,optimConst,kwargs,trainSet, args,startEpoch,nb_video_per_content):

    epoch = startEpoch
    dist = args.stop_crit+1
    lrCounter = 0

    distArray = np.zeros((args.epochs))
    lossArray = np.zeros((args.epochs))

    distDict = {"bias":distArray.copy(),"trueScores":distArray.copy(),\
                "incons":distArray.copy(),"diffs":distArray.copy(),"all":distArray.copy()}
    oldParam = {"bias":None,"trueScores":None,\
                "incons":None,"diffs":None}

    while epoch < args.epochs and dist > args.stop_crit:

        if epoch%args.log_interval==0:

            print("\tEpoch : ",epoch,"dist",distDict["all"][epoch-2].item())

        #This condition determines when the learning rate should be updated (to follow the learning rate schedule)
        #The optimiser have to be rebuilt every time the learning rate is updated
        if (epoch-1) % ((args.epochs + 1)//len(args.lr)) == 0 or epoch==startEpoch:

            kwargs['lr'] = args.lr[lrCounter]
            if lrCounter<len(args.lr)-1:
                lrCounter += 1

            if args.train_mode == "joint":
                if optimConst:
                    optimizer = optimConst((getattr(model,param) for param in args.param_to_opti), **kwargs)
                else:
                    optimizer = None

        if args.train_mode == "alternate":
            if (epoch-1) % args.alt_epoch_nb == 0:

                paramName = args.param_to_opti[((epoch-1)//args.alt_epoch_nb) % len(args.param_to_opti)]

                optimizer = optimConst((getattr(model,paramName+"_opti"),), **kwargs)

        for key in oldParam.keys():
            oldParam[key] = getattr(model,key).clone()

        loss = one_epoch_train(model,optimizer,trainSet,epoch, args,args.lr[lrCounter])
        lossArray[epoch-1] = loss

        #Computing distance for all parameters
        for key in oldParam.keys():
            distDict[key][epoch-1] = torch.pow(oldParam[key]-getattr(model,key),2).sum()
            distDict["all"][epoch-1] += distDict[key][epoch-1]
        for key in  oldParam.keys():
            distDict[key][epoch-1] = np.sqrt(distDict[key][epoch-1])
        distDict["all"][epoch-1] = np.sqrt(distDict["all"][epoch-1])
        dist = distDict["all"][epoch-1]

        epoch += 1

        if epoch%args.log_interval==0:
            paramsToCsv(loss,model,args.exp_id,args.ind_id,epoch,args.score_dis,args.score_min,args.score_max,nb_video_per_content)
            torch.save(model.state_dict(), "../models/{}/model{}_epoch{}".format(args.exp_id,args.ind_id,epoch))

    #Writing the array in a csv file
    if epoch<args.epochs:
        for key in distDict.keys():
            distDict[key] = distDict[key][:epoch-1]

    lossArray = lossArray[:epoch-1]
    fullDistArray = np.concatenate([distDict[key][:,np.newaxis] for key in  distDict.keys()],axis=1)

    header = ''
    for i,key in enumerate(distDict.keys()):
        header += key+"," if i<len(distDict.keys())-1 else key

    np.savetxt("../results/{}/model{}_dist.csv".format(args.exp_id,args.ind_id),fullDistArray,header=header,delimiter=",",comments='')
    np.savetxt("../results/{}/model{}_nll.csv".format(args.exp_id,args.ind_id),lossArray,delimiter=",",comments='')

    print("\tStopped at epoch ",epoch,"dist",distDict["all"][epoch-2].item())
    return loss,epoch

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--only_init',action='store_true',help='To initialise a model without training it.\
                                                                        This still computes the confidence intervals')

    argreader.parser.add_argument('--init_id',type=str,metavar="N",help='The index of the model to use as initialisation. \
                                                                            The weight of the last epoch will be used.')

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

    #Loading data
    trainSet,distorNbList = load_data.loadData(args.dataset)

    if args.ref_vid_to_score_max:
        nb_video_per_content = int(processResults.readConfFile("../data/"+args.dataset+".ini",["nb_video_per_content"])[0])
    else:
        nb_video_per_content = None

    #Write the arguments in a config file so the experiment can be re-run
    argreader.writeConfigFile("../models/{}/model{}.ini".format(args.exp_id,args.ind_id))

    if args.cuda:
        trainSet = trainSet.cuda()

    #Building the model
    model = modelBuilder.modelMaker(trainSet.size(1),len(trainSet),distorNbList,args.poly_deg,\
                                    args.score_dis,args.score_min,args.score_max,args.div_beta_var,\
                                    args.prior_update_frequ,args.extr_sco_dep,nb_video_per_content)

    if args.cuda:
        model = model.cuda()

    #Inititialise the model
    if args.start_mode == "base_init":
        model.init(trainSet,args.dataset,args.score_dis,args.param_not_gt,\
                    args.true_scores_init,args.bias_init,args.diffs_init,args.incons_init)
        startEpoch=1
    elif args.start_mode == "iter_init":
        model.init(trainSet,args.dataset,args.score_dis,args.param_not_gt,\
                    args.true_scores_init,args.bias_init,args.diffs_init,args.incons_init,iterInit=True)
        startEpoch=1
    elif args.start_mode == "fine_tune":
        init_path = sorted(glob.glob("../models/{}/model{}_epoch*".format(args.exp_id,args.init_id)),key=processResults.findNumbers)[-1]
        model.load_state_dict(torch.load(init_path))
        startEpoch = processResults.findNumbers(os.path.basename(init_path).replace("model{}".format(args.init_id),""))

    else:
        raise ValueError("Unknown init method : {}".format(args.start_mode))

    #Adding normal noise to the gradients
    gradNoise = GradNoise(ampl=args.noise)
    for p in model.parameters():
        p.register_hook(gradNoise)

    torch.save(model.state_dict(), "../models/{}/model{}_epoch0".format(args.exp_id,args.ind_id))

    #Write the parameters of the model and its confidence interval in a csv file
    loss = model(trainSet)

    paramsToCsv(loss,model,args.exp_id,args.ind_id,epoch=0,scoresDis=args.score_dis,score_min=args.score_min,score_max=args.score_max,nb_video_per_content=nb_video_per_content)

    if not args.only_init:
        #Getting the contructor and the kwargs for the choosen optimizer
        optimConst,kwargs = get_OptimConstructor(args.optim,args.momentum)

        #If no learning rate is schedule is indicated (i.e. there's only one learning rate),
        #the args.lr argument will be a float and not a float list.
        #Converting it to a list with one element makes the rest of processing easier
        if type(args.lr) is float:
            args.lr = [args.lr]

        model.setPrior(args.prior,args.dataset)

        loss,epoch = train(model,optimConst,kwargs,trainSet, args,startEpoch=startEpoch,nb_video_per_content=nb_video_per_content)
        torch.save(model.state_dict(), "../models/{}/model{}_epoch{}".format(args.exp_id,args.ind_id,epoch))
        paramsToCsv(loss,model,args.exp_id,args.ind_id,epoch,args.score_dis,args.score_min,args.score_max,nb_video_per_content=nb_video_per_content)

if __name__ == "__main__":
    main()
