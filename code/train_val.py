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

    train_loss = loss

    #Writing the CSV files in the results/NameOfTheExperience/ folder
    #writeCSV(args,epoch,neg_log_proba)
    firstTrainBatch = False

    torch.save(model.state_dict(), "../models/{}/model{}_epoch{}".format(args.exp_id,args.ind_id, epoch))

    #print('\nEpoch {} Train set: Average loss: {:.4f},\n'.format(epoch,train_loss))

    return train_loss
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

def computeConfInter(loss,model):

    hessDiagList,gradList = computeHessDiagList(loss,model)
    confInterList = []

    for hessDiag in hessDiagList:

        confInter = 1.96/torch.sqrt(hessDiag)
        confInterList.append(confInter)

    return confInterList

def plotMLE(loss,model,exp_id,epoch,mos_mean,mos_std):

    confInterList = computeConfInter(loss,model)

    for i,key in enumerate(model.state_dict()):

        if key == "video_scor":
            plt.figure(i,figsize=(13,5))

        else:
            plt.figure(i)

        plt.grid()
        plt.xlabel("Individual")
        plt.ylabel("MLE")

        if model.state_dict()[key].is_cuda:
            values = model.state_dict()[key].cpu().numpy()
            yErrors = confInterList[i].cpu().detach().numpy()
        else:
            values = model.state_dict()[key].numpy()
            yErrors = confInterList[i].detach().numpy()

        if key =="video_amb":
            values = np.abs(values)

        plt.errorbar([i for i in range(len(values))],values, yerr=yErrors, fmt="*")

        if key == "video_scor":
            plt.errorbar([i+0.5 for i in range(len(values))],mos_mean, yerr=mos_std, fmt="*")

        plt.savefig("../vis/{}/{}_epoch{}.png".format(exp_id,key,epoch))

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

def computeBaselines(trainSet):
    mos_mean,mos_std = modelBuilder.MOS(trainSet,sub_rej=False,z_score=False)
    sr_mos_mean,sr_mos_std = modelBuilder.MOS(trainSet,sub_rej=True,z_score=False)
    zs_sr_mos_mean,zs_sr_mos_std = modelBuilder.MOS(trainSet,sub_rej=True,z_score=True)

    return mos_mean,sr_mos_mean,zs_sr_mos_mean,mos_std,sr_mos_std,zs_sr_mos_std

def mean_std(model,loss):
    mle_mean = model.video_scor
    mle_std = 1.96/torch.sqrt(computeHessDiag(loss,mle_mean)[0])
    return mle_mean,mle_std

def RMSE(vec_ref,vec):

    if not (type(vec) is np.ndarray):
        if vec.is_cuda:
            vec = vec.cpu()
        vec = vec.detach().numpy()
    if not (type(vec_ref) is np.ndarray):
        if vec_ref.is_cuda:
            vec_ref = vec_ref.cpu()
        vec_ref = vec_ref.detach().numpy()

    sq_sum = np.power(vec_ref-vec,2).sum()
    #if sq_sum>500:
    #    print(vec_ref-vec)

    #print(sq_sum)
    return np.sqrt(sq_sum/len(vec))

def deteriorateData(trainSet,nb_annot,nb_corr):

    data = trainSet.clone()
    nb_annot = int(nb_annot)
    nb_corr = int(nb_corr)

    #Remove annotators
    if nb_annot < data.size(1):
        annotToRemove = random.sample(range(data.size(1)),data.size(1)-nb_annot)
        data = modelBuilder.removeColumns(data,annotToRemove)

    #Randomly scramble subject scores
    if nb_corr > 0:
        annotToScramble = random.sample(range(nb_annot),nb_corr)
        data = modelBuilder.scrambleColumns(data,annotToScramble)

    return data

def robustness(model,trainSet,distorNbList,args,paramRange,paramName,corrKwargs,nbRep=100):

    #Getting the contructor and the kwargs for the choosen optimizer
    optimConst,kwargs = get_OptimConstructor(args.optim,args.momentum)

    model = modelBuilder.modelMaker(int(args.annot_nb),len(trainSet),distorNbList)
    loss,_ = train(model,optimConst,kwargs,trainSet, args)

    mle_mean_ref,_ = mean_std(model,loss)
    mos_mean_ref,sr_mean_ref,zs_sr_mean_ref = computeBaselines(trainSet)[:3]

    valueNumber = max(paramRange)-min(paramRange)
    mle_err = np.zeros((valueNumber,nbRep))
    mos_err = np.zeros((valueNumber,nbRep))
    sr_err = np.zeros((valueNumber,nbRep))
    zs_sr_err = np.zeros((valueNumber,nbRep))

    #print(args.erase_results)
    if (not os.path.exists("../results/{}/mle_err_epoch{}.csv".format(args.exp_id,args.epochs))) or args.erase_results:
        for i,paramValue in enumerate(paramRange):

            if i%5==0:
                print(paramName,":",paramValue,"/",max(paramRange))
            for j in range(nbRep):

                corrKwargs[paramName] = paramValue

                #Randomly remove or scramble annotator scores
                data = deteriorateData(trainSet,**corrKwargs)

                model = modelBuilder.modelMaker(int(corrKwargs["nb_annot"]),len(data),distorNbList)
                loss,_ = train(model,optimConst,kwargs,data, args)

                mle_mean,_ = mean_std(model,loss)
                mos_mean,sr_mean,zs_sr_mean = computeBaselines(data)[:3]

                mle_err[i,j] = RMSE(mle_mean_ref,mle_mean)
                mos_err[i,j] = RMSE(mos_mean_ref,mos_mean)
                sr_err[i,j] = RMSE(sr_mean_ref,sr_mean)
                zs_sr_err[i,j] = RMSE(zs_sr_mean_ref,zs_sr_mean)

            #sys.exit(0)
        np.savetxt("../results/{}/mle_err_epoch{}_model{}.csv".format(args.exp_id,args.epochs,args.ind_id),mle_err)
        np.savetxt("../results/{}/mos_err_epoch{}_model{}.csv".format(args.exp_id,args.epochs,args.ind_id),mos_err)
        np.savetxt("../results/{}/sr_err_epoch{}_model{}.csv".format(args.exp_id,args.epochs,args.ind_id),sr_err)
        np.savetxt("../results/{}/zs_sr_err_epoch{}_model{}.csv".format(args.exp_id,args.epochs,args.ind_id),zs_sr_err)

    else:
        mle_err = np.genfromtxt("../results/{}/mle_err_epoch{}.csv".format(args.exp_id,args.epochs))
        mos_err =  np.genfromtxt("../results/{}/mos_err_epoch{}.csv".format(args.exp_id,args.epochs))
        sr_err = np.genfromtxt("../results/{}/sr_err_epoch{}.csv".format(args.exp_id,args.epochs))
        zs_sr_err = np.genfromtxt("../results/{}/zs_sr_err_epoch{}.csv".format(args.exp_id,args.epochs))

    plt.figure()
    #rangeAnnot = np.array(rangeAnnot)
    plt.errorbar(paramRange,mos_err.mean(axis=1), yerr=1.96*mos_err.std(axis=1)/np.sqrt(nbRep),label="MOS")
    plt.errorbar(paramRange,sr_err.mean(axis=1), yerr=1.96*sr_err.std(axis=1)/np.sqrt(nbRep),label="SR-MOS")
    plt.errorbar(paramRange,zs_sr_err.mean(axis=1), yerr=1.96*zs_sr_err.std(axis=1)/np.sqrt(nbRep),label="ZS-SR-MOS")
    plt.errorbar(paramRange,mle_err.mean(axis=1), yerr=1.96*mle_err.std(axis=1)/np.sqrt(nbRep),label="MLE")
    plt.legend()

    plt.savefig("../vis/{}/robustness_epoch{}_model{}.png".format(args.exp_id,args.epochs,args.ind_id))

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

    trainSet,distorNbList = load_data.loadData(args.dataset,int(args.annot_nb))
    if args.cuda:
        trainSet = trainSet.cuda()

    #The group of class to detect
    np.random.seed(args.seed)

    #The folders where the experience file will be written
    if not (os.path.exists("../vis/{}".format(args.exp_id))):
        os.makedirs("../vis/{}".format(args.exp_id))
    if not (os.path.exists("../results/{}".format(args.exp_id))):
        os.makedirs("../results/{}".format(args.exp_id))
    if not (os.path.exists("../models/{}".format(args.exp_id))):
        os.makedirs("../models/{}".format(args.exp_id))

    modelType = "mlp"

    #Write the arguments in a config file so the experiment can be re-run
    argreader.writeConfigFile("../models/{}/{}{}.ini".format(args.exp_id,modelType,args.ind_id))

    #Building the model
    model = modelBuilder.modelMaker(int(args.annot_nb),len(trainSet),distorNbList)
    if args.cuda:
        model = model.cuda()
    torch.save(model.state_dict(), "../models/{}/{}{}_epoch0".format(args.exp_id,modelType,args.ind_id))

    #Getting the contructor and the kwargs for the choosen optimizer
    optimConst,kwargs = get_OptimConstructor(args.optim,args.momentum)

    #If no learning rate is schedule is indicated (i.e. there's only one learning rate),
    #the args.lr argument will be a float and not a float list.
    #Converting it to a list with one element makes the rest of processing easier
    if type(args.lr) is float:
        args.lr = [args.lr]

    paramRange = range(args.param_min,args.param_max)
    corrKwargs = {"nb_annot":args.annot_nb,"nb_corr":0}

    robustness(model,trainSet,distorNbList,args,paramRange,args.param_name,corrKwargs,args.nb_rep)
    #loss,epoch = train(model,optimConst,kwargs,trainSet, args)
    #mos_mean,mos_std = modelBuilder.MOS(trainSet,sub_rej=True,z_score=False)
    #plotMLE(loss,model,args.exp_id,epoch,mos_mean,mos_std)

if __name__ == "__main__":
    main()
