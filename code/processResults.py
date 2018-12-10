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
import train_val
import math
import glob
import matplotlib.cm as cm

def getModels(args,trainSet,distorNbList):

    iniPaths = glob.glob("../models/{}/*.ini".format(args.exp_id))

    #Count the number of net in the experiment
    netNumber = len(iniPaths)

    modelList = []
    modelNameList = []
    lossList = []
    #Finding the last weights for each model
    for i in range(netNumber):

        net_id = findNumbers(os.path.basename(iniPaths[i]))
        lastModelPath = sorted(glob.glob("../models/{}/model{}_epoch*".format(args.exp_id,net_id)),key=findNumbers)[-1]

        modelNameList.append(str(net_id))

        model = modelBuilder.modelMaker(trainSet.size(1),len(trainSet),distorNbList,args.poly_deg)
        model.load_state_dict(torch.load(lastModelPath))
        modelList.append(model)

        loss = model(trainSet)
        lossList.append(loss)

    return modelList,modelNameList,lossList

def PolyCoefficients(x, coeffs):

    o = len(coeffs)

    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y

def mean_std_plot(data,distorNbList,dataset,exp_id,model):
    #cmap = cm.get_cmap(name='rainbow')
    colors = cm.rainbow(np.linspace(0, 1, len(distorNbList)))

    plt.figure()
    currInd = 0
    x = np.linspace(1, 5, 10)
    for i in range(len(distorNbList)):

        #The data for all videos made from this reference video
        videoData = data[currInd:currInd+distorNbList[i]]
        currInd += distorNbList[i]

        means = videoData.mean(dim=1)
        stds = videoData.std(dim=1)

        plt.plot(means.numpy(),stds.numpy(),"*",color=colors[i])

        #Plotting the polynomial modeling the dependency between video mean score and video std score deviation
        coeffs = model.video_amb[i].cpu().detach().numpy()
        plt.plot(x, PolyCoefficients(x, coeffs),color=colors[i])
        #print(coeffs)
    plt.xlim([1, 5])
    plt.ylim([0,max(stds)*1.2])

    plt.savefig("../vis/{}/mean-stds_{}.png".format(exp_id,dataset))

def std(data,distorNbList,dataset):

    #cmap = cm.get_cmap(name='rainbow')
    colors = cm.rainbow(np.linspace(0, 1, len(distorNbList)))

    plt.figure()
    currInd = 0
    stds_mean = torch.zeros(len(distorNbList))
    stds_confInter = torch.zeros(len(distorNbList))
    for i in range(len(distorNbList)):

        #The data for all videos made from this reference video
        videoData = data[currInd:currInd+distorNbList[i]]
        currInd += distorNbList[i]

        stds = videoData.std(dim=1)
        stds_mean[i] = stds.mean()
        stds_confInter[i] = 1.96*stds.std()/math.sqrt(distorNbList[i])

        plt.errorbar([i],stds_mean[i].numpy(), yerr=stds_confInter[i].numpy(), fmt="*",color=colors[i])

    plt.ylim([0,max(stds_mean.numpy())*1.2])

    plt.savefig("../vis/stds_{}.png".format(dataset))

def computeBaselines(trainSet):
    mos_mean,mos_std = modelBuilder.MOS(trainSet,sub_rej=False,z_score=False)
    sr_mos_mean,sr_mos_std = modelBuilder.MOS(trainSet,sub_rej=True,z_score=False)
    zs_sr_mos_mean,zs_sr_mos_std = modelBuilder.MOS(trainSet,sub_rej=True,z_score=True)

    return mos_mean,sr_mos_mean,zs_sr_mos_mean,mos_std,sr_mos_std,zs_sr_mos_std

def mean_std(model,loss):
    mle_mean = model.video_scor
    mle_std = 1.96/torch.sqrt(train_val.computeHessDiag(loss,mle_mean)[0])
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
        data = modelBuilder.c(data,annotToScramble)

    return data

def robustness(model,trainSet,distorNbList,args,paramRange,paramName,corrKwargs,nbRep=100):

    if type(args.lr) is float:
        args.lr = [args.lr]

    #Getting the contructor and the kwargs for the choosen optimizer
    optimConst,kwargs = train_val.get_OptimConstructor(args.optim,args.momentum)

    model = modelBuilder.modelMaker(trainSet.size(1),len(trainSet),distorNbList,args.poly_deg)
    loss,_ = train_val.train(model,optimConst,kwargs,trainSet, args)

    mle_mean_ref,_ = mean_std(model,loss)
    mos_mean_ref,sr_mean_ref,zs_sr_mean_ref = computeBaselines(trainSet)[:3]

    valueNumber = max(paramRange)-min(paramRange)+1
    mle_err = np.zeros((valueNumber,nbRep,len(args.model_values)))
    mos_err = np.zeros((valueNumber,nbRep))
    sr_err = np.zeros((valueNumber,nbRep))
    zs_sr_err = np.zeros((valueNumber,nbRep))
    print(mle_err.shape)
    #print(args.erase_results)
    if (not os.path.exists("../results/{}/mle_err_epoch{}.csv".format(args.exp_id,args.epochs))) or args.erase_results:
        for i,paramValue in enumerate(paramRange):

            if i%5==0:
                print(paramName,":",paramValue,"/",max(paramRange))
            for j in range(nbRep):

                corrKwargs[paramName] = paramValue

                #Randomly remove or scramble annotator scores
                data = deteriorateData(trainSet,**corrKwargs)

                for k in range(len(args.model_values)):
                    model = modelBuilder.modelMaker(int(corrKwargs["nb_annot"]),len(data),distorNbList,args.poly_deg)
                    setattr(args, args.model_param, args.model_values[k])
                    loss,_ = train_val.train(model,optimConst,kwargs,data, args)
                    mle_mean,_ = mean_std(model,loss)
                    mle_err[i,j,k] = RMSE(mle_mean_ref,mle_mean)

                mos_mean,sr_mean,zs_sr_mean = computeBaselines(data)[:3]
                mos_err[i,j] = RMSE(mos_mean_ref,mos_mean)
                sr_err[i,j] = RMSE(sr_mean_ref,sr_mean)
                zs_sr_err[i,j] = RMSE(zs_sr_mean_ref,zs_sr_mean)

            #sys.exit(0)

        np.savetxt("../results/{}/mle_err_epoch{}.csv".format(args.exp_id,args.epochs),mle_err.reshape(mle_err.shape[0],mle_err.shape[1]*mle_err.shape[2]))
        np.savetxt("../results/{}/mos_err_epoch{}.csv".format(args.exp_id,args.epochs),mos_err)
        np.savetxt("../results/{}/sr_err_epoch{}.csv".format(args.exp_id,args.epochs),sr_err)
        np.savetxt("../results/{}/zs_sr_err_epoch{}.csv".format(args.exp_id,args.epochs),zs_sr_err)

    else:
        mle_err = np.genfromtxt("../results/{}/mle_err_epoch{}_model{}.csv".format(args.exp_id,args.epochs)).reshape(mle_err.shape[0],mle_err.shape[1]//len(args.model_values),len(args.model_values))
        mos_err =  np.genfromtxt("../results/{}/mos_err_epoch{}.csv".format(args.exp_id,args.epochs))
        sr_err = np.genfromtxt("../results/{}/sr_err_epoch{}.csv".format(args.exp_id,args.epochs))
        zs_sr_err = np.genfromtxt("../results/{}/zs_sr_err_epoch{}.csv".format(args.exp_id,args.epochs))

    plt.figure()
    #rangeAnnot = np.array(rangeAnnot)

    plt.errorbar(paramRange,mos_err.mean(axis=1), yerr=1.96*mos_err.std(axis=1)/np.sqrt(nbRep),label="MOS")
    plt.errorbar(paramRange,sr_err.mean(axis=1), yerr=1.96*sr_err.std(axis=1)/np.sqrt(nbRep),label="SR-MOS")
    plt.errorbar(paramRange,zs_sr_err.mean(axis=1), yerr=1.96*zs_sr_err.std(axis=1)/np.sqrt(nbRep),label="ZS-SR-MOS")
    for k in range(len(args.model_values)):
        plt.errorbar(paramRange,mle_err[:,:,k].mean(axis=1), yerr=1.96*mle_err[:,:,k].std(axis=1)/np.sqrt(nbRep),label="MLE{}".format(k))
    plt.legend()

    plt.savefig("../vis/{}/robustness_epoch{}_model{}.png".format(args.exp_id,args.epochs,args.ind_id))

def computeConfInter(loss,model):

    hessDiagList,gradList = train_val.computeHessDiagList(loss,model)
    confInterList = []

    for hessDiag in hessDiagList:

        confInter = 1.96/torch.sqrt(hessDiag)
        confInterList.append(confInter)

    return confInterList

def plotMLE(lossList,modelList,modelNameList,exp_id,mos_mean,mos_std):

    interv = [[-1,1],[0,2],[-1,2],[1,5]]

    for i,key in enumerate(modelList[0].state_dict()):
        for j,model in enumerate(modelList):

            confInterList = computeConfInter(lossList[j],modelList[j])

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

            plt.errorbar([i+0.3*j for i in range(len(values))],values, yerr=yErrors, fmt="*",label=modelNameList[j])

            if key == "video_scor":
                plt.errorbar([i+0.5 for i in range(len(values))],mos_mean, yerr=mos_std, fmt="*")

            plt.legend()
            plt.savefig("../vis/{}/{}.png".format(exp_id,key))

            plt.figure(i+len(model.state_dict()))
            plt.grid()
            plt.xlim(interv[i])

            plt.hist(values,10,density=True)
            plt.savefig("../vis/{}/{}_hist.png".format(exp_id,key))


def findNumbers(x):
    '''Extracts the numbers of a string and returns them as an integer'''

    return int((''.join(xi for xi in str(x) if xi.isdigit())))
def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--params',action='store_true',help='To plot the parameters learned')
    argreader.parser.add_argument('--robust',action='store_true',help='To test the robustness of the model')
    argreader.parser.add_argument('--std_mean',action='store_true',help='To plot the std of score as function of mean score \
                                with the parameters tuned to model this function')

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    #The folders where the experience file will be written
    if not (os.path.exists("../vis/{}".format(args.exp_id))):
        os.makedirs("../vis/{}".format(args.exp_id))
    if not (os.path.exists("../results/{}".format(args.exp_id))):
        os.makedirs("../results/{}".format(args.exp_id))
    if not (os.path.exists("../models/{}".format(args.exp_id))):
        os.makedirs("../models/{}".format(args.exp_id))

    trainSet,distorNbList = load_data.loadData(args.dataset)

    if args.params:

        mos_mean,mos_std = modelBuilder.MOS(trainSet,sub_rej=True,z_score=False)
        modelList,modelNameList,lossList = getModels(args,trainSet,distorNbList)
        plotMLE(lossList,modelList,modelNameList,args.exp_id,mos_mean,mos_std)

    if args.robust:
        model = modelBuilder.modelMaker(trainSet.size(1),len(trainSet),distorNbList,args.poly_deg)
        paramRange = range(args.param_min,args.param_max+1)
        corrKwargs = {"nb_annot":trainSet.size(1),"nb_corr":0}
        robustness(model,trainSet,distorNbList,args,paramRange,args.param_name,corrKwargs,args.nb_rep)

    if args.std_mean:
        config = configparser.ConfigParser()
        config.read("../models/{}/model{}.ini".format(args.exp_id,args.ind_id))
        model = modelBuilder.modelMaker(trainSet.size(1),len(trainSet),distorNbList,int(config["default"]["poly_deg"]))
        lastModelPath = sorted(glob.glob("../models/{}/model{}_epoch*".format(args.exp_id,args.ind_id)),key=lambda x:findNumbers(x))[-1]
        model.load_state_dict(torch.load(lastModelPath))
        mean_std_plot(trainSet,distorNbList,args.dataset,args.exp_id,model)

if __name__ == "__main__":
    main()
