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
import generateData
from torch.distributions.beta import Beta
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from matplotlib.lines import Line2D

def distrPlot(dataset,exp_id,indModel,plotScoreDis=False,nbPlot=10,dx=0.01):

    modelConf = configparser.ConfigParser()
    modelConf.read("../models/{}/model{}.ini".format(exp_id,indModel))
    modelConf = modelConf['default']

    xInt,distorNbList = load_data.loadData(dataset)

    #Building the model
    model = modelBuilder.modelMaker(xInt.size(1),len(xInt),distorNbList,int(modelConf["poly_deg"]),modelConf["score_dis"],\
                                    int(modelConf["score_min"]),int(modelConf["score_max"]),float(modelConf["div_beta_var"]))

    paramsPaths = sorted(glob.glob("../models/{}/model{}_epoch*".format(exp_id,indModel)))
    print(paramsPaths)

    paramKeys = ["bias","incons","diffs","trueScores"]
    tensorDict = {"bias":np.zeros((len(paramsPaths),getattr(model,"bias").size(0))),\
                  "incons":np.zeros((len(paramsPaths),getattr(model,"incons").size(0))),\
                  "diffs":np.zeros((len(paramsPaths),getattr(model,"diffs").size(0))),\
                  "trueScores":np.zeros((len(paramsPaths),getattr(model,"trueScores").size(0)))}

    indexs = np.random.choice(range(xInt.size(1)),size=nbPlot)
    vidIndex = np.random.choice(range(xInt.size(0)),size=1)[0]

    if plotScoreDis:

        colors = cm.rainbow(np.linspace(0, 1, xInt.size(0)))
        markers = [m for m, func in Line2D.markers.items() if func != 'nothing' and m not in Line2D.filled_markers]
        if len(markers) < nbPlot:
            markers = ["" for i in range(xInt.size(1))]
        else:
            markers = markers[:xInt.size(1)]

        maxCdfs=0
        cdfsList = torch.zeros((len(paramsPaths),nbPlot,int(1/dx)))

        for i,paramsPath in enumerate(paramsPaths):

            epoch = findNumbers(os.path.basename(paramsPath).replace("model{}".format(indModel),""))
            print("Processing epoch",epoch)

            model.load_state_dict(torch.load(paramsPath))

            scoreDis = model.compScoreDis(xInt.is_cuda)

            cdf = lambda x: torch.exp(scoreDis.log_prob(x))
            x_coord = torch.arange(0,1,dx)
            cdfs = cdf(x_coord)

            cdfs = cdfs[vidIndex,indexs]

            #cdfs = cdfs.view(cdfs.size(0)*cdfs.size(1),cdfs.size(2))
            #cdfs = cdfs[indexs]

            if cdfs[:,5:-5].max() > maxCdfs:
                maxCdfs = cdfs[:,5:-5].max()

            cdfsList[i] = cdfs

        #Wrinting the images with the correct range
        for i,paramsPath in enumerate(paramsPaths):

            epoch = findNumbers(os.path.basename(paramsPath).replace("model{}".format(indModel),""))
            plt.figure(i)
            subplot = plt.subplot(2,1,1)

            subplot.set_ylim((0,maxCdfs.item()))
            for j,(k,cdf) in enumerate(zip(indexs,cdfs)):
                subplot.plot(x_coord.numpy(),cdfsList[i,j].detach().numpy(),color=colors[vidIndex],marker=markers[j])

            subplot = plt.subplot(2,1,2)
            subplot.hist(generateData.betaNormalize(xInt[vidIndex].detach().numpy(),int(modelConf["score_min"]),int(modelConf["score_max"])),color=colors[vidIndex],range=(0,1))

            plt.savefig("../vis/{}/scores_{}_epoch{}.png".format(exp_id,indModel,epoch))
            plt.close()

    for key in paramKeys:
        tensorPathList = sorted(glob.glob("../results/{}/model{}_epoch*_{}.csv".format(exp_id,indModel,key)),key=findNumbers)
        for i,tensorPath in enumerate(tensorPathList):
            tensorDict[key][i] = np.genfromtxt(tensorPath)[:,0].reshape(-1)

    xRangeDict = {}
    #Plot the distribution of the parameters
    for i,key in enumerate(paramKeys):
        xMin = np.min(tensorDict[key])
        xMax = np.max(tensorDict[key])

        if key=="bias":
            if np.abs(xMin) > xMax:
                maxVal = np.abs(xMin)
            else:
                maxVal = xMax
            xRangeDict[key] = (-maxVal,maxVal)
        else:
            xRangeDict[key] = (xMin,xMax)

        for j in range(len(tensorDict[key])):
            plt.figure(i+j*len(paramKeys)+len(paramsPaths),figsize=(10,5))
            plt.title(key)

            #Plot ground truth distribution
            subplot = plt.subplot(2,1,1)
            subplot.set_xlim(xRangeDict[key])
            subplot.set_ylim(0,len(tensorDict[key][j])*0.5)
            subplot.hist(np.genfromtxt("../data/{}_{}.csv".format(dataset,key)),range=xRangeDict[key],color="red")

            #Plot predicted distribution
            subplot = plt.subplot(2,1,2)
            subplot.set_xlim(xRangeDict[key])
            subplot.set_ylim(0,len(tensorDict[key][j])*0.5)
            subplot.hist(tensorDict[key][j],range=xRangeDict[key],color="blue")

            epoch = findNumbers(os.path.basename(paramsPaths[j]).replace("model{}".format(indModel),""))
            plt.savefig("../vis/{}/{}_{}_epoch{}.png".format(exp_id,key,indModel,epoch))
            plt.close()


def plotBias(dataset,exp_id):

    indList = list(map(lambda x:os.path.basename(x).replace("model","").replace(".ini",""),sorted(glob.glob("../models/{}/model*.ini".format(exp_id)),key=findNumbers)))
    paramsPaths = sorted(glob.glob("../models/{}/model{}_epoch*".format(exp_id,indList[0])))
    print(paramsPaths)
    tensor = np.zeros((len(indList),len(paramsPaths),len(np.genfromtxt("../results/{}/model{}_epoch0_bias.csv".format(exp_id,indList[0])))))
    print((len(indList),len(paramsPaths),len(np.genfromtxt("../results/{}/model{}_epoch0_bias.csv".format(exp_id,indList[0])))))

    for k,indModel in enumerate(indList):

        tensorPathList = sorted(glob.glob("../results/{}/model{}_epoch*_bias.csv".format(exp_id,indModel)),key=findNumbers)
        for i,tensorPath in enumerate(tensorPathList):
            tensor[k,i] = np.genfromtxt(tensorPath)[:,0].reshape(-1)

    xMin = np.min(tensor)
    xMax = np.max(tensor)

    xRangeDict = (xMin,xMax)

    for k,indModel in enumerate(indList):

        paramsPaths = sorted(glob.glob("../models/{}/model{}_epoch*".format(exp_id,indModel)),key=findNumbers)

        #Plot the bias error as a function of the absolute value of the bias
        for j in range(len(tensor[k])):

            epoch = findNumbers(os.path.basename(paramsPaths[j]).replace("model{}".format(indModel),""))
            bias_gt = np.genfromtxt("../data/{}_bias.csv".format(dataset))

            plt.figure(2*j,figsize=(10,5))
            error = (np.abs(bias_gt-tensor[k,j])/np.abs(bias_gt))
            plt.xlim(0,max(np.abs(xRangeDict[0]),np.abs(xRangeDict[1])))
            plt.plot(np.abs(tensor[k,j]),error,"*",label=indModel)
            if k==len(indList)-1:
                plt.legend()
                plt.savefig("../vis/{}/biasError_epoch{}.png".format(exp_id,epoch))
                plt.close()

            plt.figure(2*j+1,figsize=(10,5))
            plt.plot(bias_gt,tensor[k,j],"*",label=indModel)
            x = np.arange(xRangeDict[0],xRangeDict[1],0.01)
            plt.plot(x,x)

            plt.xlim(xRangeDict)
            plt.ylim(xRangeDict)

            if k==len(indList)-1:
                plt.legend()
                plt.savefig("../vis/{}/biasVSgt_epoch{}.png".format(exp_id,epoch))
                plt.close()

def scatterPlot(dataset,exp_id,indModel):

    plt.figure()

    trueScores_gt = np.genfromtxt("../data/"+dataset+"_trueScores.csv")

    modelConf = configparser.ConfigParser()
    modelConf.read("../models/{}/model{}.ini".format(exp_id,indModel))
    modelConf = modelConf['default']

    trueScoresPathList = sorted(glob.glob("../results/{}/model{}_epoch*_trueScores.csv".format(exp_id,indModel)),key=findNumbers)
    colors = cm.rainbow(np.linspace(0, 1, len(trueScoresPathList)))

    for i,trueScoresPath in enumerate(trueScoresPathList):
        print(trueScoresPath)
        trueScores = np.genfromtxt(trueScoresPath)
        epoch = findNumbers(os.path.basename(trueScoresPath).replace("model{}".format(indModel),""))
        plt.plot(trueScores[:,0],trueScores_gt,"*",label="epoch{}".format(epoch),color=colors[i])

    plt.legend()
    plt.savefig("../vis/{}/scatterPlot_{}.png".format(exp_id,indModel))

def fakeDataDIstr(args):

    dataConf = configparser.ConfigParser()
    dataConf.read("../data/{}.ini".format(args.dataset))
    dataConf = dataConf['default']

    paramNameList = ["trueScores","diffs","incons","bias"]
    dx = 0.01


    dist_dic = {"trueScores":lambda x:torch.exp(Uniform(1,5).log_prob(x)),\
                "diffs":lambda x:torch.exp(Beta(float(dataConf['diff_alpha']), float(dataConf["diff_beta"])).log_prob(x)), \
                "incons":lambda x:torch.exp(Beta(float(dataConf["incons_alpha"]), float(dataConf["incons_beta"])).log_prob(x)),\
                "bias":lambda x:torch.exp(Normal(torch.zeros(1), float(dataConf["bias_std"])*torch.eye(1)).log_prob(x))}

    range_dic = {"trueScores":torch.arange(1,5,dx),\
                "diffs":torch.arange(0,1,dx), \
                "incons":torch.arange(0,1,dx),\
                "bias":torch.arange(-3*float(dataConf["bias_std"]),3*float(dataConf["bias_std"]),dx)}

    for i,paramName in enumerate(paramNameList):

        paramValues = np.genfromtxt("../data/artifData{}_{}.csv".format(dataConf["dataset_id"],paramName))
        trueCDF = dist_dic[paramName](range_dic[paramName]).numpy().reshape(-1)

        plt.figure(i)
        plt.plot(range_dic[paramName].numpy(),trueCDF)
        plt.hist(paramValues,10,density=True)
        plt.savefig("../vis/{}/{}_dis.png".format(args.exp_id,paramName))

def getModels(args,trainSet,distorNbList):

    iniPaths = sorted(glob.glob("../models/{}/*.ini".format(args.exp_id)))

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

def deteriorateData(trainSet,nb_annot,nb_corr,noise_percent):

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

    if noise_percent>0:
        data = modelBuilder.scoreNoise(data,noise_percent)

    return data

def robustness(model,trainSet,distorNbList,args,paramValues,paramName,corrKwargs,nbRep=100):

    if type(args.lr) is float:
        args.lr = [args.lr]

    #Getting the contructor and the kwargs for the choosen optimizer
    optimConst,kwargs = train_val.get_OptimConstructor(args.optim,args.momentum)

    model = modelBuilder.modelMaker(trainSet.size(1),len(trainSet),distorNbList,args.poly_deg)
    loss,_ = train_val.train(model,optimConst,kwargs,trainSet, args)

    mle_mean_ref,_ = mean_std(model,loss)
    mos_mean_ref,sr_mean_ref,zs_sr_mean_ref = computeBaselines(trainSet)[:3]

    valueNumber = len(paramValues)
    mle_err = np.zeros((valueNumber,nbRep,len(args.model_values)))
    mos_err = np.zeros((valueNumber,nbRep))
    sr_err = np.zeros((valueNumber,nbRep))
    zs_sr_err = np.zeros((valueNumber,nbRep))

    if (not os.path.exists("../results/{}/mle_err_epoch{}.csv".format(args.exp_id,args.epochs))) or args.erase_results:
        for i,paramValue in enumerate(paramValues):

            if i%5==0:
                print(paramName,":",paramValue,"/",valueNumber)
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


    plt.errorbar(paramValues,mos_err.mean(axis=1), yerr=1.96*mos_err.std(axis=1)/np.sqrt(nbRep),label="MOS")
    plt.errorbar(paramValues,sr_err.mean(axis=1), yerr=1.96*sr_err.std(axis=1)/np.sqrt(nbRep),label="SR-MOS")
    plt.errorbar(paramValues,zs_sr_err.mean(axis=1), yerr=1.96*zs_sr_err.std(axis=1)/np.sqrt(nbRep),label="ZS-SR-MOS")
    for k in range(len(args.model_values)):
        plt.errorbar(paramValues,mle_err[:,:,k].mean(axis=1), yerr=1.96*mle_err[:,:,k].std(axis=1)/np.sqrt(nbRep),label="MLE{}".format(k))
    plt.ylim(ymin=0)
    plt.legend()

    plt.savefig("../vis/{}/robustness_epoch{}_model{}.png".format(args.exp_id,args.epochs,args.ind_id))

def computeConfInter(loss,model):

    hessDiagList,gradList = train_val.computeHessDiagList(loss,model)
    confInterList = []

    for hessDiag in hessDiagList:

        #print(hessDiag)
        confInter = 1.96/torch.sqrt(hessDiag)
        confInterList.append(confInter)

    return confInterList

def plotMLE(lossList,modelList,modelNameList,exp_id,dataset=None,ind_id=None,mos_mean=None,mos_std=None):

    interv = [[-1,1],[0,2],[-1,2],[1,5]]

    if not (ind_id is None):
        modelList = [modelList[ind_id]]
        lossList = [lossList[ind_id]]
        modelNameList = [modelNameList[ind_id]]

    for i,key in enumerate(modelList[0].state_dict()):

        if os.path.exists("../data/{}_{}.csv".format(dataset,key)):
            gt_values = np.genfromtxt("../data/{}_{}.csv".format(dataset,key))
        else:
            gt_values = None

        for j,model in enumerate(modelList):

            confInterList = computeConfInter(lossList[j],modelList[j])

            if key == "video_scor":
                plt.figure(i,figsize=(20,5))
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
            #plt.plot([i+0.3*j for i in range(len(values))],values,"*",label=modelNameList[j])

            if key == "video_scor" and (not (mos_mean is None)):
                plt.errorbar([i+0.5 for i in range(len(values))],mos_mean, yerr=mos_std, fmt="*")

            if not (gt_values is None):
                plt.plot([i for i in range(len(gt_values))],gt_values,"*",label="GT")

            plt.legend()
            plt.savefig("../vis/{}/{}.png".format(exp_id,key))

            plt.figure(i+len(model.state_dict()))
            plt.grid()
            plt.xlim(interv[i])

            plt.hist(values,10,density=True)
            plt.savefig("../vis/{}/{}_hist.png".format(exp_id,key))

def error(dictPred,dictGT,paramNames):

    errList = []

    for name in paramNames:
        errList.append(errorVec(dictPred[name][:,0],dictGT[name]))

    return errList

def errorVec(vec,vec_ref):

    if not (type(vec) is np.ndarray):
        if vec.is_cuda:
            vec = vec.cpu()
        vec = vec.detach().numpy()
    if not (type(vec_ref) is np.ndarray):
        if vec_ref.is_cuda:
            vec_ref = vec_ref.cpu()
        vec_ref = vec_ref.detach().numpy()


    return (np.abs(vec_ref-vec)/np.abs(vec_ref)).mean()

def includPerc(dictPred,dictGT,paramNames):

    inclList = []

    for name in paramNames:
        inclList.append(includPercVec(dictPred[name][:,0],dictPred[name][:,1],dictGT[name]))

    return inclList

def includPercVec(mean,confIterv,gt):

    includNb = ((mean - confIterv < gt)* (gt < mean + confIterv)).sum()

    return includNb/len(mean)

def extractParamName(path):

    path = os.path.basename(path).replace(".csv","")
    return path[path.find("_")+1:]

def compareWithGroundTruth(exp_id,dataset,varParams):

    paramFiles = glob.glob("../data/{}_*.csv".format(dataset))
    paramFiles.remove("../data/{}_scores.csv".format(dataset))
    paramNameList = sorted(list(map(extractParamName,paramFiles)))

    gtParamDict = {}
    for paramName in paramNameList:
        gtParamDict[paramName] = np.genfromtxt("../data/{}_{}.csv".format(dataset,paramName))

    paramsDicList = []
    modelConfigPaths = sorted(glob.glob("../models/{}/*.ini".format(exp_id)),key=findNumbers)
    csvHead = "{}".format(varParams)+"".join(["\t{}".format(paramNameList[i]) for i in range(len(paramNameList))])

    csvErr = ""
    csvInclu = ""

    for i in range(len(modelConfigPaths)):

        modelInd = findNumbers(os.path.basename(modelConfigPaths[i]))

        config = configparser.ConfigParser()
        config.read(modelConfigPaths[i])
        config = config["default"]
        paramValue = ''
        for varParam in varParams:
            paramValue += config[varParam]+","

        paramDict = {}
        for paramName in paramNameList:

            lastEpochPath = sorted(glob.glob("../results/{}/model{}_epoch*_{}.csv".format(exp_id,modelInd,paramName)),key=findNumbers)[-1]

            lastEpoch = findNumbers(os.path.basename(lastEpochPath).replace("model{}".format(modelInd),""))
            paramDict[paramName] = np.genfromtxt("../results/{}/model{}_epoch{}_{}.csv".format(exp_id,modelInd,lastEpoch,paramName))

        errors = error(paramDict,gtParamDict,paramNameList)
        csvErr += "{}".format(paramValue)+"".join(["\t{}".format(round(100*errors[i],2)) for i in range(len(errors))])+"\n"

        incPer = includPerc(paramDict,gtParamDict,paramNameList)
        csvInclu += "{}".format(paramValue)+"".join(["\t{}".format(round(100*incPer[i],2)) for i in range(len(incPer))])+"\n"

    with open("../results/{}/err_{}.csv".format(exp_id,dataset),"w") as text_file:
        print(csvHead,file=text_file)
        print(csvErr,file=text_file,end="")

    with open("../results/{}/inclPerc_{}.csv".format(exp_id,dataset),"w") as text_file:
        print(csvHead,file=text_file)
        print(csvInclu,file=text_file)

def findNumbers(x):
    '''Extracts the numbers of a string and returns them as an integer'''

    return int((''.join(xi for xi in str(x) if xi.isdigit())))

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--params',type=int, metavar='S',nargs="*",help='To plot the parameters learned')

    argreader.parser.add_argument('--mos',action='store_true',help='To plot the scores found by mos with the --params plot')
    argreader.parser.add_argument('--robust',action='store_true',help='To test the robustness of the model')
    argreader.parser.add_argument('--std_mean',action='store_true',help='To plot the std of score as function of mean score \
                                with the parameters tuned to model this function')
    argreader.parser.add_argument('--comp_gt',type=str,nargs="*",metavar='PARAM',help='To compare the parameters found with the ground truth parameters. Require a fake dataset. The argument should\
                                    be the name of the parameter varying across the different models in the experiment.')
    argreader.parser.add_argument('--artif_data',action='store_true',help='To plot the real and empirical distribution of the parameters of a fake dataset. \
                                    The fake dataset to plot is set by the --dataset argument')
    argreader.parser.add_argument('--distr_plot',type=int,help='To plot the distribution of each score for a given model. The argument value is the model id. \
                                    The first argument should be the model id')
    argreader.parser.add_argument('--param_distr_plot',type=int,help='Like --distr_plot but only plot the parameters distribution and not the scores distributions.')
    argreader.parser.add_argument('--scatter_plot',type=int,help='To plot the real and predicted scores of a fake dataset in a 2D plot. \
                                    The first argument should be the model id')
    argreader.parser.add_argument('--plot_bias',action='store_true',help='To plot the error of every bias parameters at each epoch for each model')

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


    #The folders where the experience file will be written
    if not (os.path.exists("../vis/{}".format(args.exp_id))):
        os.makedirs("../vis/{}".format(args.exp_id))
    if not (os.path.exists("../results/{}".format(args.exp_id))):
        os.makedirs("../results/{}".format(args.exp_id))
    if not (os.path.exists("../models/{}".format(args.exp_id))):
        os.makedirs("../models/{}".format(args.exp_id))

    trainSet,distorNbList = load_data.loadData(args.dataset)

    if args.params:

        modelList,modelNameList,lossList = getModels(args,trainSet,distorNbList)
        if len(args.params) > 0:
            ind_id = args.params[0]
        else:
            ind_id = None

        if args.mos:
            mos_mean,mos_std = modelBuilder.MOS(trainSet,sub_rej=True,z_score=False)
        else:
            mos_mean,mos_std = None,None

        plotMLE(lossList,modelList,modelNameList,args.exp_id,dataset=args.dataset,ind_id=ind_id,mos_mean=mos_mean,mos_std=mos_std)

    if args.robust:
        model = modelBuilder.modelMaker(trainSet.size(1),len(trainSet),distorNbList,args.poly_deg)
        corrKwargs = {"nb_annot":trainSet.size(1),"nb_corr":0,"score_noise":0}
        if type(args.rob_param_values) is float:
            args.rob_param_values = [args.rob_param_values]
        if type(args.model_values) is float:
            args.model_values = [args.model_values]

        robustness(model,trainSet,distorNbList,args,args.rob_param_values,args.rob_param,corrKwargs,args.nb_rep)

    if args.std_mean:
        config = configparser.ConfigParser()
        config.read("../models/{}/model{}.ini".format(args.exp_id,args.ind_id))
        model = modelBuilder.modelMaker(trainSet.size(1),len(trainSet),distorNbList,int(config["default"]["poly_deg"]))
        lastModelPath = sorted(glob.glob("../models/{}/model{}_epoch*".format(args.exp_id,args.ind_id)),key=lambda x:findNumbers(x))[-1]
        model.load_state_dict(torch.load(lastModelPath))
        mean_std_plot(trainSet,distorNbList,args.dataset,args.exp_id,model)

    if args.comp_gt:
        compareWithGroundTruth(args.exp_id,args.dataset,args.comp_gt)

    if args.artif_data:
        fakeDataDIstr(args)

    if args.distr_plot:
        distrPlot(args.dataset,args.exp_id,args.distr_plot,plotScoreDis=True)

    if args.param_distr_plot:
        distrPlot(args.dataset,args.exp_id,args.param_distr_plot,plotScoreDis=False)

    if args.scatter_plot:
        scatterPlot(args.dataset,args.exp_id,args.scatter_plot)

    if args.plot_bias:
        plotBias(args.dataset,args.exp_id)
if __name__ == "__main__":
    main()
