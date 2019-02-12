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
import math
import scipy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

paramKeys = ["bias","incons","diffs","trueScores"]

#Plot the true scores on an axis
def plotTrueScores(exp_id,dataset,indList):

    plt.figure(figsize=(60,5))

    #Finding the index of the first video coming from each content
    refVid = np.genfromtxt("../data/{}_scores.csv".format(dataset))[1:,0]
    lastVid = None
    firstVidIndList = []
    for i,vid in enumerate(refVid):
        if (not lastVid) or  lastVid != vid:
            firstVidIndList.append(i)
        lastVid = vid

    '''
    #Artificial dataset
    if os.path.exists("../data/{}_trueScores.csv".format(dataset)):
        trueScores_gt = np.genfromtxt("../data/{}_trueScores.csv".format(dataset))

        scores = np.genfromtxt("../data/{}_scores.csv".format(dataset))[1:]
        print(firstVidIndList)
        refVidIndList = []
        for i in range(len(firstVidIndList)-1):
            refVidIndList.append(firstVidIndList[i]+np.argmax(trueScores_gt[firstVidIndList[i]:firstVidIndList[i+1]]))

        print(refVidIndList)

        plt.eventplot(trueScores_gt, orientation='horizontal', colors='b')
        plt.eventplot(trueScores_gt[refVidIndList], orientation='horizontal', colors='red')
    '''

    colors = cm.rainbow(np.linspace(0, 1,len(indList)))

    #Real dataset
    for i,modelInd in enumerate(indList):

        modelTtrueScorePath = sorted(glob.glob("../results/{}/model{}_epoch*_trueScores.csv".format(exp_id,modelInd)),key=findNumbers)[-1]
        plt.eventplot(np.genfromtxt(modelTtrueScorePath)[:,0], orientation='horizontal',color=colors[i])

    plt.savefig("../vis/{}/trueScores_{}_{}.png".format(exp_id,exp_id,indList))

def plotVideoRawScores(dataset,lineList,score_min,score_max):

    lineList = np.array(lineList).astype(int)

    lines = np.genfromtxt("../data/{}_scores.csv".format(dataset),dtype=str)[lineList]
    lineNames = lines[:,1]
    lines = lines[:,2:].astype(int)

    width = 1
    nb_lines = int(math.sqrt(len(lines)))

    plt.figure()
    #plt
    plt.subplots_adjust(hspace=0.3)

    for i,line in enumerate(lines):

        subplot = plt.subplot(nb_lines,nb_lines,i+1)
        bins = np.arange(score_min,score_max+1)-0.5*width

        subplot.hist(line, score_max-score_min+1,color='black',range=(score_min,score_max+1))
        subplot.set_xticks(np.arange(score_min,score_max+1)+0.5*width)
        subplot.set_xticklabels(np.arange(score_min,score_max+1).astype(str))
        subplot.set_title(lineNames[i])

    plt.savefig("../vis/{}_scores.png".format(dataset))

def agregate(pairList):

    nbAnnotAgreg = []
    ymeans = []
    yerr = []

    pairList = sorted(pairList,key=lambda x:x[1])
    err,nb_annot = zip(*pairList)

    errToAgreg = [err[0]]
    nbAnnotAgreg = [nb_annot[0]]
    for j in range(1,len(err)):
        if nb_annot[j] != nb_annot[j-1]:
            ymeans.append(np.array(errToAgreg).mean())
            yerr.append(1.96*np.array(errToAgreg).std()/np.sqrt(len(errToAgreg)))
            nbAnnotAgreg.append(nb_annot[j])
            errToAgreg = [err[j]]
        else:
            errToAgreg.append(err[j])

    ymeans.append(np.array(errToAgreg).mean())
    yerr.append(np.array(errToAgreg).std())
    #errToAgreg = [err[0]]
    #nbAnnotAgreg = [nb_annot[0]]

    return nbAnnotAgreg,ymeans,yerr

def baseLineError(datasetName,baseLineRefDict,baselineName):
    dataset,_ = load_data.loadData(datasetName)
    baseline,_ = computeBaselines(dataset,baselineName)

    error = np.sqrt(np.power(baseline-baseLineRefDict[baselineName],2).sum()/len(baseLineRefDict[baselineName]))

    return error

def readConfFile(path,keyList):

    conf = configparser.ConfigParser()
    conf.read(path)
    conf = conf["default"]
    resList = []
    for key in keyList:
        try:
            resList.append(conf[key])
        except KeyError:
            pass

    return resList

def convSpeed(exp_id,refModelIdList,refModelSeedList,varParamList):

    modelConfigPaths = sorted(glob.glob("../models/{}/model*.ini".format(exp_id)),key=findNumbers)
    modelIds = list(map(lambda x:findNumbers(os.path.basename(x)),modelConfigPaths))

    baselinesTypes = ['mos','sr_mos','zs_sr_mos']

    #Collect the scores of each reference model
    refTrueScoresDict = {}
    allBaseDict = {}
    for j,refModelId in enumerate(refModelIdList):
        refTrueScoresPath = sorted(glob.glob("../results/{}/model{}_epoch*_trueScores.csv".format(exp_id,refModelId)),key=findNumbers)[-1]
        refTrueScores = np.genfromtxt(refTrueScoresPath,delimiter="\t")[:,0]

        datasetName = readConfFile("../models/{}/model{}.ini".format(exp_id,refModelId),["dataset"])[0]

        paramValue = ''
        for varParam in varParamList:
            paramValue += " "+lookInModelAndData("../models/{}/model{}.ini".format(exp_id,refModelId),varParam,typeVal=str)
        #print(paramValue)
        dataset,_ = load_data.loadData(datasetName)
        #print(datasetName,paramValue)
        baseLineRefDict = {}
        for baselineType in baselinesTypes:
            baseLineRefDict[baselineType],_ = computeBaselines(dataset,baselineType)

        #Get the color for each baseline
        baseColMaps = cm.Blues(np.linspace(0, 1,int(1.5*len(baseLineRefDict.keys()))))
        baseColMapsDict = {}
        for i,key in enumerate(baseLineRefDict):
            baseColMapsDict[key] = baseColMaps[-i-1]

        #Collect the true scores of this reference model
        if not paramValue in refTrueScoresDict.keys():
            refTrueScoresDict[paramValue] = {}
        refTrueScoresDict[paramValue][refModelSeedList[j]] = refTrueScores


        if not refModelSeedList[j] in allBaseDict.keys():
            allBaseDict[refModelSeedList[j]] = baseLineRefDict


    errorArray = np.zeros(len(modelConfigPaths))
    nbAnnotArray = np.zeros(len(modelConfigPaths))

    #Store the error of each baseline method
    allErrorBaseDict = {}
    #for key in baseLineRefDict:
    #    errorArrayDict[key] = np.zeros(len(modelConfigPaths))

    paramValueList = []
    colorInds = []

    #Will contain a list of error for each value of the varying parameters
    valuesDict = {}
    baseDict = {}
    for i,modelPath in enumerate(modelConfigPaths):

        datasetName,modelId = readConfFile(modelPath,["dataset","ind_id"])

        paramValue = ''
        for varParam in varParamList:
            paramValue += " "+lookInModelAndData(modelPath,varParam,typeVal=str)

        if not paramValue in paramValueList:
            paramValueList.append(paramValue)

        colorInds.append(paramValueList.index(paramValue))

        nbAnnot,seed = readConfFile("../data/{}.ini".format(datasetName),["nb_annot","seed"])

        trueScoresPath = sorted(glob.glob("../results/{}/model{}_epoch*_trueScores.csv".format(exp_id,modelId)),key=findNumbers)[-1]
        trueScores = np.genfromtxt(trueScoresPath,delimiter="\t")[:,0]

        error = np.sqrt(np.power(trueScores-refTrueScoresDict[paramValue][int(seed)],2).sum()/len(refTrueScoresDict[paramValue][int(seed)]))

        if not paramValue in valuesDict.keys():
            valuesDict[paramValue] = [(error,nbAnnot)]
        else:
            valuesDict[paramValue].append((error,nbAnnot))


        for baselineType in baselinesTypes:

            if not baselineType in allErrorBaseDict.keys():
                allErrorBaseDict[baselineType] = {}

            if not nbAnnot in allErrorBaseDict[baselineType].keys():
                allErrorBaseDict[baselineType][nbAnnot] = {}

            if not int(seed) in allErrorBaseDict[baselineType][nbAnnot].keys():
                #Computing the baseline error relative to the right baseline
                error = baseLineError(datasetName,allBaseDict[int(seed)],baselineType)
                #print(baselineType,nbAnnot,int(seed))
                allErrorBaseDict[baselineType][nbAnnot][int(seed)] = error


    colors = cm.autumn(np.linspace(0, 1,len(paramValueList)))
    markers = [m for m, func in Line2D.markers.items() if func != 'nothing' and m not in Line2D.filled_markers]
    paramValueList = list(map(lambda x:paramValueList[x],colorInds))

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    plt.xlabel("Nb of annotators")
    plt.ylabel("RMSE")

    #Plot the models
    for i,paramValue in enumerate(valuesDict.keys()):
        nbAnnotAgreg,ymeans,yerr = agregate(valuesDict[paramValue])
        plt.errorbar(np.array(nbAnnotAgreg,dtype=str).astype(int)+0.1*i,ymeans,yerr=yerr,label=paramValue,marker=markers[i])


    #Plot the baselines
    for baseline in allErrorBaseDict.keys():

        means = np.zeros(len(allErrorBaseDict[baseline].keys()))
        stds = np.zeros(len(allErrorBaseDict[baseline].keys()))
        annotNbs = np.zeros(len(allErrorBaseDict[baseline].keys()))
        for i,annotNb in enumerate(allErrorBaseDict[baseline].keys()):
            valToAgr = np.array([allErrorBaseDict[baseline][annotNb][seed] for seed in allErrorBaseDict[baseline][annotNb].keys()])
            means[i],stds[i] = valToAgr.mean(),valToAgr.std()
            annotNbs[i] = annotNb

        #annotNbs = annotNbs.astype(int).astype(str)
        means,stds,annotNbs = zip(*sorted(zip(means,stds,annotNbs),key=lambda x:x[2]))

        plt.errorbar(annotNbs,means,yerr=stds,color=baseColMapsDict[baseline],label=baseline)

    fig.legend(loc='right')
    plt.savefig("../vis/{}/convSpeed_{}.png".format(exp_id,exp_id))

def lookInModelAndData(modelConfigPath,key,typeVal=float):

    if key != "nb_videos":
        res = readConfFile(modelConfigPath,[key])
        if len(res) == 0:
            datasetName = readConfFile(modelConfigPath,["dataset"])[0]
            value = readConfFile("../data/{}.ini".format(datasetName),[key])[0]
            return typeVal(value)
        else:
            return typeVal(res[0])
    else:
        datasetName = readConfFile(modelConfigPath,["dataset"])[0]
        nb_content,nb_video_per_content = readConfFile("../data/{}.ini".format(datasetName),["nb_content","nb_video_per_content"])
        return int(nb_content)*int(nb_video_per_content)

def distHeatMap(exp_id,params,param1,param2,minLog=0,maxLog=10,nbStep=100,nbEpochsMean=10):

    configFiles = sorted(glob.glob("../models/{}/model*.ini".format(exp_id)),key=findNumbers)

    colors = cm.plasma(np.linspace(0, 1,nbStep))

    for i,configFile in enumerate(configFiles):

        param1Value = lookInModelAndData(configFile,param1)
        param2Value = lookInModelAndData(configFile,param2)

        distFilePath = "../results/{}/model{}_dist.csv".format(exp_id,findNumbers(os.path.basename(configFile)))
        distFile = np.genfromtxt(distFilePath,delimiter=",",dtype=str)
        header = distFile[0]
        distFile = distFile[1:].astype(float) + 1e-9

        for j in range(distFile.shape[1]):

            if header[j] in params:
                plt.figure(j)
                plt.xlabel(param1)
                plt.ylabel(param2)

                neg_log_dist = -np.log10(distFile[-nbEpochsMean:,j].mean())
                color = colors[int(nbStep*neg_log_dist/maxLog)]

                plt.scatter(float(param1Value),float(param2Value),color=color,s=400)

                if i==len(configFiles)-1:
                    plt.savefig("../vis/{}/distHeatMap_{}.png".format(exp_id,header[j]))
                    plt.close()

def plot_points_arrows(repres,filePath,alphas,colors):

    plt.figure()

    #for i,point in enumerate(repres):
    #    if i<len(repres)-1:
    #        plt.plot([repres[i,0],repres[i+1,0]],[repres[i,1],repres[i+1,1]],alpha=alphas[i], zorder=1,color="black")

    plt.scatter(repres[:,0],repres[:,1],color=colors, zorder=2)

    plt.savefig(filePath)

def twoDimRepr(exp_id,model_id,start_epoch,paramPlot):

    def getEpoch(path):
        return findNumbers(os.path.basename(path).replace("model{}".format(model_id),""))

    for key in paramPlot:
        paramFiles = sorted(glob.glob("../results/{}/model{}_epoch*_{}.csv".format(exp_id,model_id,key)),key=findNumbers)

        paramFiles = list(filter(lambda x:getEpoch(x)>start_epoch,paramFiles))
        colors = cm.plasma(np.linspace(0, 1,len(paramFiles)))

        params = list(map(lambda x:np.genfromtxt(x)[:,0],paramFiles))
        alphas = np.power(np.arange(len(params))/len(params),4)

        repr_tsne = TSNE(n_components=2,init='pca',random_state=1,learning_rate=20).fit_transform(params)
        repr_pca = PCA(n_components=2).fit_transform(params)

        plot_points_arrows(repr_tsne,"../vis/{}/model{}_{}_tsne.png".format(exp_id,model_id,key),alphas,colors)
        plot_points_arrows(repr_pca,"../vis/{}/model{}_{}_pca.png".format(exp_id,model_id,key),alphas,colors)

def plotDistNLL(exp_id,ind_list,startEpoch,endEpoch):

    colors = cm.rainbow(np.linspace(0, 1, len(paramKeys)+1))

    markers = [m for m, func in Line2D.markers.items() if func != 'nothing' and m not in Line2D.filled_markers]
    if len(markers) < len(ind_list):
        raise ValueError("Too many model to plot : {}. {} is the maximum".format(nbPlot,len(markers)))
    else:
        markers = markers[:len(ind_list)]

    fig = plt.figure(figsize=(10,5))
    axDist = fig.add_subplot(111)
    axNLL = axDist.twinx()

    axDist.set_yscale('log')
    #axNLL.set_yscale("linear")
    minLL = None
    maxLL = None

    for i,ind in enumerate(ind_list):
        distArray = np.genfromtxt("../results/{}/model{}_dist.csv".format(exp_id,ind),delimiter=",",dtype=str)
        header = distArray[0]
        distArray = distArray[1:].astype(float)

        for j,key in enumerate(header):
            if key != "all":
                axDist.plot(distArray[startEpoch:endEpoch,j],label="model{} {}".format(ind,key),color=colors[j],marker=markers[i],alpha=0.5)

        if os.path.exists("../results/{}/model{}_nll.csv".format(exp_id,ind)):
            llArray = -np.genfromtxt("../results/{}/model{}_nll.csv".format(exp_id,ind),delimiter=",",dtype=float)

            if (minLL is None or llArray[-100:].min() < minNLL):
                minLL = llArray[-500:].min()
            if (maxLL is None or llArray[-100:].max() > maxNLl):
                maxLL = llArray[-500:].max()
            #axNLL.plot(llArray[startEpoch:endEpoch,j],label="model{}".format(ind),color="black",marker=markers[i],alpha=0.5)


    box = axDist.get_position()
    axDist.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axDist.set_xticks(np.arange(1,endEpoch-startEpoch,25))
    axDist.set_xticklabels(np.arange(startEpoch,endEpoch,25).astype(str))

    axNLL.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axNLL.set_ylim(bottom=minLL,top=maxLL)

    fig.legend(loc='right',prop={'size': 10})

    plt.savefig("../vis/{}/dist_nll_{}.png".format(exp_id,ind_list))

def distrPlot(exp_id,indModel,plotScoreDis,nbPlot=10,dx=0.01):

    modelConf = configparser.ConfigParser()
    modelConf.read("../models/{}/model{}.ini".format(exp_id,indModel))
    modelConf = modelConf['default']

    datasetName = readConfFile("../models/{}/model{}.ini".format(exp_id,indModel),["dataset"])[0]

    xInt,distorNbList = load_data.loadData(datasetName)

    #Building the model
    model = modelBuilder.modelMaker(xInt.size(1),len(xInt),distorNbList,int(modelConf["poly_deg"]),modelConf["score_dis"],\
                                    int(modelConf["score_min"]),int(modelConf["score_max"]),float(modelConf["div_beta_var"]), \
                                    int(modelConf["nb_freez_truescores"]),int(modelConf["nb_freez_bias"]),int(modelConf["nb_freez_diffs"]),\
                                    int(modelConf["nb_freez_incons"]))

    paramsPaths = sorted(glob.glob("../models/{}/model{}_epoch*".format(exp_id,indModel)))

    nb_annot,nb_content,nb_video_per_content = readConfFile("../data/{}.ini".format(datasetName),["nb_annot","nb_content","nb_video_per_content"])

    tensorDict = {"bias":np.zeros((len(paramsPaths),int(nb_annot))),\
                  "incons":np.zeros((len(paramsPaths),int(nb_annot))),\
                  "diffs":np.zeros((len(paramsPaths),int(nb_content))),\
                  "trueScores":np.zeros((len(paramsPaths),int(nb_video_per_content)*int(nb_content)))}

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
        cdfsList = torch.zeros((len(paramsPaths),nbPlot,int(1/dx)+1))
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
            subplot.hist(np.genfromtxt("../data/{}_{}.csv".format(datasetName,key)),range=xRangeDict[key],color="red")

            #Plot predicted distribution
            subplot = plt.subplot(2,1,2)
            subplot.set_xlim(xRangeDict[key])
            subplot.set_ylim(0,len(tensorDict[key][j])*0.5)
            subplot.hist(tensorDict[key][j],range=xRangeDict[key],color="blue")

            epoch = findNumbers(os.path.basename(paramsPaths[j]).replace("model{}".format(indModel),""))
            plt.savefig("../vis/{}/{}_{}_epoch{}.png".format(exp_id,key,indModel,epoch))
            plt.close()

def plotParam(dataset,exp_id,indList):

    paramsPaths = sorted(glob.glob("../models/{}/model{}_epoch*".format(exp_id,indList[0])))
    colors = cm.rainbow(np.linspace(0, 1,len(indList)))

    for key in paramKeys:

        tensor = np.zeros((len(indList),len(paramsPaths),len(np.genfromtxt("../results/{}/model{}_epoch0_{}.csv".format(exp_id,indList[0],key)))))

        for k,indModel in enumerate(indList):

            tensorPathList = sorted(glob.glob("../results/{}/model{}_epoch*_{}.csv".format(exp_id,indModel,key)),key=findNumbers)
            for i,tensorPath in enumerate(tensorPathList):
                tensor[k,i] = np.genfromtxt(tensorPath)[:,0].reshape(-1)

        xMin = np.min(tensor)
        xMax = np.max(tensor)

        xRangeDict = (xMin-0.1,xMax+0.1)

        maxErr = 0

        for k,indModel in enumerate(indList):

            paramsPaths = sorted(glob.glob("../models/{}/model{}_epoch*".format(exp_id,indModel)),key=findNumbers)

            #Plot the param error as a function of the absolute value of the param
            for j in range(len(tensor[k])):

                epoch = findNumbers(os.path.basename(paramsPaths[j]).replace("model{}".format(indModel),""))
                param_gt = np.genfromtxt("../data/{}_{}.csv".format(dataset,key))

                plt.figure(2*j,figsize=(10,5))
                #error = (np.abs(param_gt-tensor[k,j])/np.abs(param_gt))
                #error = (param_gt-tensor[k,j])/np.abs(param_gt)
                error = np.sqrt(np.power(param_gt-tensor[k,j],2)/len(param_gt))
                if np.max(error) > maxErr:
                    maxErr = np.max(error)

                plt.xlim(xRangeDict)
                plt.plot(tensor[k,j],error,"*",label=indModel,color=colors[k])

                x = np.arange(xRangeDict[0],xRangeDict[1],0.01)
                plt.plot(x,np.zeros_like(x))
                if k==len(indList)-1:
                    plt.ylim(0,maxErr)
                    plt.legend()
                    plt.savefig("../vis/{}/model{}_{}Error_epoch{}.png".format(exp_id,indList,key,epoch))
                    plt.close()

                plt.figure(2*j+1,figsize=(10,5))
                plt.plot(param_gt,tensor[k,j],"*",label=indModel,color=colors[k])
                x = np.arange(xRangeDict[0],xRangeDict[1],0.01)
                plt.plot(x,x)

                plt.xlim(xRangeDict)
                plt.ylim(xRangeDict)

                if k==len(indList)-1:
                    plt.legend()
                    plt.savefig("../vis/{}/model{}_{}VSgt_epoch{}_.png".format(exp_id,indList,key,epoch))
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
        trueScores = np.genfromtxt(trueScoresPath)
        epoch = findNumbers(os.path.basename(trueScoresPath).replace("model{}".format(indModel),""))
        plt.plot(trueScores[:,0],trueScores_gt,"*",label="epoch{}".format(epoch),color=colors[i])

    plt.legend()
    plt.savefig("../vis/{}/scatterPlot_{}.png".format(exp_id,indModel))

def fakeDataDIstr(args):

    dataConf = configparser.ConfigParser()
    dataConf.read("../data/{}.ini".format(args.dataset))
    dataConf = dataConf['default']

    paramKeys = ["trueScores","diffs","incons","bias"]
    dx = 0.01


    dist_dic = {"trueScores":lambda x:torch.exp(Uniform(1,5).log_prob(x)),\
                "diffs":lambda x:torch.exp(Beta(float(dataConf['diff_alpha']), float(dataConf["diff_beta"])).log_prob(x)), \
                "incons":lambda x:torch.exp(Beta(float(dataConf["incons_alpha"]), float(dataConf["incons_beta"])).log_prob(x)),\
                "bias":lambda x:torch.exp(Normal(torch.zeros(1), float(dataConf["bias_std"])*torch.eye(1)).log_prob(x))}

    range_dic = {"trueScores":torch.arange(1,5,dx),\
                "diffs":torch.arange(0,1,dx), \
                "incons":torch.arange(0,1,dx),\
                "bias":torch.arange(-3*float(dataConf["bias_std"]),3*float(dataConf["bias_std"]),dx)}

    for i,paramName in enumerate(paramKeys):

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

def computeBaselines(trainSet,baselineName):

    if baselineName == "mos":
        mean,std = modelBuilder.MOS(trainSet,sub_rej=False,z_score=False)
    elif baselineName == "sr_mos":
        mean,std = modelBuilder.MOS(trainSet,sub_rej=True,z_score=False)
    elif baselineName == "zs_sr_mos":
        mean,std = modelBuilder.MOS(trainSet,sub_rej=True,z_score=True)

    return mean,std

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

def computeConfInter(loss,tensor):

    hessDiag,grad = train_val.computeHessDiag(loss,tensor)

    confInter = 1.96/torch.sqrt(hessDiag)

    return confInter

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

def error(dictPred,dictGT,paramNames,errFunc):

    errList = []

    for name in paramNames:

        errList.append(errorVec(dictPred[name][:,0],dictGT[name],errFunc))

    return errList

def relative(vec,vec_ref):
    return (np.abs(vec_ref-vec)/np.abs(vec_ref)).mean()

def rmse(vec,vec_ref):

    return np.sqrt(np.power(vec_ref-vec,2).sum()/len(vec))

def errorVec(vec,vec_ref,errFunc):

    if not (type(vec) is np.ndarray):
        if vec.is_cuda:
            vec = vec.cpu()
        vec = vec.detach().numpy()
    if not (type(vec_ref) is np.ndarray):
        if vec_ref.is_cuda:
            vec_ref = vec_ref.cpu()
        vec_ref = vec_ref.detach().numpy()

    if len(vec) != len(vec_ref):
        return -1
    else:
        return errFunc(vec,vec_ref)

def includPerc(dictPred,dictGT,paramNames):

    inclList = []

    for name in paramNames:
        inclList.append(includPercVec(dictPred[name][:,0],dictPred[name][:,1],dictGT[name]))

    return inclList

def includPercVec(mean,confIterv,gt):

    if len(mean) != len(gt):
        return -1
    else:

        includNb = ((mean - confIterv < gt)* (gt < mean + confIterv)).sum()
        return includNb/len(mean)

def extractParamName(path):

    path = os.path.basename(path).replace(".csv","")
    return path[path.find("_")+1:]

def getGT(dataset,gtParamDict,paramKeys):

    if not dataset in gtParamDict.keys():
        gtParamDict[dataset] = {}

        for param in paramKeys:
            gtParamDict[dataset][param] = np.genfromtxt("../data/{}_{}.csv".format(dataset,param))

    return gtParamDict[dataset]

def compareWithGroundTruth(exp_id,varParams,error_metric):

    errFunc = globals()[error_metric]

    allGtParamDict = {}
    #for paramName in paramKeys:
    #    gtParamDict[paramName] = np.genfromtxt("../data/{}_{}.csv".format(dataset,paramName))

    paramsDicList = []
    modelConfigPaths = sorted(glob.glob("../models/{}/*.ini".format(exp_id)),key=findNumbers)
    csvHead = "{}".format(varParams)+"".join(["\t{}".format(paramKeys[i]) for i in range(len(paramKeys))])

    csvErr = ""
    csvInclu = ""

    for i in range(len(modelConfigPaths)):

        modelInd = findNumbers(os.path.basename(modelConfigPaths[i]))

        dataset_id = readConfFile(modelConfigPaths[i],["dataset"])[0]

        paramValue = ''
        for varParam in varParams:
            paramValue += ","+lookInModelAndData(modelConfigPaths[i],varParam,typeVal=str)

        #Get the ground truth of the dataset on which this model has been trained
        gtParamDict = getGT(dataset_id,allGtParamDict,paramKeys)

        paramDict = {}
        for paramName in paramKeys:
            lastEpochPath = sorted(glob.glob("../results/{}/model{}_epoch*_{}.csv".format(exp_id,modelInd,paramName)),key=findNumbers)[-1]

            lastEpoch = findNumbers(os.path.basename(lastEpochPath).replace("model{}".format(modelInd),""))
            paramDict[paramName] = np.genfromtxt("../results/{}/model{}_epoch{}_{}.csv".format(exp_id,modelInd,lastEpoch,paramName))

        errors = error(paramDict,gtParamDict,paramKeys,errFunc)
        csvErr += "{}".format(paramValue)+"".join(["\t{}".format(round(100*errors[i],2)) for i in range(len(errors))])+"\n"

        incPer = includPerc(paramDict,gtParamDict,paramKeys)
        csvInclu += "{}".format(paramValue)+"".join(["\t{}".format(round(100*incPer[i],2)) for i in range(len(incPer))])+"\n"

    with open("../results/{}/err.csv".format(exp_id),"w") as text_file:
        print(csvHead,file=text_file)
        print(csvErr,file=text_file,end="")

    with open("../results/{}/inclPerc.csv".format(exp_id),"w") as text_file:
        print(csvHead,file=text_file)
        print(csvInclu,file=text_file)

def agregateCpWGroundTruth(exp_id,resFilePath):

    #Agregating the results
    resFile = np.genfromtxt(resFilePath,delimiter="\t",dtype="str")
    header = resFile[0]
    resFile = resFile[1:]

    #Grouping the lines using the value of the first column
    groupedLines = {}
    for line in resFile:
        if line[0] in groupedLines.keys():
            groupedLines[line[0]].append(line)
        else:
            groupedLines[line[0]] = [line]

    csv = "\t".join(header)+"\n"
    mean = np.zeros((len(groupedLines.keys()),resFile.shape[1]-1))
    std =  np.zeros((len(groupedLines.keys()),resFile.shape[1]-1))

    keys = sorted(groupedLines.keys())

    for i,key in enumerate(keys):
        groupedLines[key] = np.array(groupedLines[key])[:,1:].astype(float)

        mean[i] =  groupedLines[key].mean(axis=0)
        std[i] = groupedLines[key].std(axis=0)

        csv += key
        for j in range(len(mean[0])):
            csv += "\t"+str(round(mean[i,j],2))+"\pm"+str(round(std[i,j],2))

        csv += "\n"

    with open(resFilePath.replace(".csv","_agreg.csv"),"w") as text_file:
        print(csv,file=text_file)

    #Plot the agregated results
    fig = plt.figure()
    plt.subplots_adjust(bottom=0.2)
    #plt.tight_layout()
    ax = fig.add_subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    for i in range(len(mean)):
        ax.bar(np.arange(resFile.shape[1]-1)+0.1*i,mean[i],width=0.1,label=keys[i],yerr=std[i])

    imageName = os.path.basename(resFilePath).replace(".csv",".png")
    fig.legend(loc='right')
    plt.ylim(0,60)
    plt.ylabel("RMSE")
    plt.gca().set_ylim(bottom=0)
    plt.xticks(np.arange(resFile.shape[1]-1),header[1:],rotation=45,horizontalalignment="right")
    plt.savefig("../vis/{}/{}".format(exp_id,imageName))

    #Computing the t-test

    #The header first element is the list of varying parameters and we only want the
    #parameters name
    header = header[1:]
    ttest_matrix(groupedLines,exp_id,header,resFilePath)

def ttest_matrix(groupedLines,exp_id,modelKeys,resFilePath):

    keys = sorted(list(groupedLines.keys()))

    mat = np.zeros((len(keys),len(keys),len(modelKeys)))

    for i in range(len(keys)):
        for j in range(len(keys)):
            for k in range(len(modelKeys)):

                _,mat[i,j,k] = scipy.stats.ttest_ind(groupedLines[keys[i]][:,k],groupedLines[keys[j]][:,k],equal_var=True)

    mat = mat.astype(str)
    for k in range(len(modelKeys)):
        csv = "\t"+"\t".join(keys)+"\n"
        for i in range(len(keys)):
            csv += keys[i]+"\t"+"\t".join(mat[i,:,k])+"\n"

        resFileName = os.path.basename(resFilePath).replace(".csv","")
        with open("../results/{}/ttest_{}_{}.csv".format(exp_id,modelKeys[k],resFileName),"w") as text_file:
            print(csv,file=text_file)

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
                                    be the list of parameters varying across the different models in the experiment.')
    argreader.parser.add_argument('--comp_gt_agr',type=str,nargs="*",metavar='PARAM',help='To compare the parameters found with the ground truth parameters. Require a fake dataset. The argument should\
                                    be the list of parameters varying across the different models in the experiment. The accuracies of models having the same value for those parameters will be agregated.')
    argreader.parser.add_argument('--error_metric',type=str,metavar='ERROR',default="rmse",help='The error metric used in \'--comp_gt\' and \'--comp_gt_agr\'. Can be \'rmse\' or \'relative\'. Default is \'RMSE\'.')

    argreader.parser.add_argument('--artif_data',action='store_true',help='To plot the real and empirical distribution of the parameters of a fake dataset. \
                                    The fake dataset to plot is set by the --dataset argument')
    argreader.parser.add_argument('--distr_plot',type=int,help='To plot the distribution of each score for a given model. The argument value is the model id. \
                                    The first argument should be the model id')
    argreader.parser.add_argument('--param_distr_plot',type=int,help='Like --distr_plot but only plot the parameters distribution and not the scores distributions.')
    argreader.parser.add_argument('--scatter_plot',type=int,help='To plot the real and predicted scores of a fake dataset in a 2D plot. \
                                    The first argument should be the model id')
    argreader.parser.add_argument('--plot_param',type=str,nargs="*",help='To plot the error of every parameters at each epoch for each model. The argument values are the index of the models to plot.')
    argreader.parser.add_argument('--plot_dist_nll',type=int,nargs="*",help='To plot the distance travelled by each parameters and the negative log-likelihood at each epoch. \
                                    The argument values are the index of the models to plot. The two last arguments are the epochs at which to start and finish the plot.')

    argreader.parser.add_argument('--two_dim_repr',type=str,nargs="*",help='To plot the t-sne visualisation of the values taken by the parameters during training. \
                                    The first argument value is the id of the model to plot and the second is the start epoch. The following argument are the parameters to plot.')

    argreader.parser.add_argument('--dist_heatmap',type=str,nargs="*",help='To plot the average distance travelled by parameters at the end of training for each model. The value of this argument is a list\
                                    of parameters to plot and the two last value are the parameters to plot.')

    argreader.parser.add_argument('--conv_speed',type=str,nargs='*',metavar='ID',help='To plot the error as a function of the number of annotator. The value is a list of parameters varying between \
                                    the reference models.')

    argreader.parser.add_argument('--plot_video_raw_scores',type=str,nargs='*',metavar='ID',help='To plot histograms of scores for some videos of a dataset. The value of this argument is the list of videos \
                                    line index to plot. The dataset should also be indicated with the dataset argument')

    argreader.parser.add_argument('--plot_true_scores',type=str,nargs='*',metavar='ID',help='To plot the true scores found by several models on an axis. The argument value is the list of model indexs to plot.')


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
        compareWithGroundTruth(args.exp_id,args.comp_gt,args.error_metric)
    if args.comp_gt_agr:
        compareWithGroundTruth(args.exp_id,args.comp_gt_agr,args.error_metric)
        agregateCpWGroundTruth(args.exp_id,"../results/{}/err.csv".format(args.exp_id))
        agregateCpWGroundTruth(args.exp_id,"../results/{}/inclPerc.csv".format(args.exp_id))

    if args.artif_data:
        fakeDataDIstr(args)

    if args.distr_plot:
        distrPlot(args.exp_id,args.distr_plot,plotScoreDis=True)

    if args.param_distr_plot:
        distrPlot(args.exp_id,args.param_distr_plot,plotScoreDis=False)

    if args.scatter_plot:
        scatterPlot(args.dataset,args.exp_id,args.scatter_plot)

    if args.plot_param:
        plotParam(args.dataset,args.exp_id,args.plot_param)

    if args.plot_dist_nll:
        plotDistNLL(args.exp_id,args.plot_dist_nll[:len(args.plot_dist_nll)-2],args.plot_dist_nll[-2],args.plot_dist_nll[-1])

    if args.two_dim_repr:
        twoDimRepr(args.exp_id,int(args.two_dim_repr[0]),int(args.two_dim_repr[1]),args.two_dim_repr[2:])

    if args.dist_heatmap:
        distHeatMap(args.exp_id,args.dist_heatmap[:-2],param1=args.dist_heatmap[-2],param2=args.dist_heatmap[-1])

    if args.conv_speed:

        configFiles = glob.glob("../models/{}/model*.ini".format(args.exp_id))

        def getProp(x):
            datasetName = readConfFile(x,["dataset"])[0]
            seed,nb_annot = readConfFile("../data/{}.ini".format(datasetName),["seed","nb_annot"])
            return int(seed),int(nb_annot)

        ids = list(map(lambda x:findNumbers(os.path.basename(x)),configFiles))
        seeds,nb_annots = zip(*list(map(getProp,configFiles)))

        ids_seeds_nbAnnots = zip(ids,seeds,nb_annots)

        argmaxs = np.argwhere(nb_annots == np.amax(nb_annots)).flatten()

        ids = np.array(ids)[argmaxs]
        seeds = np.array(seeds)[argmaxs]

        ids,seeds = zip(*sorted(zip(ids,seeds),key=lambda x:x[0]))

        convSpeed(args.exp_id,ids,seeds,args.conv_speed)

    if args.plot_video_raw_scores:
        plotVideoRawScores(args.dataset,args.plot_video_raw_scores,args.score_min,args.score_max)

    if args.plot_true_scores:

        plotTrueScores(args.exp_id,args.dataset,args.plot_true_scores)

if __name__ == "__main__":
    main()
