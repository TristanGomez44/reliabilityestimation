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
import train
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
baselinesTypes = ['mos','sr_mos','zs_sr_mos']

def plotVideoRawScores(dataset,videoList,score_min,score_max):
    ''' Plot histograms of videos raw scores
    Args:
        dataset (str): the name of the dataset
        videoList (list): the list of videos index to plot the raw scores of
        score_min (int): the minimum score that can be given
        score_max (int): the maximum score that can be given
    '''

    videoList = np.array(videoList).astype(int)

    lines = np.genfromtxt("../data/{}_scores.csv".format(dataset),dtype=str)[videoList]
    lineNames = lines[:,1]
    lines = lines[:,2:].astype(int)

    width = 1
    nb_lines = int(math.sqrt(len(lines)))

    plt.figure()

    plt.subplots_adjust(hspace=0.6,wspace = 0.4)

    for i,line in enumerate(lines):

        subplot = plt.subplot(nb_lines,nb_lines,i+1)
        bins = np.arange(score_min,score_max+1)-0.5*width

        subplot.hist(line, score_max-score_min+1,color='black',range=(score_min,score_max+1))
        subplot.set_xticks(np.arange(score_min,score_max+1)+0.5*width)
        subplot.set_xticklabels(np.arange(score_min,score_max+1).astype(str))
        subplot.set_title(lineNames[i].replace(".yuv",""))
        subplot.set_ylabel("Empirical count")
        subplot.set_xlabel("Raw scores")

    plt.savefig("../vis/{}_scores.png".format(dataset))

def agregate(pairList):
    ''' Takes a list of pairs (number of annotators, error) and agregate the pairs having the same number of annotators

    The aggregation process consists to computing the mean and the variance of all pairs with same nb of annotators.

    Args:
        pairList (list): a list of pairs (number of annotators, error) giving the error of several models with several annotator number
    Returns
        The list of distinct annotator number values met during the list parsing
        The list of error mean for each of those distinct values
        The list of error variance for each of those distinct values
    '''

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

    return nbAnnotAgreg,ymeans,yerr

def baseLineError(datasetName,baseLineRefDict,baselineName):
    ''' Compute baselines method and evaluate them by comparing them to some reference

    Args:
        datasetName (str): the name of the dataset to compute baselines on
        baseLineRefDict (dict): a dictionnary containing one reference vector for each type of baseline
        baselineName (str): the type of baseline desired (can be 'mos','sr_mos' or 'zs_sr_mos'.)
    Returns:
        the error made by the chosen baseline method on the chosen dataset relative to the reference
    '''

    dataset,_ = load_data.loadData(datasetName)
    baseline,_ = computeBaselines(dataset,baselineName)

    error = np.sqrt(np.power(baseline-baseLineRefDict[baselineName],2).sum()/len(baseLineRefDict[baselineName]))

    return error

def readConfFile(path,keyList):
    ''' Read a config file and get the value of desired argument

    Args:
        path (str): the path to the config file
        keyList (list): the list of argument to read name)
    Returns:
        the argument value, in the same order as in keyList
    '''

    conf = configparser.ConfigParser()
    conf.read(path)
    conf = conf["default"]
    resList = []
    for key in keyList:
        resList.append(conf[key])

    return resList

def convSpeed(exp_id,refModelIdList,refModelSeedList,varParamList):
    ''' Plot the distance between the vector found by models on subdatasets and the vector found by those same models on the full datasets

    The distance is plot as a function of the annotator number.
    Some parameters can be varying among models (e.g. the score distribution, the learning rate). One curve is draws for each combination of those parameters.

    Args:
        exp_id (str): the experience name
        refModelIdList (list): the list of ids of models which are trained on a full dataset
        refModelSeedList (list): the list of seeds of models which are trained on a full dataset
        varParamList (list): the list of parameters which are varying among all the models in the experiment
    '''

    modelConfigPaths = sorted(glob.glob("../models/{}/model*.ini".format(exp_id)),key=findNumbers)
    modelIds = list(map(lambda x:findNumbers(os.path.basename(x)),modelConfigPaths))

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
    ''' Look for a parameter if the model config file and in the dataset config file

    If the parameters is not found in the model config file, its searches in the config file of the model dataset
    Args:
        modelConfigPath (str): the path to the config file of the model
        key (str): the name of the parameter to find
        typeVal (class): the type of the parameter
    Returns:
        the value of the parameter
    '''

    if key != "nb_videos":

        try:
            res = readConfFile(modelConfigPath,[key])
            return typeVal(res[0])
        except KeyError:
            datasetName = readConfFile(modelConfigPath,["dataset"])[0]
            value = readConfFile("../data/{}.ini".format(datasetName),[key])[0]
            return typeVal(value)

    else:
        datasetName = readConfFile(modelConfigPath,["dataset"])[0]
        nb_content,nb_video_per_content = readConfFile("../data/{}.ini".format(datasetName),["nb_content","nb_video_per_content"])
        return int(nb_content)*int(nb_video_per_content)

def twoDimRepr(exp_id,model_id,start_epoch,paramPlot,plotRange):
    ''' Plot parameters trajectory across training epochs of a model.

    The trajectories are ploted using t-sne and PCA representation

    Args:
        exp_id (str): the experience name
        model_id (str): the model id
        start_epoch (int): the epoch at which to start the plot
        paramPlot (list): the parameters to plot (can be \'trueScores\', \'bias\', \'incons\' or \'diffs\'.)
        plotRange (list): the axis limits to use. It should be a list of values like this (xMin,xMax,yMin,yMax).
    '''

    def getEpoch(path):
        return findNumbers(os.path.basename(path).replace("model{}".format(model_id),""))

    cleanNamesDict = {"trueScores":"True Scores","bias":"Biases","diffs":"Ambiguities","incons":"Inconsistencies"}

    for key in paramPlot:
        paramFiles = sorted(glob.glob("../results/{}/model{}_epoch*_{}.csv".format(exp_id,model_id,key)),key=findNumbers)

        paramFiles = list(filter(lambda x:getEpoch(x)>start_epoch,paramFiles))
        colors = cm.plasma(np.linspace(0, 1,len(paramFiles)))

        params = list(map(lambda x:np.genfromtxt(x)[:,0],paramFiles))

        repr_tsne = TSNE(n_components=2,init='pca',random_state=1,learning_rate=20).fit_transform(params)
        repr_pca = PCA(n_components=2).fit_transform(params)

        plt.figure()
        plt.title("model "+str(model_id)+" : "+cleanNamesDict[key])
        plt.scatter(repr_tsne[:,0],repr_tsne[:,1],color=colors, zorder=2)
        plt.savefig("../vis/{}/model{}_{}_tsne.png".format(exp_id,model_id,key))

        plt.figure()
        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")
        plt.title("model "+str(model_id)+" : "+cleanNamesDict[key])
        plt.scatter(repr_pca[:,0],repr_pca[:,1],color=colors, zorder=2)
        if plotRange:
            plt.xlim(plotRange[0],plotRange[1])
            plt.ylim(plotRange[2],plotRange[3])
        plt.savefig("../vis/{}/model{}_{}_pca.png".format(exp_id,model_id,key))

def plotDist(exp_id,model_id,startEpoch,endEpoch,plotRange):
    ''' Plot the distance traveled by model parameters across epochs.

    Args:
        exp_id (str): the experience name
        model_id (str): the model id to plot
        startEpoch (int): the epoch at which to start the plot
        endEpoch (int): the epoch at which to end the plot
        plotRange (list): the y-axis limits to use. It should be a pair of values like this (yMin,yMax).
    '''

    colors = cm.rainbow(np.linspace(0, 1, len(paramKeys)+1))

    cleanNameDict = {"trueScores":"True scores","bias":"Biases","incons":"Inconsistencies","diffs":"Difficulties"}

    fig = plt.figure(figsize=(10,5))
    axDist = fig.add_subplot(111)

    axDist.set_yscale('log')

    distArray = np.genfromtxt("../results/{}/model{}_dist.csv".format(exp_id,model_id),delimiter=",",dtype=str)
    header = distArray[0]
    distArray = distArray[1:].astype(float)

    for j,key in enumerate(header):
        if key != "all":
            axDist.plot(distArray[startEpoch:endEpoch,j],label="{}".format(cleanNameDict[key]),color=colors[j],alpha=0.5)

    box = axDist.get_position()
    axDist.set_xlabel("Epochs")
    axDist.set_ylabel("Euclidean distance")
    axDist.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axDist.set_xticks(np.arange(1,endEpoch-startEpoch,25))
    axDist.set_xticklabels(np.arange(startEpoch,endEpoch,25).astype(str))
    axDist.set_ylim(plotRange)

    fig.legend(loc='right',prop={'size': 15})

    plt.savefig("../vis/{}/dist_{}.png".format(exp_id,model_id))

def plotParam(dataset,exp_id,indList,labels):
    ''' Plot the parameters found by models against a the ground truth parameters

    It produces a scatter plot where the x-axis represents the ground-truth values and the y-axis represents the value found by models.
    This plot is produced for every epoch for which the model had its parameters saved during training

    Args:
        dataset (str): the dataset name. The dataset should be artificial because this plot requires ground truth parameters.
        exp_id (str): the experience name.
        indList (list): the list of model ids to plot
    '''

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

        if not labels:
            labels = indList

        for k,indModel in enumerate(indList):

            paramsPaths = sorted(glob.glob("../models/{}/model{}_epoch*".format(exp_id,indModel)),key=findNumbers)
            #print(k,indModel,len(paramsPaths),tensor.shape)
            #for path in paramsPaths:
            #    print(path)
            #Plot the param error as a function of the absolute value of the param
            for j in range(len(tensor[k])):

                epoch = findNumbers(os.path.basename(paramsPaths[j]).replace("model{}".format(indModel),""))
                param_gt = np.genfromtxt("../data/{}_{}.csv".format(dataset,key))

                plt.figure(j,figsize=(10,5))
                plt.plot(param_gt,tensor[k,j],"*",label=labels[k],color=colors[k])
                x = np.arange(xRangeDict[0],xRangeDict[1],0.01)
                plt.plot(x,x)

                plt.xlabel("Ground truth")
                plt.ylabel("Estimation")
                plt.xlim(xRangeDict)
                plt.ylim(xRangeDict)

                if k==len(indList)-1:
                    plt.legend()
                    plt.savefig("../vis/{}/model{}_{}VSgt_epoch{}_.png".format(exp_id,indList,key,epoch))
                    plt.close()

def fakeDataDIstr(args):
    ''' Plot empirical and real distribution of an artificial dataset ground-truth parameters.

    For each of the four parameter vector (true scores, biases, inconsistencies and difficulties),
    it plots the real density and the emirical distribution of the same figure.

    Args:
        args (namespace): the namespace collected in the begining of the script containing all arguments about model training and evaluation.
    '''

    dataConf = configparser.ConfigParser()
    dataConf.read("../data/{}.ini".format(args.dataset))
    dataConf = dataConf['default']

    paramKeys = ["trueScores","diffs","incons","bias"]

    cleanNames = ["True Scores","Difficulties","Inconsistencies","Biases"]
    dx = 0.05
    xLim = 0.4

    dist_dic = {"trueScores":lambda x:torch.exp(Uniform(1,5).log_prob(x)),\
                "diffs":lambda x:torch.exp(Beta(float(dataConf['diff_alpha']), float(dataConf["diff_beta"])).log_prob(x)), \
                "incons":lambda x:torch.exp(Beta(float(dataConf["incons_alpha"]), float(dataConf["incons_beta"])).log_prob(x)),\
                "bias":lambda x:torch.exp(Normal(torch.zeros(1), float(dataConf["bias_std"])*torch.eye(1)).log_prob(x))}

    range_dic = {"trueScores":torch.arange(1,5,dx),\
                "diffs":torch.arange(0,1,dx), \
                "incons":torch.arange(0,1,dx),\
                "bias":torch.arange(-3*float(dataConf["bias_std"]),3*float(dataConf["bias_std"]),dx)}

    for i,paramName in enumerate(paramKeys):

        paramValues = np.genfromtxt("../data/{}_{}.csv".format(dataConf["dataset_id"],paramName))
        trueCDF = dist_dic[paramName](range_dic[paramName]).numpy().reshape(-1)

        fig,empiCountAx = plt.subplots()

        plt.title(cleanNames[i])
        plt.ylabel("Density")

        plt.xlabel("Value")

        handles = []
        distAx = empiCountAx.twinx()

        if paramName == "incons" or paramName == "diffs":
            empiCountAx.hist(paramValues,10,label="Empirical distribution",color="orange",range=[0,xLim])
            #plt.xlim(0,xLim)
        else:
            empiCountAx.hist(paramValues,10,label="Empirical distribution",color="orange")

        handles += distAx.plot(range_dic[paramName].numpy(),trueCDF,label="True distribution",color="blue")
        print(trueCDF.max())
        #distAx.set_ylim(0,trueCDF.max())

        leg = plt.legend(handles=handles,title="Test")

        plt.gca().add_artist(leg)
        plt.savefig("../vis/{}_{}_dis.png".format(args.dataset,paramName))

def computeBaselines(scoreMat,baselineName):
    ''' Run one baseline method on some dataset and returns the true scores vector produced

    Args:
        scoreMat (torch.tensor): the score matrix
        baselineName (str): the name of the baseline to compute
    Returns
        the true score vector found by the baselines and the corresponding confidence interval
    '''

    if baselineName == "mos":
        value,conf = modelBuilder.MOS(scoreMat,sub_rej=False,z_score=False)
    elif baselineName == "sr_mos":
        value,conf = modelBuilder.MOS(scoreMat,sub_rej=True,z_score=False)
    elif baselineName == "zs_sr_mos":
        value,conf = modelBuilder.MOS(scoreMat,sub_rej=True,z_score=True)

    return value,conf

def computeConfInter(loss,vector):
    ''' Computes the confidence interval of a vector values relative to the loss.
    Args:
        loss (torch.Tensor): the loss (used to compute confidence interval)
        vector (torch.tensor): a vector tensor. One confidence interval will be computed on each element of the vector.
    Returns:
        a vector of confidence intervals
    '''

    hessDiag,grad = train.computeHessDiag(loss,vector)

    confInter = 1.96/torch.sqrt(hessDiag)

    return confInter

def error(dictPred,dictGT,paramNames,errFunc):
    ''' Compute the error made on the 4 parameter vectors
    Args:
        dictPred (dict): a dictionnary containing the four vector found by estimation
        dictGT (dict): a dictionnary containing the four ground-truth parameters
         paramNames (list): the list of the four parameter vectors name
         errFunc (function): the function to use to compute the error (can be \'relative\' or \'rmse\')
    Returns:
        the list of error for each vector in the same order as in paramNames
    '''

    errList = []

    for name in paramNames:

        errList.append(errorVec(dictPred[name][:,0],dictGT[name],errFunc))

    return errList

def relative(vec,vec_ref):
    ''' Compute the relative error between two vector
    Args:
        vec (torc.tensor, numpy.array): the vector found by estimation
        vec_ref (torch.tensor,numpy.array): the ground-truth vector
    Returns:
        the error value
    '''

    return (np.abs(vec_ref-vec)/np.abs(vec_ref)).mean()

def rmse(vec,vec_ref):
    ''' Compute the rmse error between two vector
    Args:
        vec (torc.tensor, numpy.array): the vector found by estimation
        vec_ref (torch.tensor,numpy.array): the ground-truth vector
    Returns:
        the error value
    '''

    return np.sqrt(np.power(vec_ref-vec,2).sum()/len(vec))

def errorVec(vec,vec_ref,errFunc):
    ''' Compute the error between two vectors

    This function takes care that the vector have the correct type on the correct hardware (cpu and not gpu).

    Args:
        vec (torc.tensor, numpy.array): the vector found by estimation
        vec_ref (torch.tensor,numpy.array): the ground-truth vector
        errFunc (function): the function to use to compute the error (can be \'relative\' or \'rmse\')
    Returns:
        the error value
    '''

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
    ''' Compute the proportion of ground-truth vector included in the confidence interval of estimation for the four vectors
    Args:
        dictPred (dict): a dictionnary containing the four vector found by estimation
        dictGT (dict): a dictionnary containing the four ground-truth parameters
         paramNames (list): the list of the four parameter vectors name
    Returns:
        the list of vector proportion included in the confidence interval for each vector in the same order as in paramNames
    '''

    inclList = []

    for name in paramNames:
        inclList.append(includPercVec(dictPred[name][:,0],dictPred[name][:,1],dictGT[name]))

    return inclList

def includPercVec(mean,confIterv,gt):
    ''' Compute the proportion of ground-truth vector included in the confidence interval of estimation for one vector

    Args:
        vec (torc.tensor, numpy.array): the vector found by estimation
        vec_ref (torch.tensor,numpy.array): the ground-truth vector
        errFunc (function): the function to use to compute the error (can be \'relative\' or \'rmse\')
    Returns:
        the error value
    '''

    includNb = ((mean - confIterv < gt)* (gt < mean + confIterv)).sum()
    return includNb/len(mean)

def extractParamName(path):
    ''' Extract a parameter name from a csv path '''

    path = os.path.basename(path).replace(".csv","")
    return path[path.find("_")+1:]

def getGT(dataset,gtParamDict,paramKeys):
    ''' Read the dictionnary of ground-truth parameters for one dataset
    Args:
        dataset (str): the dataset name to read groundtruth of
        gtParamDict (dict): the dictionnary containing previously read groundtruth dictionnaries
        paramKeys (list): the list of vector parameters name (i.e. "trueScores", "bias", "diffs" and "incons")
    Returns:
        the dictionnary of ground truth parameters for the desired dataset
    '''

    #If the dataset ground truth have not been previously read
    if not dataset in gtParamDict.keys():
        gtParamDict[dataset] = {}

        for param in paramKeys:
            gtParamDict[dataset][param] = np.genfromtxt("../data/{}_{}.csv".format(dataset,param))

    return gtParamDict[dataset]

def getEpoch(paths,epoch):

    if epoch == -1:
        return sorted(paths,key=findNumbers)[-1]
    else:
        epochNotFound = True
        i=0
        while epochNotFound and i < len(paths):

            if paths[i].find("epoch{}_".format(epoch)) != -1:
                epochNotFound = False

            i += 1

        if epochNotFound:
            raise ValueError("Epoch file not found for epoch",epoch,"in files : ",paths)
        else:
            return paths[i-1]

def compareWithGroundTruth(exp_id,varParams,error_metric,epoch=-1):
    ''' Compare the parameters found by several models to the ground truth

    This function compute the error between parameter found and ground-truth but it also
    compute the proportion of groundtruth parameters included in the confidence interval of estimated value

    Args:
        exp_id (str): the name of the experience
        varParams (list): the list of meta-parameters varying during the experiment
        error_metric (str): the name of the function to use to compute the error (can be \'rmse\' or \'relative\')
    '''

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

            lastEpochPath = getEpoch(glob.glob("../results/{}/model{}_epoch*_{}.csv".format(exp_id,modelInd,paramName)),epoch)

            lastEpoch = findNumbers(os.path.basename(lastEpochPath).replace("model{}".format(modelInd),""))
            paramDict[paramName] = np.genfromtxt("../results/{}/model{}_epoch{}_{}.csv".format(exp_id,modelInd,lastEpoch,paramName))

        errors = error(paramDict,gtParamDict,paramKeys,errFunc)
        csvErr += "{}".format(paramValue)+"".join(["\t{}".format(round(100*errors[i],2)) for i in range(len(errors))])+"\n"

        incPer = includPerc(paramDict,gtParamDict,paramKeys)
        csvInclu += "{}".format(paramValue)+"".join(["\t{}".format(round(100*incPer[i],2)) for i in range(len(incPer))])+"\n"

    with open("../results/{}/err_epoch{}.csv".format(exp_id,epoch),"w") as text_file:
        print(csvHead,file=text_file)
        print(csvErr,file=text_file,end="")

    with open("../results/{}/inclPerc_epoch{}.csv".format(exp_id,epoch),"w") as text_file:
        print(csvHead,file=text_file)
        print(csvInclu,file=text_file)

def agregateCpWGroundTruth(exp_id,resFilePath):
    ''' Agregate the results of a comparison with ground truth

    This computes the mean and std of error (and inclusion percentage) made by the models with the same combination of varying parameters
    (this combination value is indicated in the first column of the two files produced by the compareWithGroundTruth function.

    Args:
        exp_id (str): the experience name
        resFilePath (str): the path to the csv file containing the performance of each separate models
    '''

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
    ''' Computes the two sample t-test over groud of models

    Each combination of meta-parameters is put against every other by computing the p value of the two sample t-test.

    Args:
        groupedLines (dict): a dictionnary containing the error (or inclusion percentage) of several models having the same combination of varying parameters
        exp_id (str): the experience name
        modelKeys (list): the list of the 4 parameter vectors (i.e. \'trueScores\', \'bias\', \'incons\' or \'diffs\'.)
        resFilePath (str): the path to the csv file containing the performance of each separate models

    '''

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

def csvToDict(csvPath):

    csv = np.genfromtxt(csvPath,dtype=str)
    csvDict = {}
    for i in range(1,csv.shape[0]):
        modelDict = {}
        for j in range(1,csv.shape[1]):
            mean,std = csv[i,j].split("\pm")
            modelDict[csv[0,j]] = {"mean":float(mean),"std":float(std)}
        csvDict[csv[i,0]] = modelDict

    return csvDict

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--comp_gt',type=str,nargs="*",metavar='PARAM',help='To compare the parameters found with the ground truth parameters. Require a fake dataset. The argument should\
                                    be the list of parameters varying across the different models in the experiment.')
    argreader.parser.add_argument('--comp_gt_agr',type=str,nargs="*",metavar='PARAM',help='To compare the parameters found with the ground truth parameters. Require a fake dataset. The argument should\
                                    be the list of parameters varying across the different models in the experiment. The accuracies of models having the same value for those parameters will be agregated.')

    argreader.parser.add_argument('--comp_gt_evol',type=str,nargs="*",metavar='PARAM',help='To plot the evolution of the error across epochs. The argument should\
                                    be the list of parameters varying across the different models in the experiment.')

    argreader.parser.add_argument('--error_metric',type=str,metavar='ERROR',default="rmse",help='The error metric used in \'--comp_gt\' and \'--comp_gt_agr\'. Can be \'rmse\' or \'relative\'. Default is \'RMSE\'.')

    argreader.parser.add_argument('--artif_data',action='store_true',help='To plot the real and empirical distribution of the parameters of a fake dataset. \
                                    The fake dataset to plot is set by the --dataset argument')

    argreader.parser.add_argument('--plot_param',type=str,nargs="*",help='To plot the error of every parameters at each epoch for each model. The argument values are the index of the models to plot.')
    argreader.parser.add_argument('--plot_dist',type=int,nargs="*",help='To plot the distance travelled by each parameters and the negative log-likelihood at each epoch. \
                                    The argument values are the index of the models to plot. The two last arguments are the epochs at which to start and finish the plot.')

    argreader.parser.add_argument('--two_dim_repr',type=str,nargs="*",help='To plot the t-sne visualisation of the values taken by the parameters during training. \
                                    The first argument value is the id of the model to plot and the second is the start epoch. The following argument are the parameters to plot.')

    argreader.parser.add_argument('--conv_speed',type=str,nargs='*',metavar='ID',help='To plot the error as a function of the number of annotator. The value is a list of parameters varying between \
                                    the reference models.')

    argreader.parser.add_argument('--plot_video_raw_scores',type=str,nargs='*',metavar='ID',help='To plot histograms of scores for some videos of a dataset. The value of this argument is the list of videos \
                                    line index to plot. The dataset should also be indicated with the dataset argument')

    argreader.parser.add_argument('--plot_range_pca',type=float,nargs=4,metavar="RANGE",help='The range to use when ploting the PCA. The values should be indicated in this order : xmin,xmax,ymin,ymax.')
    argreader.parser.add_argument('--plot_range_dist',type=float,nargs=2,metavar="RANGE",help='The range to use when ploting the distance. The values should be indicated in this order : ymin,ymax.')
    argreader.parser.add_argument('--labels',type=str,nargs='*',metavar="RANGE",help='The label names for the model, in the order where they will be appear in the plot.')

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

    scoreMat,distorNbList = load_data.loadData(args.dataset)

    if args.comp_gt:
        compareWithGroundTruth(args.exp_id,args.comp_gt,args.error_metric)
    if args.comp_gt_agr:
        compareWithGroundTruth(args.exp_id,args.comp_gt_agr,args.error_metric)
        agregateCpWGroundTruth(args.exp_id,"../results/{}/err_epoch-1.csv".format(args.exp_id))
        agregateCpWGroundTruth(args.exp_id,"../results/{}/inclPerc_epoch-1.csv".format(args.exp_id))

    if args.comp_gt_evol:

        #Find an id of one model in the experiment
        modelInd = findNumbers(os.path.basename(sorted(glob.glob("../models/{}/model*.ini".format(args.exp_id)))[0]))
        #The list of epoch number that have been logged
        epochs = sorted(list(map(lambda x:findNumbers(os.path.basename(x).split("_")[1]),glob.glob("../results/{}/model{}_epoch*_{}.csv".format(args.exp_id,modelInd,paramKeys[0])))))
        epochs = np.array(epochs)

        csvDict = {}
        for epoch in epochs:
            if not os.path.exists("../results/{}/err_epoch{}_agreg.csv".format(args.exp_id,epoch)):
                compareWithGroundTruth(args.exp_id,args.comp_gt_evol,args.error_metric,epoch=epoch)
                agregateCpWGroundTruth(args.exp_id,"../results/{}/err_epoch{}.csv".format(args.exp_id,epoch))
            csvDict[epoch] = csvToDict("../results/{}/err_epoch{}_agreg.csv".format(args.exp_id,epoch))

        #Collect the values of the varying hyper parameters:
        #These values identifie the models from each other
        varHyperParamValues = csvDict[list(csvDict.keys())[0]].keys()

        for param in paramKeys:
            print("Ploting ",param)
            plt.figure()

            for i,hyperParam in enumerate(varHyperParamValues):

                points = list(map(lambda x:csvDict[x][hyperParam][param]['mean'],epochs))
                stds = list(map(lambda x:csvDict[x][hyperParam][param]['std'],epochs))

                plt.errorbar(epochs+100*i,points,yerr=stds,label="{}={}".format(args.comp_gt_evol,hyperParam))

            plt.legend()
            plt.savefig("../vis/{}/err_evol_{}.png".format(args.exp_id,param))

    if args.artif_data:
        fakeDataDIstr(args)

    if args.plot_param:
        plotParam(args.dataset,args.exp_id,args.plot_param,args.labels)

    if args.plot_dist:
        plotDist(args.exp_id,args.plot_dist[0],args.plot_dist[1],args.plot_dist[2],args.plot_range_dist)

    if args.two_dim_repr:
        twoDimRepr(args.exp_id,int(args.two_dim_repr[0]),int(args.two_dim_repr[1]),args.two_dim_repr[2:],args.plot_range_pca)

    if args.conv_speed:

        #Collect the configuration files
        configFiles = glob.glob("../models/{}/model*.ini".format(args.exp_id))

        def get_Seed_NbAnnot(x):
            datasetName = readConfFile(x,["dataset"])[0]
            seed,nb_annot = readConfFile("../data/{}.ini".format(datasetName),["seed","nb_annot"])
            return int(seed),int(nb_annot)

        #Gets the ids, the seeds and the nb of annotators for every models in the experience
        ids = list(map(lambda x:findNumbers(os.path.basename(x)),configFiles))
        seeds,nb_annots = zip(*list(map(get_Seed_NbAnnot,configFiles)))

        ids_seeds_nbAnnots = zip(ids,seeds,nb_annots)

        #Find the ids and seeds of models which are trained on the full dataset
        argmaxs = np.argwhere(nb_annots == np.amax(nb_annots)).flatten()
        ids = np.array(ids)[argmaxs]
        seeds = np.array(seeds)[argmaxs]

        #Sort with the ids value to make debut easier
        ids,seeds = zip(*sorted(zip(ids,seeds),key=lambda x:x[0]))

        convSpeed(args.exp_id,ids,seeds,args.conv_speed)

    if args.plot_video_raw_scores:
        plotVideoRawScores(args.dataset,args.plot_video_raw_scores,args.score_min,args.score_max)

if __name__ == "__main__":
    main()
