import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from args import str2bool
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
import math
import numpy as np
from scipy.stats import moment
import random
from torch.distributions.normal import Normal

from torch.distributions.beta import Beta
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

import configparser

import generateData
import os

import train

import torch.optim as optim
class MLE(nn.Module):
    ''' Implement all the models proposed in our paper including the one proposed in Zhi et al. (2017) : Recover Subjective Quality Scores from Noisy Measurements'''

    def __init__(self,videoNb,annotNb,distorNbList,polyDeg,score_dis,score_min,score_max,div_beta_var,priorUpdateFrequ,extr_sco_dep,nb_video_per_content):
        '''
        Args:
            videoNb (int): the total number of video
            annotNb (int): the number of annotators
            distorNbList (list): the number of videos for each reference videos, in the same order than in the dataset.
            polyDeg (int): the degree of the polynom used to model video ambiguities. E.g. 0 makes the video ambiguity depends only on the content
            scoresDis (str): the score distribution used
            score_min (int): the minimum score that can be given to a video
            score_max (int): the maximum score that can be given to a video
            div_beta_var (int): the factor with which to reduce the variance of the beta distribution, to ensure numerical stability.
            priorUpdateFrequ (int): the number of epoch to wait before updating the empirical prior. Ignored if the empirical prior is not used.
            extr_sco_dep (bool): whether or not to add a dependency between the variance and the mean of videos. If true, raw score variance of videos with very high or very low scores\
            will be lower.
        '''

        super(MLE, self).__init__()

        contentNb = len(distorNbList)

        self.trueScores  = nn.Parameter(torch.ones(videoNb).double())
        self.bias = nn.Parameter(torch.ones(annotNb).double())
        self.incons  = nn.Parameter(torch.ones(annotNb).double())
        self.diffs  = nn.Parameter(torch.ones(contentNb*(polyDeg+1)).double())

        self.annotNb = annotNb
        self.videoNb = videoNb
        self.contentNb = contentNb

        self.polyDeg = polyDeg
        self.distorNbList = distorNbList

        self.score_min = score_min
        self.score_max = score_max
        self.score_dis = score_dis

        self.div_beta_var = div_beta_var

        self.prior = None
        self.priorName = None
        self.disDict = None
        self.paramProc = None

        self.priorUpdateCount = priorUpdateFrequ
        self.priorUpdateFrequ = priorUpdateFrequ

        self.extr_sco_dep = extr_sco_dep

        if not nb_video_per_content is None:
            self.vidInds = torch.arange(videoNb)
            self.nb_video_per_content = nb_video_per_content

    def forward(self,scoreMat):
        '''Compute the negative log probability of the data according to the model
        Args:
            scoreMat (torch.tensor): the score matrix to train on
        '''

        if self.priorName == "empirical":
            #print(self.priorUpdateCount)
            if self.priorUpdateCount == self.priorUpdateFrequ:

                self.updateEmpirical()

                self.priorUpdateCount = 0
            else:
                self.priorUpdateCount += 1

        x = scoreMat.double()
        scoresDis = self.compScoreDis(x.is_cuda)

        x = generateData.betaNormalize(x,self.score_min,self.score_max,rawScore=True)

        log_prob = scoresDis.log_prob(x).sum()

        return log_prob

    def compScoreDis(self,x_is_cuda):
        '''Build the raw scores distributions
        Args:
            x_is_cuda (bool): whether or not to use cuda
        '''

        scor_bias = self.trueScoresBiasMatrix()

        scor_bias = torch.clamp(scor_bias,self.score_min+0.001,self.score_max-0.001)

        amb_incon = self.ambInconsMatrix(x_is_cuda,scor_bias)

        scor_bias = generateData.betaNormalize(scor_bias,self.score_min,self.score_max)
        amb_incon = generateData.betaNormalize(amb_incon,self.score_min,self.score_max,variance=True)
        #The variance of the beta distribution can not be too big
        #This line clamps it value
        #print(torch.min(amb_incon, scor_bias*(1-scor_bias)-0.0001))
        amb_incon = torch.max(torch.min(amb_incon, scor_bias*(1-scor_bias)-0.0001), torch.tensor([0.00001]).expand_as(amb_incon).double())

        if self.score_dis == "Beta":

            #print("Forward")
            #print(self.bias)

            alpha,beta = generateData.meanvar_to_alphabeta(scor_bias*0.0001+0.5,amb_incon*0.00001+0.01)
            #print(alpha,beta)
            scoresDis = Beta(alpha,beta)

        elif self.score_dis == "Normal":
            scoresDis = Normal(scor_bias,amb_incon)
        else:
            raise ValueError("Unknown score distribution : {}".format(self.score_dis))

        return scoresDis

    def ambInconsMatrix(self,xIsCuda,meanMat):
        '''Computes the variances of the raw score distributions according to the model.

        The result is a matrix where each cell is the variance of a raw score density

        Args:
            x_is_cuda (bool): whether or not to use cuda
        '''

        self.diffs_mat = self.diffs.view(self.contentNb,self.polyDeg+1)

        #Matrix containing the sum of all (video_amb,annot_incons) possible pairs
        tmp = []
        for i in range(self.contentNb):
            tmp.append(torch.sigmoid(self.diffs_mat[i]).unsqueeze(1).expand(self.polyDeg+1,self.distorNbList[i]))

        amb = torch.cat(tmp,dim=1).permute(1,0)

        vid_means = self.trueScores.unsqueeze(1).expand(self.trueScores.size(0),self.polyDeg+1)
        powers = torch.arange(self.polyDeg+1).double()
        if xIsCuda:
            powers = powers.cuda()
        powers = powers.unsqueeze(1).permute(1,0).expand(vid_means.size(0),self.polyDeg+1)
        vid_means_pow = torch.pow(vid_means,powers)

        amb_pol = (amb*vid_means_pow).sum(dim=1)

        amb_sq = torch.pow(amb_pol,2)
        incon_sq = torch.pow(torch.sigmoid(self.incons),2)

        if self.extr_sco_dep:
            amb_sq = amb_sq*(-(self.trueScores-self.score_min)*(self.trueScores-self.score_max))
            amb_sq = amb_sq.unsqueeze(1).expand(self.videoNb, self.annotNb)

            incon_sq = incon_sq.unsqueeze(0).expand(self.videoNb, self.annotNb)
            incon_sq = incon_sq*(-(meanMat-self.score_min)*(meanMat-self.score_max))

        else:
            amb_sq = amb_sq.unsqueeze(1).expand(self.videoNb, self.annotNb)
            incon_sq = incon_sq.unsqueeze(0).expand(self.videoNb, self.annotNb)

        return amb_sq+incon_sq

    def trueScoresBiasMatrix(self):
        '''Computes the means of the raw score distributions according to the model.

        The result is a matrix where each cell is the mean of a raw score density

        '''

        if not self.nb_video_per_content is None:
            trueScores = torch.zeros(len(self.trueScores)).double()

            trueScores[self.vidInds%self.nb_video_per_content == 0] = self.score_max
            trueScores[self.vidInds%self.nb_video_per_content != 0] = self.trueScores[self.vidInds%self.nb_video_per_content != 0]

        else:
            trueScores = self.trueScores

        #Matrix containing the sum of all (video_scor,annot_bias) possible pairs
        scor = trueScores.unsqueeze(1).expand(self.videoNb,self.annotNb)
        bias = self.bias.unsqueeze(0).expand(self.videoNb, self.annotNb)

        return scor+bias

    def init(self,scoreMat,datasetName,score_dis,paramNotGT,true_scores_init,bias_init,diffs_init,incons_init,iterInit=False):
        ''' Initialise the parameters of the model using ground-truth or approximations only based on data
        Args:
            scoreMat (torch.tensor): the score matrix to train on
            datasetName (str): the dataset name (useful only if the dataset is artificial and if some parameters have to initialised with ground truth)
            score_dis (str): the score distribution to use. Can be \'Beta\' or \'Normal\'.
            paramNotGT (list): the name of parameters not initialised with ground truth but with approximations only based on data. The parameters will be
                initialised in the same order they appear in this list.
            true_scores_init (str): the name of the function to use to initialise true scores, if it is not initialised with ground truth
            bias_init (str): same than true_scores_init, but with biases.
            diffs_init (str): same than true_scores_init, but with difficulties.
            incons_init (str): same than true_scores_init, but with inconsistencies.
            iterInit (bool): indicates if the parameter initialisation is done with the iterative method or not.
        '''

        paramNameList = list(self.state_dict().keys())

        #if the list of parameters not to set at ground truth iis long as the number of parameters
        #it means that all parameters will be initialised with aproximation
        if len(paramNameList) > len(paramNotGT):

            gtParamDict = {}
            for paramName in paramNameList:

                gtParamDict[paramName] = np.genfromtxt("../data/{}_{}.csv".format(datasetName,paramName))

            for key in paramNameList:

                tensor = torch.tensor(gtParamDict[key]).view(getattr(self,key).size()).double()

                oriSize = tensor.size()
                tensor = tensor.view(-1)

                tensor = tensor.view(oriSize)

                if (key == "incons" or key == "diffs") and score_dis=="Beta":
                    tensor = torch.log(tensor/(1-tensor))

                setattr(self,key,nn.Parameter(tensor.double()))

        functionNameDict = {'bias':bias_init,'trueScores':true_scores_init,'diffs':diffs_init,'incons':incons_init}

        if iterInit and len(paramNotGT) == len(paramNameList):

            self.iterativeInit(scoreMat)
        else:

            for key in paramNotGT:

                initFunc = getattr(self,functionNameDict[key])

                tensor = initFunc(scoreMat)
                if (key == "incons" or key == "diffs") and score_dis=="Beta":
                    tensor = torch.log(tensor/(1-tensor))

                setattr(self,key,nn.Parameter(tensor))

    def iterativeInit(self,scoreMat):

        print("Iterative init")

        #print(self.trueScores)
        self.trueScores = nn.Parameter(self.tsInitBase(scoreMat))
        #print(self.trueScores)
        self.bias = nn.Parameter(self.bInitBase(scoreMat))
        self.incons = nn.Parameter(self.iInitBase(scoreMat))
        self.diffs = nn.Parameter(self.dInitBase(scoreMat))

        i=0
        maxIt = 10
        minDist = 0.0001
        dist = minDist+1
        while i<maxIt and dist>minDist:

            oldFlatParam = torch.cat((self.trueScores.detach().clone(),self.bias.detach().clone(),self.incons.detach().clone(),self.diffs.detach().clone()),dim=0)

            #print(self.trueScores)
            self.trueScores = nn.Parameter(self.tsInitBase(scoreMat))
            #print(self.trueScores)
            self.bias = nn.Parameter(self.bInitBase(scoreMat))
            self.incons = nn.Parameter(self.iInitBase(scoreMat))
            self.diffs = nn.Parameter(self.dInitBase(scoreMat))

            dist = torch.pow(oldFlatParam - torch.cat((self.trueScores,self.bias,self.incons,self.diffs),dim=0),2).sum()
            print("\t",i,dist)

    def tsInitBase(self,scoreMat):
        ''' Compute the mean opinion score for each video. Can be used to initialise the true score vector using only data
        Args:
            scoreMat (torch.tensor): the score matrix
        '''

        res = scoreMat.double().mean(dim=1)
        return res

    def bInitBase(self,scoreMat):
        ''' Compute the mean of the difference between true scores and raw scores for every annotator.
        Can be used to initialise the bias vector, like tsInitBase
        Args:
            scoreMat (torch.tensor): the score matrix
        '''

        return (scoreMat.double()-self.trueScores.unsqueeze(1).expand_as(scoreMat)).mean(dim=0)

    def dInitBase(self,scoreMat):
        '''Compute the standard deviation of scores given to every content.
        Can be used to initialise the difficulties vector, like tsInitBase
        Args:
            scoreMat (torch.tensor): the score matrix
        '''

        video_amb = torch.pow(self.trueScoresBiasMatrix()-scoreMat.double(),2).mean(dim=1)

        content_amb = torch.zeros(len(self.distorNbList)*(self.polyDeg+1))
        sumInd = 0

        for i in range(len(self.distorNbList)):

            content_amb[i] = torch.sqrt(video_amb[sumInd:sumInd+self.distorNbList[i]].mean())
            sumInd += self.distorNbList[i]

        #Setting min to 0 or the variance is always estimated too big
        content_amb = (content_amb-content_amb.min())
        #Clamping because std of data can be bigger than 1 and for numerical stability
        return torch.clamp(content_amb,0.01,0.99).double()

    def iInitBase(self,scoreMat):
        '''Compute the standard deviation of scores given by every annotators.
        Can be used to initialise the inconsistencies vector, like tsInitBase
        Args:
            scoreMat (torch.tensor): the score matrix
        '''

        res = torch.sqrt((torch.pow(self.trueScoresBiasMatrix()-scoreMat.double(),2)).mean(dim=0))

        #Setting min to 0 or the variance is always estimated too big
        res = (res-res.min())
        #Clamping because std of data can be bigger than 1 and for numerical stability
        res = torch.clamp(res,0.01,0.99).double()
        return res

    def updateEmpirical(self):
        '''Update the empirical prior.

        Compute the mean and variance of the four vector of parameters (true scores, biases, inconsistencies and difficulties),
        and use those means and variances as prior for the next iterations.

        '''

        for key in self.paramProc.keys():

            if key == "bias":

                self.disDict["bias"] = Normal(torch.zeros(1).double(), torch.pow(torch.std(self.bias),2)*torch.eye(1).double())

            else:
                sigTensor = torch.sigmoid(getattr(self,key))
                mean,var = sigTensor.mean(),torch.pow(sigTensor.std(),2)
                alpha,beta = generateData.meanvar_to_alphabeta(mean,var)
                self.disDict[key] = Beta(alpha, beta)

    def empiricalPrior(self,loss):
        ''' Add the prior terms for the empirical prior
        Args:
            loss (torch.Tensor): the loss
        Returns
            the loss with the prior terms added
        '''

        for param in self.disDict.keys():
            procFunc = self.paramProc[param]
            #Apply a preprocessing function (like sigmoid for the inconsistencies and difficulties) and computes the log prob.

            loss -= self.disDict[param].log_prob(procFunc(getattr(self,param))).mean()

        return loss

    def unifPrior(self,loss):
        ''' The uniform prior
        Args:
            loss (torch.Tensor): the loss
        Returns
            the unchanged loss'''
        return loss
        return loss

    def oraclePrior(self,loss):
        ''' Add the prior terms for the oracle prior
        The oracle prior uses ground-truth parameters of the dataset, which should be artificial.
        Args:
            loss (torch.Tensor): the loss
        Returns
            the loss with the prior terms added
        '''

        for param in self.disDict.keys():
            procFunc = self.paramProc[param]
            loss -= self.disDict[param].log_prob(procFunc(getattr(self,param))).sum()

        return loss

    def setPrior(self,priorName,dataset):
        ''' Set the desired prior

        This function prepare the preprocessing functions (also reads the ground-truth parameters if the prior is and oracle)

        Args:
            priorName (str): the name of the prior desired. Can be :
            - uniform : the prior is uniform for all parameters (i.e. no prior)
            - oracle_bias : the prior is uniform for all parameters except the biases where it uses the ground truth
            - oracle : the prior uses the ground truth parameters
            - empirical : the prior is updated with the parameters means and variances found during optimisation
        '''


        self.priorName = priorName
        if priorName == "empirical":
            self.prior = self.empiricalPrior

            self.disDict = {}

            self.paramProc = {"diffs":lambda x:torch.sigmoid(x), \
                             "incons":lambda x:torch.sigmoid(x),\
                             "bias":lambda x:x}

        elif priorName == "oracle":
            self.prior = self.oraclePrior

            self.disDict = {}
            dataConf = configparser.ConfigParser()
            if os.path.exists("../data/{}.ini".format(dataset)):
                dataConf.read("../data/{}.ini".format(dataset))
                dataConf = dataConf['default']

                self.disDict = {"diffs":Beta(float(dataConf['diff_alpha']), float(dataConf["diff_beta"])), \
                                "incons":Beta(float(dataConf["incons_alpha"]), float(dataConf["incons_beta"])),\
                                "bias":Normal(torch.zeros(1), np.power(float(dataConf["bias_std"]),2)*torch.eye(1))}

                self.paramProc = {"diffs":lambda x:torch.sigmoid(x), \
                                 "incons":lambda x:torch.sigmoid(x),\
                                 "bias":lambda x:x}

            else:
                raise ValueError("Oracle prior require artificial dataset with a config file")
        elif priorName == "oracle_bias":

            self.prior = self.oraclePrior

            self.disDict = {}
            dataConf = configparser.ConfigParser()
            if os.path.exists("../data/{}.ini".format(dataset)):
                dataConf.read("../data/{}.ini".format(dataset))
                dataConf = dataConf['default']

                self.disDict = {"bias":Normal(torch.zeros(1), float(dataConf["bias_std"])*torch.eye(1))}
                self.paramProc = {"bias":lambda x:x}

            else:
                raise ValueError("Oracle prior require artificial dataset with a config file")
        elif priorName == "uniform":
            self.prior = self.unifPrior
            self.disDict = None
            self.paramProc = None
        else:
            raise ValueError("No such prior : {}".format(priorName))

    def getFlatParam(self):
        ''' Return a vector with the four parameters vector concatenated (true scores, biases, inconsistencies and difficulties) '''

        return torch.cat((self.trueScores,self.bias,self.incons.view(-1),self.diffs.view(-1)),dim=0)

def subRej(dataTorch):
    ''' Creates a list of subject to reject based on how far they are from average answer. Comes from ITU-R BT.500: Methodology for the Subjective Assessment of the Quality of
        Television Pictures. [Online]. Available: https://www.itu.int/rec/R-REC-BT.500

        Args:
            dataTorch (torch.tensor): the score matrix
        Returns:
            the list of subjects to reject indexs

    '''

    if dataTorch.is_cuda:
        dataTorch = dataTorch.cpu()
    data = dataTorch.numpy()

    p = np.zeros(data.shape[1])
    q = np.zeros(data.shape[1])
    kurt = np.zeros(data.shape[0])
    eps = np.zeros(data.shape[0])
    for i in range(data.shape[0]):

        mom_sec = moment(data[i],2)
        kurt = moment(data[i],4)/(mom_sec*mom_sec)

        if 2 <= kurt and kurt <= 4:
            eps = 2
        else:
            eps = math.sqrt(20)

        vidMeans = data.mean(axis=1)
        vidStds = data.std(axis=1)
        for j in range(data.shape[1]):
            if data[i,j] >= vidMeans[j]+ eps*vidStds[j]:
                p[j] += 1
            if data[i,j] <= vidMeans[j]- eps*vidStds[j]:
                q[j] += 1
    rejList = []
    for j in range(data.shape[1]):
        if ((p[j]+q[j])/data.shape[1] >= 0.05) and (np.abs((p[j]-q[j])/(p[j]+q[j]))<0.3):
            rejList.append(j)

    return rejList

def removeColumns(data,removeList):
    ''' Removes columns from the score matrix
    Args:
        data (torch.tensor): the score matrix
        removeList (list): the list of column indexs to remove
    Returns:
        The score matrix with the chosen columns removed

    '''

    data_rm = torch.zeros(data.size(0),data.size(1)-len(removeList))
    colCount = 0
    for j in range(data.size(1)):
        if not j in removeList:
            data_rm[:,colCount] = data[:,j]
            colCount += 1
    return data_rm

def MOS(scoreMat,z_score,sub_rej):
    ''' Computes the mean opinion score (MOS) and the standard deviation of opinion scores (SOS) for every video

    Args:
        data (torch.tensor): the score matrix
        z_score (bool): whether or not to compute z-scoring as preprocessing
        sub_rej (bool): whether or not to use the subjecti rejection algorithm
    Returns
        the MOS and the SOS

    '''

    data = scoreMat.double()

    if sub_rej:
        rejList = subRej(data)
        data = removeColumns(data,rejList)

    mean_sub = data.mean(dim=0)
    mean_sub = mean_sub.unsqueeze(-1).permute(1,0).expand(data.size())

    std_sub = data.std(dim=0)
    std_sub = std_sub.unsqueeze(-1).permute(1,0).expand(data.size())

    if z_score:
        data = (data-mean_sub)/std_sub

    mos_mean = data.mean(dim=1)
    mos_conf = data.std(dim=1)

    if mos_mean.is_cuda:
        mos_mean = mos_mean.cpu()
        mos_conf = mos_conf.cpu()

    return mos_mean.numpy(),mos_conf.numpy()

def modelMaker(annot_nb,nbVideos,distorNbList,polyDeg,score_dis,score_min,score_max,div_beta_var,priorUpdateFrequ,extr_sco_dep,nb_video_per_content):
    '''Build a model
    Args:
        annot_nb (int): the number of annotators
        nbVideos (int): the total number of video
        distorNbList (list): the number of videos for each reference videos, in the same order than in the dataset.
        polyDeg (int): the degree of the polynom used to model video ambiguities. E.g. 0 makes the video ambiguity depends only on the content

        scoresDis (str): the score distribution used
        score_min (int): the minimum score that can be given to a video
        score_max (int): the maximum score that can be given to a video

        div_beta_var (int): the factor with which to reduce the variance of the beta distribution, to ensure numerical stability.
        priorUpdateFrequ (int): the number of epoch to wait before updating the empirical prior. Ignored if the empirical prior is not used.

        extr_sco_dep (bool): whether or not to add a dependency between the variance and the mean of videos. If true, raw score variance of videos with very high or very low scores\
        will be lower.

    Returns:
        the built model
    '''

    model = MLE(nbVideos,annot_nb,distorNbList,polyDeg,score_dis,score_min,score_max,div_beta_var,priorUpdateFrequ,extr_sco_dep,nb_video_per_content)
    return model

if __name__ == '__main__':

    mle = MLE(192,24,28)

    mle(torch.ones((192,28)))
