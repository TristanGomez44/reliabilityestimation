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

class MLE(nn.Module):
    ''' Implement the model proposed in Zhi et al. (2017) : Recover Subjective Quality Scores from Noisy Measurements'''

    def __init__(self,videoNb,contentNb,annotNb,distorNbList,polyDeg,score_dis,score_min,score_max,div_beta_var):

        super(MLE, self).__init__()

        self.bias  = nn.Parameter(torch.ones(annotNb))
        self.incons  = nn.Parameter(torch.ones(annotNb))
        self.diffs  = nn.Parameter(torch.ones(contentNb,polyDeg+1))
        self.trueScores  = nn.Parameter(torch.ones(videoNb))
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
        self.disDict = None
        self.paramProc = None

    def forward(self,xInt):

        x = xInt.float()
        scoresDis = self.compScoreDis(x.is_cuda)

        x = generateData.betaNormalize(x,self.score_min,self.score_max)

        #print(scoresDis.log_prob(x.unsqueeze(2)))
        #sys.exit(0)

        log_prob = scoresDis.log_prob(x.unsqueeze(2)).sum()
        #print(x.unsqueeze(2))
        return -log_prob

    def compScoreDis(self,x_is_cuda):

        amb_incon = self.ambInconsMatrix(x_is_cuda)
        scor_bias = self.trueScoresBiasMatrix()

        if self.score_dis == "Beta":

            scor_bias = torch.clamp(scor_bias,self.score_min,self.score_max)
            scor_bias = generateData.betaNormalize(scor_bias,self.score_min,self.score_max)
            alpha,beta = generateData.meanvar_to_alphabeta(scor_bias,amb_incon/self.div_beta_var)

            scoresDis = Beta(alpha.unsqueeze(2),beta.unsqueeze(2))

            for i in range(scor_bias.size(0)):
                for j in range(scor_bias.size(1)):
                    if scor_bias[i,j]>1 or scor_bias[i,j]<0:
                        print(i,j,scor_bias[i,j])

        elif self.score_dis == "Normal":
            scoresDis = Normal(scor_bias.unsqueeze(2),amb_incon.unsqueeze(2))
        else:
            raise ValueError("Unknown score distribution : {}".format(self.score_dis))

        return scoresDis

    def ambInconsMatrix(self,xIsCuda):

        #print("diff",torch.sigmoid(self.diffs[0]),"incons",torch.sigmoid(self.incons[0]))

        #Matrix containing the sum of all (video_amb,annot_incons) possible pairs
        #amb = self.diffs.unsqueeze(1).expand(self.contentNb, self.distorNb).contiguous().view(-1)
        tmp = []
        for i in range(self.contentNb):
            tmp.append(torch.sigmoid(self.diffs[i]).unsqueeze(1).expand(self.polyDeg+1,self.distorNbList[i]))

        amb = torch.cat(tmp,dim=1).permute(1,0)
        #amb = amb.expand(amb.size(0),self.polyDeg)
        #print(amb.size())

        vid_means = self.trueScores.unsqueeze(1).expand(self.trueScores.size(0),self.polyDeg+1)
        powers = torch.arange(self.polyDeg+1).float()
        if xIsCuda:
            powers = powers.cuda()
        powers = powers.unsqueeze(1).permute(1,0).expand(vid_means.size(0),self.polyDeg+1)
        vid_means_pow = torch.pow(vid_means,powers)
        #print(powers)
        amb_pol = (amb*vid_means_pow).sum(dim=1)

        #print("diff",amb_pol[0])

        amb_sq = torch.pow(amb_pol,2)
        amb_sq = amb_sq.unsqueeze(1).expand(self.videoNb, self.annotNb)
        incon_sq = torch.pow(torch.sigmoid(self.incons),2)
        incon_sq = incon_sq.unsqueeze(0).expand(self.videoNb, self.annotNb)

        #print("diff",amb_sq[0,0],"incons",incon_sq[0,0])
        return amb_sq+incon_sq

    def trueScoresBiasMatrix(self):

        #Matrix containing the sum of all (video_scor,annot_bias) possible pairs
        scor = self.trueScores.unsqueeze(1).expand(self.videoNb,self.annotNb)
        bias = self.bias.unsqueeze(0).expand(self.videoNb, self.annotNb)
        return scor+bias

    def setParams(self,annot_bias,annot_incons,video_amb,video_scor):

        self.bias = nn.Parameter(annot_bias)
        self.incons = nn.Parameter(annot_incons)
        self.diffs = nn.Parameter(video_amb)
        self.trueScores = nn.Parameter(video_scor)

    def init_base(self,dataInt):

        annot_bias = torch.zeros(dataInt.size(1))
        #Use std to initialise inconsistency seems a bad idea.
        #Even really consistent annotators can give score
        #in a large range
        data = dataInt.float()
        annot_incons = data.std(dim=0)
        vid_score = data.mean(dim=1)

        #Computing initial values for video ambiguity

        #data3D = data.view(self.distorNb,self.contentNb,self.annotNb).permute(1,0,2)
        data3D = []
        currInd = 0
        video_ambTensor = torch.zeros(self.contentNb,self.polyDeg+1)

        if dataInt.is_cuda:
            annot_bias = annot_bias.cuda()
            video_ambTensor = video_ambTensor.cuda()

        for i in range(self.contentNb):

            #The data for all videos made from this reference video
            videoData = data[currInd:currInd+self.distorNbList[i]]
            data3D.append(videoData)
            currInd += self.distorNbList[i]
            #print(data.mean(dim=0))
            expanded_use_score_mean = data.mean(dim=0).unsqueeze(1).permute(1, 0).expand_as(videoData)
            video_ambTensor[i,0] = torch.pow(videoData-expanded_use_score_mean,2).sum()/(self.annotNb*self.contentNb)
            video_ambTensor[i,0] = torch.sqrt(video_ambTensor[i,0])

        self.setParams(annot_bias,annot_incons,video_ambTensor,vid_score)

    def init(self,dataInt,datasetName,paramNotGT):

        self.init_base(dataInt)
        paramNameList = list(self.state_dict().keys())

        gtParamDict = {}
        for paramName in paramNameList:
            gtParamDict[paramName] = np.genfromtxt("../data/{}_{}.csv".format(datasetName,paramName))

        for key in paramNameList:

            tensor = torch.tensor(gtParamDict[key]).view(getattr(self,key).size()).float()

            oriSize = tensor.size()
            tensor = tensor.view(-1)

            tensor = tensor.view(oriSize)

            if key == "incons" or key == "diffs":
                tensor = torch.log(tensor/(1-tensor))

            setattr(self,key,nn.Parameter(tensor))

        for key in paramNotGT:

            initFunc = getattr(self,"init_{}".format(key))

            if key=="diffs" or key=="incons":
                tensor = initFunc(dataInt)
                tensor = torch.log(tensor/(1-tensor))
                setattr(self,key,nn.Parameter(tensor))
            else:
                setattr(self,key,nn.Parameter(initFunc(dataInt)))

    def init_trueScores(self,dataInt):
        return dataInt.float().mean(dim=1)

    def init_bias(self,dataInt):
        return (dataInt.float()-self.trueScores.unsqueeze(1).expand_as(dataInt)).mean(dim=0)

    def init_diffs(self,dataInt):

        video_amb = torch.pow(self.trueScoresBiasMatrix()-dataInt.float(),2).mean(dim=1)
        content_amb = torch.zeros(len(self.distorNbList),1)
        sumInd = 0

        for i in range(len(self.distorNbList)):

            content_amb[i] = video_amb[sumInd:sumInd+self.distorNbList[i]].mean()
            sumInd += self.distorNbList[i]

        return content_amb

    def init_incons(self,dataInt):

        #print(torch.pow(self.trueScoresBiasMatrix()-dataInt,2)[:,-1])
        #print(torch.pow(self.trueScoresBiasMatrix()-dataInt,2).mean(dim=0)[-1])

        res = torch.sqrt(torch.pow(self.trueScoresBiasMatrix()-dataInt.float(),2).mean(dim=0))
        res = torch.clamp(res,0.01,0.99)

        return res

    def unifPrior(self,loss):
        return loss

    def oraclePrior(self,loss):
        for param in self.disDict.keys():

            loss -= self.disDict[param].log_prob(getattr(self,param)).sum()

            #print(param,getattr(self,param),self.disDict[param].log_prob(getattr(self,param)))
        return loss

    def setPrior(self,priorName,dataset):

        if priorName == "oracle":
            self.prior = self.oraclePrior

            self.disDict = {}
            dataConf = configparser.ConfigParser()
            dataConf.read("../data/{}.ini".format(dataset))

            #self.disDict = {"diffs":Beta(float(dataConf['diff_alpha']), float(dataConf["diff_beta"])), \
            #                "incons":Beta(float(dataConf["incons_alpha"]), float(dataConf["incons_beta"])),\
            #                "bias":Normal(torch.zeros(1), float(dataConf["bias_std"])*torch.eye(1))}

            #self.paramProc = {"diffs":lambda x:torch.sigmoid(x), \
            #                 "incons":lambda x:torch.sigmoid(x),\
            #                 "bias":lambda x:x}

            dataConf = dataConf['default']
            self.disDict = {"bias":Normal(torch.zeros(1), float(dataConf["bias_std"])*torch.eye(1))}

            self.paramProc = {"bias":lambda x:x}

        elif priorName == "uniform":
            self.prior = self.unifPrior
            self.disDict = None
            self.paramProc = None
        else:
            raise ValueError("No such prior : {}".format(priorName))

    def newtonUpdate(xInt):

        x = xInt.float()

        #Matrix containing the sum of all (video_amb,annot_incons) possible pairs
        #amb = self.diffs.unsqueeze(1).expand(self.contentNb, self.distorNb).contiguous().view(-1)
        tmp = []
        for i in range(self.contentNb):
            tmp.append(self.diffs[i].expand(self.distorNbList[i]))

        amb = torch.cat(tmp,dim=1).unsqueeze(0)
        amb_sq = torch.pow(amb,2)
        amb_sq = amb_sq.unsqueeze(1).expand(self.videoNb, self.annotNb)
        incon_sq = torch.pow(self.incons,2)
        incon_sq = incon_sq.unsqueeze(0).expand(self.videoNb, self.annotNb)
        w_es = amb_sq+incon_sq

        scor = self.trueScores.unsqueeze(1).expand(self.videoNb, self.annotNb)
        bias = self.bias.unsqueeze(0).expand(self.videoNb, self.annotNb)

        scor_new = (w_es*(x-bias)/w_es).sum(dim=1)
        bias_new = (w_es*(x-scor)/w_es).sum(dim=0)

        #incon = self.incons.unsqueeze(0).expand(self.videoNb, self.annotNb)
        v = self.incons

        w_es_sq = torch.pow(w_es,2)

        incons_new = v - ((w_es*v-w_es_sq*v*torch.pow(x-scor-bias,2))/()).sum(dim=0)

    def getFlatParam(self):

        return torch.cat((self.trueScores,self.bias,self.incons.view(-1),self.diffs.view(-1)),dim=0)

def subRej(dataTorch):

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

    data_rm = torch.zeros(data.size(0),data.size(1)-len(removeList))
    colCount = 0
    for j in range(data.size(1)):
        if not j in removeList:
            data_rm[:,colCount] = data[:,j]
            colCount += 1
    return data_rm

def scrambleColumns(data,annotToScramble):

    data_scr = data.clone()

    randomScores = torch.zeros((data.size(0),len(annotToScramble))).int().random_(1,6)
    #print(randomScores)
    c=0
    for i in range(len(data)):
        if i in annotToScramble:
            data_scr[:,i] = randomScores[:,c]
            c += 1
    #print(data_scr)
    return data_scr

def scoreNoise(data,noisePercent):

    dataFlat = data.view(-1)
    noisy_score_nb = int(noisePercent*dataFlat.size(0))

    scoresInd = random.sample(range(dataFlat.size(0)),noisy_score_nb)
    scoresNoise = torch.LongTensor(noisy_score_nb).random_(0, 2)*2-1

    c = 0
    for i in range(len(dataFlat)):
        if i in scoresInd:
            if dataFlat[i]==1:
                dataFlat[i]=2
            elif  dataFlat[i]==5:
                dataFlat[i]=4
            else:
                dataFlat[i] += scoresNoise[c]
            c += 1
    return dataFlat.view(data.size())

def MOS(dataInt,z_score,sub_rej):

    data = dataInt.float()

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
    mos_conf = 1.96*data.std(dim=1)/math.sqrt(data.size(1))

    if mos_mean.is_cuda:
        mos_mean = mos_mean.cpu()
        mos_conf = mos_conf.cpu()

    return mos_mean.numpy(),mos_conf.numpy()

def modelMaker(annot_nb,nbVideos,distorNbList,polyDeg,score_dis,score_min,score_max,div_beta_var):
    '''Build a model
    Args:
        annot_nb (int): the bumber of annotators
        nbVideos (int): the number of videos in the dataset
        distorNbList (list): the number of distortion video for each reference video in the dataset
    Returns:
        the built model
    '''

    model = MLE(nbVideos,len(distorNbList),annot_nb,distorNbList,polyDeg,score_dis,score_min,score_max,div_beta_var)
    return model

if __name__ == '__main__':

    mle = MLE(192,24,28)

    #mle.setParams(torch.rand(28),torch.rand(28),torch.rand(24),torch.rand(192))
    mle(torch.ones((192,28)))
