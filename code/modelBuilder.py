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
class MLE(nn.Module):
    ''' Implement the model proposed in Zhi et al. (2017) : Recover Subjective Quality Scores from Noisy Measurements'''

    def __init__(self,videoNb,contentNb,annotNb,distorNbList,polyDeg,score_dis,score_min,score_max,div_beta_var,\
                nbFreezTrueScores,nbFreezBias,nbFreezDiffs,nbFreezIncons,priorUpdateFrequ):

        super(MLE, self).__init__()

        self.nbOptiDict = {"trueScores":nbFreezTrueScores,"bias":nbFreezBias,"diffs":nbFreezDiffs,"incons":nbFreezIncons}

        self.bias_opti  = nn.Parameter(torch.ones(annotNb-nbFreezBias))
        self.incons_opti  = nn.Parameter(torch.ones(annotNb-nbFreezIncons))
        self.diffs_opti  = nn.Parameter(torch.ones(contentNb-nbFreezDiffs,polyDeg+1))
        self.trueScores_opti  = nn.Parameter(torch.ones(videoNb-nbFreezTrueScores))

        self.bias_freez = torch.Tensor.requires_grad_(torch.ones((nbFreezBias)))
        self.incons_freez = torch.Tensor.requires_grad_(torch.ones((nbFreezIncons)))
        self.diffs_freez = torch.Tensor.requires_grad_(torch.ones((nbFreezDiffs,polyDeg+1)))
        self.trueScores_freez = torch.Tensor.requires_grad_(torch.ones((nbFreezTrueScores)))

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

    def forward(self,xInt):

        if self.priorName == "empirical":
            #print(self.priorUpdateCount)
            if self.priorUpdateCount == self.priorUpdateFrequ:

                self.updateEmpirical()

                self.priorUpdateCount = 0
            else:
                self.priorUpdateCount += 1

        self.bias  = torch.cat((self.bias_freez,self.bias_opti),dim=0)
        self.incons  = torch.cat((self.incons_freez,self.incons_opti),dim=0)
        self.diffs = torch.cat((self.diffs_freez,self.diffs_opti),dim=0)
        self.trueScores = torch.cat((self.trueScores_freez,self.trueScores_opti),dim=0)

        x = xInt.float()
        scoresDis = self.compScoreDis(x.is_cuda)

        if self.score_dis == "Beta":
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

    def init(self,dataInt,datasetName,score_dis,paramNotGT,true_scores_init,bias_init,diffs_init,incons_init):

        #self.init_base(dataInt)
        paramNameList = list(self.state_dict().keys())
        paramNameList = list(map(lambda x:x.replace("_opti",""),paramNameList))

        #if the list of parameters not to set at ground truth iis long as the number of parameters
        #it means that all parameters will be initialised with aproximation
        if len(paramNameList) > len(paramNotGT):

            gtParamDict = {}
            for paramName in paramNameList:

                gtParamDict[paramName] = np.genfromtxt("../data/{}_{}.csv".format(datasetName,paramName))

            for key in paramNameList:

                firstDimSize = getattr(self,key+"_freez").size(0)+getattr(self,key+"_opti").size(0)
                if len(getattr(self,key+"_freez").size()) == 1:
                    tensor = torch.tensor(gtParamDict[key]).view(firstDimSize).float()
                else:
                    tensor = torch.tensor(gtParamDict[key]).view(firstDimSize,getattr(self,key+"_freez").size(1)).float()

                oriSize = tensor.size()
                tensor = tensor.view(-1)

                tensor = tensor.view(oriSize)

                if (key == "incons" or key == "diffs") and score_dis=="Beta":
                    tensor = torch.log(tensor/(1-tensor))

                setattr(self,key+"_freez",torch.Tensor.requires_grad_(tensor[:self.nbOptiDict[key]]))
                setattr(self,key+"_opti",nn.Parameter(tensor[self.nbOptiDict[key]:]))
                setattr(self,key,torch.cat((getattr(self,key+"_freez"),getattr(self,key+"_opti")),dim=0))

        self.incons  = torch.cat((self.incons_freez,self.incons_opti),dim=0)
        self.diffs = torch.cat((self.diffs_freez,self.diffs_opti),dim=0)
        self.trueScores = torch.cat((self.trueScores_freez,self.trueScores_opti),dim=0)

        functionNameDict = {'bias':bias_init,'trueScores':true_scores_init,'diffs':diffs_init,'incons':incons_init}

        for key in paramNotGT:

            initFunc = getattr(self,functionNameDict[key])

            tensor = initFunc(dataInt)
            if (key == "incons" or key == "diffs") and score_dis=="Beta":
                tensor = torch.log(tensor/(1-tensor))

            setattr(self,key+"_freez",torch.Tensor.requires_grad_(tensor[:self.nbOptiDict[key]]))
            setattr(self,key+"_opti",nn.Parameter(tensor[self.nbOptiDict[key]:]))
            setattr(self,key,torch.cat((getattr(self,key+"_freez"),getattr(self,key+"_opti")),dim=0))

    def tsInitBase(self,dataInt):
        return dataInt.float().mean(dim=1)

    def bInitBase(self,dataInt):
        return (dataInt.float()-self.trueScores.unsqueeze(1).expand_as(dataInt)).mean(dim=0)

    def bInitZeros(self,dataInt):
        return torch.zeros(dataInt.size(1))

    def dInitBase(self,dataInt):

        #video_amb = torch.sqrt(torch.pow(self.trueScoresBiasMatrix()-dataInt.float(),2).mean(dim=1))
        video_amb = torch.pow(self.trueScoresBiasMatrix()-dataInt.float(),2).mean(dim=1)

        content_amb = torch.zeros(len(self.distorNbList),1)
        sumInd = 0

        for i in range(len(self.distorNbList)):

            content_amb[i] = torch.sqrt(video_amb[sumInd:sumInd+self.distorNbList[i]].mean())
            sumInd += self.distorNbList[i]

        return torch.clamp(content_amb,0.01,0.99)

    def dInitWithIncons(self,dataInt):

        exp_incons = torch.pow(self.incons.unsqueeze(0).expand(dataInt.size(0),self.incons.size(0)),2)

        video_amb = (torch.pow(self.trueScoresBiasMatrix()-dataInt.float(),2)-exp_incons).mean(dim=1)
        content_amb = torch.zeros(len(self.distorNbList),1)
        sumInd = 0

        print(video_amb)
        for i in range(len(self.distorNbList)):

            content_amb[i] = torch.sqrt(video_amb[sumInd:sumInd+self.distorNbList[i]].mean())
            sumInd += self.distorNbList[i]


        return content_amb

    def iInitBase(self,dataInt):

        res = torch.sqrt((torch.pow(self.trueScoresBiasMatrix()-dataInt.float(),2)).mean(dim=0))
        res = torch.clamp(res,0.01,0.99)

        return res

    def updateEmpirical(self):

        self.bias  = torch.cat((self.bias_freez,self.bias_opti),dim=0)
        self.incons  = torch.cat((self.incons_freez,self.incons_opti),dim=0)
        self.diffs = torch.cat((self.diffs_freez,self.diffs_opti),dim=0)

        for key in self.paramProc.keys():

            if key == "bias":
                self.disDict["bias"] = Normal(torch.zeros(1), np.power(getattr(self,key).std().detach().numpy(),2)*torch.eye(1))
            else:

                sigTensor = torch.sigmoid(getattr(self,key))
                mean,var = sigTensor.mean(),torch.pow(sigTensor.std(),2)
                alpha,beta = generateData.meanvar_to_alphabeta(mean,var)
                self.disDict[key] = Beta(alpha, beta)

    def empiricalPrior(self,loss):
        #print("Adding prior term")
        for param in self.disDict.keys():
            procFunc = self.paramProc[param]
            loss -= self.disDict[param].log_prob(procFunc(getattr(self,param+"_opti"))).sum()

        return loss

    def unifPrior(self,loss):
        return loss

    def oraclePrior(self,loss):
        for param in self.disDict.keys():
            procFunc = self.paramProc[param]
            loss -= self.disDict[param].log_prob(procFunc(getattr(self,param+"_opti"))).sum()

        return loss

    def setPrior(self,priorName,dataset):

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

    def newtonUpdate(xInt):

        x = xInt.float()

        #Matrix containing the sum of all (video_amb,annot_incons) possible pairs
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

def modelMaker(annot_nb,nbVideos,distorNbList,polyDeg,score_dis,score_min,score_max,div_beta_var,\
               nbFreezTrueScores,nbFreezBias,nbFreezDiffs,nbFreezIncons,priorUpdateFrequ):
    '''Build a model
    Args:
        annot_nb (int): the bumber of annotators
        nbVideos (int): the number of videos in the dataset
        distorNbList (list): the number of distortion video for each reference video in the dataset
    Returns:
        the built model
    '''

    model = MLE(nbVideos,len(distorNbList),annot_nb,distorNbList,polyDeg,score_dis,score_min,score_max,div_beta_var,\
                nbFreezTrueScores,nbFreezBias,nbFreezDiffs,nbFreezIncons,priorUpdateFrequ)
    return model

if __name__ == '__main__':

    mle = MLE(192,24,28)

    mle(torch.ones((192,28)))
