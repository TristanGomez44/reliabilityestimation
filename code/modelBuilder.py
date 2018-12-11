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

class MLE(nn.Module):
    ''' Implement the model proposed in Zhi et al. (2017) : Recover Subjective Quality Scores from Noisy Measurements'''

    def __init__(self,videoNb,contentNb,annotNb,distorNbList,polyDeg):

        super(MLE, self).__init__()

        self.annot_bias  = nn.Parameter(torch.ones(annotNb))
        self.annot_incons  = nn.Parameter(torch.ones(annotNb))
        self.video_amb  = nn.Parameter(torch.ones(contentNb,polyDeg+1))
        self.video_scor  = nn.Parameter(torch.ones(videoNb))
        self.annotNb = annotNb
        self.videoNb = videoNb
        self.contentNb = contentNb

        self.polyDeg = polyDeg
        self.distorNbList = distorNbList

    def forward(self,xInt):

        x = xInt.float()

        #Matrix containing the sum of all (video_amb,annot_incons) possible pairs
        #amb = self.video_amb.unsqueeze(1).expand(self.contentNb, self.distorNb).contiguous().view(-1)
        tmp = []
        for i in range(self.contentNb):
            tmp.append(self.video_amb[i].unsqueeze(1).expand(self.polyDeg+1,self.distorNbList[i]))

        amb = torch.cat(tmp,dim=1).permute(1,0)
        #amb = amb.expand(amb.size(0),self.polyDeg)
        #print(amb.size())

        vid_means = x.mean(dim=1)
        vid_means = vid_means.unsqueeze(1).expand(vid_means.size(0),self.polyDeg+1)
        powers = torch.arange(self.polyDeg+1).float()
        if xInt.is_cuda:
            powers = powers.cuda()
        powers = powers.unsqueeze(1).permute(1,0).expand(vid_means.size(0),self.polyDeg+1)
        vid_means_pow = torch.pow(vid_means,powers)
        amb_pol = (amb*vid_means_pow).sum(dim=1)

        amb_sq = torch.pow(amb_pol,2)
        amb_sq = amb_sq.unsqueeze(1).expand(self.videoNb, self.annotNb)
        incon_sq = torch.pow(self.annot_incons,2)
        incon_sq = incon_sq.unsqueeze(0).expand(self.videoNb, self.annotNb)
        amb_incon = amb_sq+incon_sq

        #Matrix containing the sum of all (video_scor,annot_bias) possible pairs
        scor = self.video_scor.unsqueeze(1).expand(self.videoNb, self.annotNb)
        bias = self.annot_bias.unsqueeze(0).expand(self.videoNb, self.annotNb)
        scor_bias = scor+bias

        log_proba = (-torch.log(amb_incon)-torch.pow(x-scor_bias,2)/(amb_incon)).sum()

        return -log_proba

    def setParams(self,annot_bias,annot_incons,video_amb,video_scor):

        self.annot_bias = nn.Parameter(annot_bias)
        self.annot_incons = nn.Parameter(annot_incons)
        self.video_amb = nn.Parameter(video_amb)
        self.video_scor = nn.Parameter(video_scor)

    def initParams(self,dataInt):

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

            expanded_use_score_mean = data.mean(dim=0).unsqueeze(1).permute(1, 0).expand_as(videoData)
            video_ambTensor[i,0] = torch.pow(videoData-expanded_use_score_mean,2).sum()/(self.annotNb*self.contentNb)
            video_ambTensor[i,0] = torch.sqrt(video_ambTensor[i,0])


        self.setParams(annot_bias,annot_incons,video_ambTensor,vid_score)

    def newtonUpdate(xInt):

        x = xInt.float()

        #Matrix containing the sum of all (video_amb,annot_incons) possible pairs
        #amb = self.video_amb.unsqueeze(1).expand(self.contentNb, self.distorNb).contiguous().view(-1)
        tmp = []
        for i in range(self.contentNb):
            tmp.append(self.video_amb[i].expand(self.distorNbList[i]))

        amb = torch.cat(tmp,dim=1).unsqueeze(0)
        amb_sq = torch.pow(amb,2)
        amb_sq = amb_sq.unsqueeze(1).expand(self.videoNb, self.annotNb)
        incon_sq = torch.pow(self.annot_incons,2)
        incon_sq = incon_sq.unsqueeze(0).expand(self.videoNb, self.annotNb)
        w_es = amb_sq+incon_sq

        scor = self.video_scor.unsqueeze(1).expand(self.videoNb, self.annotNb)
        bias = self.annot_bias.unsqueeze(0).expand(self.videoNb, self.annotNb)

        scor_new = (w_es*(x-bias)/w_es).sum(dim=1)
        bias_new = (w_es*(x-scor)/w_es).sum(dim=0)

        #incon = self.annot_incons.unsqueeze(0).expand(self.videoNb, self.annotNb)
        v = self.annot_incons

        w_es_sq = torch.pow(w_es,2)

        incons_new = v - ((w_es*v-w_es_sq*v*torch.pow(x-scor-bias,2))/()).sum(dim=0)

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

def modelMaker(annot_nb,nbVideos,distorNbList,polyDeg):
    '''Build a model
    Args:
        annot_nb (int): the bumber of annotators
        nbVideos (int): the number of videos in the dataset
        distorNbList (list): the number of distortion video for each reference video in the dataset
    Returns:
        the built model
    '''

    model = MLE(nbVideos,len(distorNbList),annot_nb,distorNbList,polyDeg)
    return model

if __name__ == '__main__':

    mle = MLE(192,24,28)

    #mle.setParams(torch.rand(28),torch.rand(28),torch.rand(24),torch.rand(192))
    mle(torch.ones((192,28)))

    mle.initParams(torch.ones((192,28)))
