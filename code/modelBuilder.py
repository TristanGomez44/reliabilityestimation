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
class MLE(nn.Module):
    ''' Implement the model proposed in Zhi et al. (2017) : Recover Subjective Quality Scores from Noisy Measurements'''

    def __init__(self,videoNb,contentNb,annotNb,distorNbList):

        super(MLE, self).__init__()

        self.annot_bias  = nn.Parameter(torch.ones(annotNb))
        self.annot_incons  = nn.Parameter(torch.ones(annotNb))
        self.video_amb  = nn.Parameter(torch.ones(contentNb))
        self.video_scor  = nn.Parameter(torch.ones(videoNb))
        self.annotNb = annotNb
        self.videoNb = videoNb
        self.contentNb = contentNb

        self.distorNbList = distorNbList

    def forward(self,x):

        #Matrix containing the sum of all (video_amb,annot_incons) possible pairs
        #amb = self.video_amb.unsqueeze(1).expand(self.contentNb, self.distorNb).contiguous().view(-1)
        tmp = []
        for i in range(self.contentNb):
            tmp.append(self.video_amb[i].unsqueeze(0).expand(self.distorNbList[i]))

        amb = torch.cat(tmp,dim=0)
        amb_sq = torch.pow(amb,2)
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

    def initParams(self,data):

        annot_bias = torch.zeros(data.size(1))
        #Use std to initialise inconsistency seems a bad idea.
        #Even really consistent annotators can give score
        #in a large range
        annot_incons = data.std(dim=0)
        vid_score = data.mean(dim=1)

        #Computing initial values for video ambiguity

        #data3D = data.view(self.distorNb,self.contentNb,self.annotNb).permute(1,0,2)
        data3D = []
        currInd = 0
        video_ambTensor = torch.ones(self.contentNb)
        for i in range(self.contentNb):

            #The data for all videos made from this reference video
            videoData = data[currInd:currInd+self.distorNbList[i]]

            data3D.append(videoData)
            currInd += self.distorNbList[i]

            expanded_use_score_mean = data.mean(dim=0).unsqueeze(1).permute(1, 0).expand_as(videoData)
            video_ambTensor[i] = torch.pow(videoData-expanded_use_score_mean,2).sum()/(self.annotNb*self.contentNb)
            video_ambTensor[i] = torch.sqrt(video_ambTensor[i])

        self.setParams(annot_bias,annot_incons,video_ambTensor,vid_score)
        #print(self.video_scor.detach().numpy())
def subRej(data):

    p = torch.zeros(data.size(1))
    q = torch.zeros(data.size(1))
    kurt = torch.zeros(data.size(0))
    eps = torch.zeros(data.size(0))
    for i in range(data.size(0)):

        mom_sec = moment(data[i].numpy(),2)
        kurt = moment(data[i].numpy(),4)/(mom_sec*mom_sec)

        if 2 <= kurt and kurt <= 4:
            eps = 2
        else:
            eps = math.sqrt(20)

        vidMeans = data.mean(dim=1)
        vidStds = data.std(dim=1)
        for j in range(data.size(1)):
            if data[i,j] >= vidMeans[j]+ eps*vidStds[j]:
                p[j] += 1
            if data[i,j] <= vidMeans[j]- eps*vidStds[j]:
                q[j] += 1
    rejList = []
    for j in range(data.size(1)):
        if ((p[j]+q[j])/data.size(1) >= 0.05) and (torch.abs((p[j]-q[j])/(p[j]+q[j]))<0.3):
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

def MOS(data,z_score,sub_rej):

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

    return mos_mean.numpy(),mos_conf.numpy()

def modelMaker(annot_nb,nbVideos,distorNbList):
    '''Build a model
    Args:
        annot_nb (int): the bumber of annotators
        nbVideos (int): the number of videos in the dataset
        distorNbList (list): the number of distortion video for each reference video in the dataset
    Returns:
        the built model
    '''

    model = MLE(nbVideos,len(distorNbList),annot_nb,distorNbList)
    return model

if __name__ == '__main__':

    mle = MLE(192,24,28)

    #mle.setParams(torch.rand(28),torch.rand(28),torch.rand(24),torch.rand(192))
    mle(torch.ones((192,28)))

    mle.initParams(torch.ones((192,28)))
