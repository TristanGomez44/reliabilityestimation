from args import ArgReader
from args import str2bool

from torch.distributions.beta import Beta
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.multinomial import Multinomial
import argparse

import matplotlib.gridspec as gridspec

import torch
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

from matplotlib.lines import Line2D
import matplotlib.cm as cm
import os

import modelBuilder
def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--bias_std', metavar='STD',type=float,default=0.25,help='The standard deviation of the bias gaussian distribution. Ignored if bias are sampled from a beta distribution')

    argreader.parser.add_argument('--incons_alpha', metavar='STD',type=float,default=1,help='The alpha parameter of the inconsistency beta distribution')
    argreader.parser.add_argument('--incons_beta', metavar='STD',type=float,default=10,help='The beta parameter of the inconsistency beta distribution')
    argreader.parser.add_argument('--diff_alpha', metavar='STD',type=float,default=1,help='The alpha parameter of the difficulty beta distribution')
    argreader.parser.add_argument('--diff_beta', metavar='STD',type=float,default=10,help='The beta parameter of the difficulty beta distribution')

    argreader.parser.add_argument('--nb_annot', metavar='STD',type=int,default=30,help='The number of annotators')
    argreader.parser.add_argument('--nb_video_per_content', metavar='STD',type=int,default=8,help='The number of videos per content')
    argreader.parser.add_argument('--nb_content', metavar='STD',type=int,default=25,help='The number of content')

    argreader.parser.add_argument('--dataset_id', metavar='STD',type=str,default=0,help='The dataset name')
    argreader.parser.add_argument('--sample_mode',metavar="MODE",type=str,default="clip",help='The sample mode for the true scores. Can be \'continuous\' to generate continuous raw scores, \
                                                                                                                            \'clip\' : sample continuous score and clip them to integer values\
                                                                                                                            \'multinomial\' : to sample from a multinomial distribution built with the continuous density.')

    argreader.parser.add_argument('--init_from',metavar='DATASETNAME',type=str,help='A new dataset can be created by removing lines or columns from of previously created one. \
                                  This argument allow this and its value should be the name of the older dataset.')

    argreader.parser.add_argument('--seed_mat',metavar='SEED',type=int,help='The seed to use to generate the score matrix. \
                                    If unspecified, The RNG state will not be reset after generating parameters.')

    argreader.parser.add_argument('--plot',action='store_true',help='To plot the generated dataset.')

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    #Case where the dataset is created from scratch
    if not args.init_from:

        #Write the arguments in a config file so the experiment can be re-run
        argreader.writeConfigFile("../data/{}.ini".format(args.dataset_id))

        trueScoreDis = Uniform(args.score_min,args.score_max)
        diffDis = Beta(args.diff_alpha, args.diff_beta)
        inconsDis = Beta(args.incons_alpha, args.incons_beta)

        if args.bias_dis == "Normal":
            biasDis = Normal(torch.zeros(1), args.bias_std*torch.eye(1))
        elif args.bias_dis == "Beta":
            biasDis = Beta(2,2)
        else:
            raise ValueError("Unkown bias distribution : ",args.bias_dis)

        nb_videos = args.nb_video_per_content*args.nb_content

        print("Generating parameters")
        trueScores = trueScoreDis.sample((nb_videos,)).double()
        trueScores[-1] = 4.9

        diffs = diffDis.sample((args.nb_content,)).double()
        incons = inconsDis.sample((args.nb_annot,)).double()
        bias = biasDis.sample((args.nb_annot,)).double()

        if args.bias_dis == "Beta":
            bias = (bias-0.5)*args.bias_ampl

        #Sort the true score and the bias for better visualisation
        trueScores = trueScores.sort()[0]
        bias = bias.sort(dim=0)[0]

        if args.seed_mat:
            torch.manual_seed(args.seed_mat)
            random.seed(args.seed_mat)
            np.random.seed(args.seed_mat)

        distorNbList = [args.nb_video_per_content for i in range(args.nb_content)]
        model = modelBuilder.modelMaker(args.nb_annot,nb_videos,distorNbList,args)

        print("Preparing tensors")
        model.preprocessAndSet(trueScores,"trueScores",args.score_dis,args.truescores_tanh,args.bias_tanh,args.bias_ampl)
        model.preprocessAndSet(bias,"bias",args.score_dis,args.truescores_tanh,args.bias_tanh,args.bias_ampl)
        model.preprocessAndSet(incons,"incons",args.score_dis,args.truescores_tanh,args.bias_tanh,args.bias_ampl)
        model.preprocessAndSet(diffs,"diffs",args.score_dis,args.truescores_tanh,args.bias_tanh,args.bias_ampl)

        print("Sampling scores")
        scoreDis = model.compScoreDis(x_is_cuda=False)
        scoreMat = scoreDis.sample()

        if args.sample_mode == "continuous":
            scoreMat = betaDenormalize(scoreMat,args.score_min,args.score_max)

        elif args.sample_mode == "clip":
            ampl = args.score_max-args.score_min+1
            scoreMat = torch.clamp(torch.ceil(ampl*scoreMat),args.score_min,args.score_max)

        if args.plot:

            print("Ploting the data generated")
            colors = cm.rainbow(np.linspace(0, 1, nb_videos))

            markers = [m for m, func in Line2D.markers.items() if func != 'nothing' and m not in Line2D.filled_markers]
            if args.nb_annot == 5:
                markers = [".",",","v","1","s"]
            if len(markers) < args.nb_annot:
                markers = ["" for i in range(args.nb_annot)]
            else:
                markers = markers[:args.nb_annot]

            gridspec.GridSpec(3+nb_videos,1)
            fontSize = 20

            fig = plt.figure(figsize=(10,11))
            plt.subplots_adjust(hspace=1.5,wspace = 0)
            #ax = fig.add_subplot((1+nb_videos)*100+1*10+1,figsize=(10,5))
            densityAxis = plt.subplot2grid((3+nb_videos,1), (0,0), colspan=1, rowspan=3)
            box = densityAxis.get_position()
            densityAxis.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            densityAxis.set_xlabel("Normalized continuous raw score",fontsize=fontSize)
            densityAxis.set_ylabel("Density",fontsize=fontSize)

            #Plot the legend
            handlesVid = []
            for i in range(nb_videos):
                handlesVid += densityAxis.plot((0,0),marker='',color=colors[i],label=i)
            legVid = plt.legend(handles=handlesVid, loc='right' ,title="Videos",fontsize=fontSize,bbox_to_anchor=(1.35, 0.5))
            plt.setp(legVid.get_title(),fontsize=fontSize)
            plt.gca().add_artist(legVid)
            handlesAnnot = []
            for j in range(args.nb_annot):
                handlesAnnot += densityAxis.plot((0,0),marker=markers[j],color="black",label=j)
            legAnnot = plt.legend(handles=handlesAnnot, loc='right' ,title="Annotators",fontsize=fontSize,bbox_to_anchor=(1.4, -0.7),markerscale=4)
            plt.setp(legAnnot.get_title(),fontsize=fontSize)
            plt.gca().add_artist(legAnnot)

            dx = 0.01
            x_coord = torch.arange(0,1,dx)
            plt.xlim(0,1)

            cdfs = torch.exp(scoreDis.log_prob(x_coord.unsqueeze(1).unsqueeze(1).double()))

            for i in range(nb_videos):
                for j in range(args.nb_annot):

                    densityAxis.plot(x_coord.numpy(),cdfs[:,i,j].cpu().detach().numpy(),color=colors[i],marker=markers[j])
                    densityAxis.set_ylim(0,20)

                #Plot score histograms
                #scoreNorm = betaNormalize(scoreMat[i],args.score_min,args.score_max)
                scoreNorm = (scoreMat[i] - args.score_min)/(args.score_max+1-args.score_min)+1/(2*(args.score_max+1-args.score_min))

                histAx = plt.subplot2grid((3+nb_videos,1), (3+i,0), colspan=1, rowspan=1)

                if i  == nb_videos-1:
                    histAx.set_xlabel("Raw scores",fontsize=fontSize)
                if i == int(nb_videos//2):
                    histAx.set_ylabel("Empirical count",fontsize=fontSize)

                box = histAx.get_position()
                histAx.set_position([box.x0, box.y0, box.width * 0.8, box.height])

                plt.hist(scoreNorm,color=colors[i],width=1/(args.score_max-args.score_min+1),range=(0,1),bins=args.score_max-args.score_min+1)
                plt.xlim(0,1)
                plt.ylim(0,args.nb_annot+1)

                scores = np.arange(args.score_min,args.score_max+1)
                plt.xticks(betaNormalize(scores,args.score_min,args.score_max,rawScore=True),scores)
            plt.savefig("../vis/{}_scoreDis.png".format(args.dataset_id))

        csv = "videos\tencode"+"".join(["\t{}".format(i) for i in range(args.nb_annot)])+"\n"

        for i in range(0,nb_videos):
            csv += str(i//args.nb_video_per_content+1)+"\tvid{}".format(i)+"".join("\t{}".format(scoreMat[i,j]) for j in range(args.nb_annot))+"\n"

        with open("../data/{}_scores.csv".format(args.dataset_id),"w") as text_file:
            print(csv,file=text_file)

        np.savetxt("../data/{}_trueScores.csv".format(args.dataset_id),trueScores.numpy(),delimiter="\t")
        np.savetxt("../data/{}_diffs.csv".format(args.dataset_id),diffs.numpy(),delimiter="\t")
        np.savetxt("../data/{}_incons.csv".format(args.dataset_id),incons.numpy(),delimiter="\t")
        np.savetxt("../data/{}_bias.csv".format(args.dataset_id),bias.numpy(),delimiter="\t")

        print("Finished generating {}".format(args.dataset_id))

    #Case where the dataset is created from removing lines or columns from an existing dataset (artificial or not)
    else:

        #Write the arguments in a config file
        argreader.writeConfigFile("../data/{}{}.ini".format(args.init_from,args.dataset_id))

        #The number of contents of the old dataset:
        old_scores = np.genfromtxt("../data/{}_scores.csv".format(args.init_from),delimiter="\t",dtype=str)
        videoRef = old_scores[1:,0]
        nb_content_old = len(np.unique(videoRef))
        nb_annot_old = old_scores.shape[1]-2

        #Checking if the number of video per reference if constant
        vidDict = {}
        for video in videoRef:
            if video in vidDict:
                vidDict[video] += 1
            else:
                vidDict[video] = 1
        constantVideoPerRef = (len(list(set([vidDict[ref] for ref in vidDict.keys()]))) == 1)

        nb_video_per_content_old = len(videoRef)//nb_content_old if constantVideoPerRef else None

        #the permutation use for permuting the annotators randomly
        perm = np.random.permutation(nb_annot_old)

        #If the dataset to use is an artififical one, ground truth parameters are known
        #and should be processed too
        if os.path.exists("../data/{}.ini".format(args.init_from)):

            trueScores = np.genfromtxt("../data/{}_trueScores.csv".format(args.init_from),delimiter="\t")

            #Reshaping the true score vector as a 2D tensor with shape (NB_CONTENT,NB_VIDEOS_PER_CONTENT)
            #With sush a shape, it is easy to remove all videos corresponding to specific contents
            if constantVideoPerRef:
                trueScores = trueScores.reshape(nb_content_old,nb_video_per_content_old)
                trueScores = trueScores[:args.nb_content,:args.nb_video_per_content]
                trueScores = trueScores.reshape(-1)

            bias = np.genfromtxt("../data/{}_bias.csv".format(args.init_from),delimiter="\t")
            incons = np.genfromtxt("../data/{}_incons.csv".format(args.init_from),delimiter="\t")

            bias,incons = bias[perm][:args.nb_annot],incons[perm][:args.nb_annot]

            diffs = np.genfromtxt("../data/{}_diffs.csv".format(args.init_from),delimiter="\t")[:args.nb_content]

            np.savetxt("../data/{}{}_trueScores.csv".format(args.init_from,args.dataset_id),trueScores,delimiter="\t")
            np.savetxt("../data/{}{}_bias.csv".format(args.init_from,args.dataset_id),bias,delimiter="\t")
            np.savetxt("../data/{}{}_incons.csv".format(args.init_from,args.dataset_id),incons,delimiter="\t")
            np.savetxt("../data/{}{}_diffs.csv".format(args.init_from,args.dataset_id),diffs,delimiter="\t")

        scoreMat = np.genfromtxt("../data/{}_scores.csv".format(args.init_from),delimiter="\t",dtype=str)
        scoreMat[:,2:] = np.transpose(np.transpose(scoreMat[:,2:])[perm])

        #Reshaping the matrix score as a 3D tensor with shape (NB_CONTENT,NB_VIDEOS_PER_CONTENT,NB_ANNOTATORS)
        #With sush a shape, it is easy to remove all videos corresponding to specific contents
        if constantVideoPerRef:
            scoreMat = scoreMat[1:].reshape(nb_content_old,nb_video_per_content_old,nb_annot_old+2)
            scoreMat = scoreMat[:args.nb_content,:args.nb_video_per_content,:args.nb_annot+2]
            scoreMat = scoreMat.reshape(scoreMat.shape[0]*scoreMat.shape[1],scoreMat.shape[2])

        else:
            scoreMat = scoreMat[1:,:args.nb_annot+2]

        header = "videos\tencode"+"".join(["\t{}".format(i) for i in range(args.nb_annot)])
        np.savetxt("../data/{}{}_scores.csv".format(args.init_from,args.dataset_id),scoreMat.astype(str),header=header,delimiter="\t",fmt='%s',comments='')

        print("Finished generating {}{}".format(args.init_from,args.dataset_id))

def meanvar_to_alphabeta(mean,var):
    ''' Compute the alpha and beta parameters of a beta distribution having some mean and variance

    Args:
        mean (float): the mean of the beta distribution
        var (float): the variance of the beta distribution
    Returns:
        the alpha and beta parameters of the beta distribution
     '''

    alpha = ((1-mean)/var-1/mean)*torch.pow(mean,2)

    beta = alpha*(1/mean-1)

    return alpha,beta

def identity(x,score_min,score_max):
    return x

def betaNormalize(x,score_min,score_max,rawScore=False,variance=False):
    ''' Normalize scores between 1 and 5 to score between 0 and 1
    Args:
        x (torch.tensor or numpy.array): the value to normalise
        score_min (int): the mininmum unnormalised score
        score_max (int): the maximum unnormalised score
        rawScore (bool): indicates the values being normalized are raw scores or not.
        variance (bool): indicates if the values being normalized are variance, in which case they just need to be reduced and not centered
    Returns:
        the scores normalised
    '''

    #The 0.1 value helps for numerical stability, because beta distribution can go up to
    #plus infinity when x goes to 0 or 1

    if rawScore:

        res = (x-score_min)/(score_max-score_min)

        #Re-normalising to [0.1,0.9] because the raw score can be exactly 0 or 1,
        #and the beta density is equal to 0 at those values
        halfInterv = 1/(2*(score_max-score_min+1))
        high = 1-halfInterv
        low = 0+halfInterv

        return res*(high-low)+low

    else:

        if variance:
            return x/(score_max-score_min)
        else:
            return (x-score_min)/(score_max-score_min)

def betaDenormalize(x,score_min,score_max):
    ''' Un-normalize scores between 0 and 1 to score between 1 and 5
    Args:
        x (torch.tensor or numpy.array): the value to un-normalise
        score_min (int): the mininmum unnormalised score
        score_max (int): the maximum unnormalised score
    Returns:
        the scores un-normalized
    '''

    #The 0.1 value helps for numerical stability, because beta distribution can go up to
    #plus infinity when x goes to 0 or 1

    low = score_min
    high = score_max

    return x*(high-low)+low

if __name__ == "__main__":
    main()
