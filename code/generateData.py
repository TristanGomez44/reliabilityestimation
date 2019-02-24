from args import ArgReader
from args import str2bool

from torch.distributions.beta import Beta
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.multinomial import Multinomial
import argparse

import torch
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

from matplotlib.lines import Line2D
import matplotlib.cm as cm
import os
def main(argv=None):


    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--bias_std', metavar='STD',type=float,default=0.25,help='The standard deviation of the bias gaussian distribution')
    argreader.parser.add_argument('--incons_alpha', metavar='STD',type=float,default=2,help='The alpha parameter of the inconsistency beta distribution')
    argreader.parser.add_argument('--incons_beta', metavar='STD',type=float,default=2,help='The beta parameter of the inconsistency beta distribution')
    argreader.parser.add_argument('--diff_alpha', metavar='STD',type=float,default=2,help='The alpha parameter of the difficulty beta distribution')
    argreader.parser.add_argument('--diff_beta', metavar='STD',type=float,default=2,help='The beta parameter of the difficulty beta distribution')

    argreader.parser.add_argument('--nb_annot', metavar='STD',type=int,default=30,help='The number of annotators')
    argreader.parser.add_argument('--nb_video_per_content', metavar='STD',type=int,default=8,help='The number of videos per content')
    argreader.parser.add_argument('--nb_content', metavar='STD',type=int,default=25,help='The number of content')

    argreader.parser.add_argument('--dataset_id', metavar='STD',type=str,default=0,help='The dataset name')
    argreader.parser.add_argument('--continuous',action='store_true',help='To generate continuous scores instead of discrete ones')

    argreader.parser.add_argument('--init_from',metavar='DATASETNAME',type=str,help='A new dataset can be created by just removing parameters of an older one.\
                                  This argument allow this and its value should be the name of the older dataset.')

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
        biasDis = Normal(torch.zeros(1), args.bias_std*torch.eye(1))

        nb_videos = args.nb_video_per_content*args.nb_content

        trueScores = trueScoreDis.sample((nb_videos,))
        diffs = diffDis.sample((args.nb_content,))
        incons = inconsDis.sample((args.nb_annot,))
        bias = biasDis.sample((args.nb_annot,))
        #print(bias)
        scores = torch.zeros((nb_videos,args.nb_annot))

        dx = 0.01
        x_coord = torch.arange(0,1,dx)

        meanList = []

        colors = cm.rainbow(np.linspace(0, 1, nb_videos))

        markers = [m for m, func in Line2D.markers.items() if func != 'nothing' and m not in Line2D.filled_markers]
        if args.nb_annot == 5:
            markers = [".",",","v","1","s"]
        if len(markers) < args.nb_annot:
            markers = ["" for i in range(args.nb_annot)]
        else:
            markers = markers[:args.nb_annot]

        plt.figure(figsize=(10,5))
        plt.xlim(0,1)

        for i in range(nb_videos):
            for j in range(args.nb_annot):

                mean = trueScores[i]+bias[j]
                var = torch.pow(diffs[i//args.nb_video_per_content],2)+torch.pow(incons[j],2)
                #print("diff",diffs[i//args.nb_video_per_content],"incons",incons[j])
                if args.score_dis == "Beta":

                    mean = torch.clamp(mean,args.score_min,args.score_max)
                    mean = betaNormalize(mean,args.score_min,args.score_max)

                    #mean = torch.sigmoid(mean)
                    #print("mean",mean,"var",var/args.div_beta_var)
                    alpha,beta = meanvar_to_alphabeta(mean,var/args.div_beta_var)

                    scoresDis = Beta(alpha,beta)
                    postProcessingFunc = lambda x:betaDenormalize(x,args.score_min,args.score_max)
                elif args.score_dis == "Normal":
                    scoresDis = Normal(trueScores[i]+bias[j], diffs[i//args.nb_video_per_content]+incons[j])
                    postProcessingFunc = lambda x:identity(x,args.score_min,args.score_max)
                else:
                    raise ValueError("Unkown distribution : {}".format(scoreDis))

                if not args.continuous:
                    probs = torch.zeros((args.score_max+1-args.score_min))
                    for k in range(args.score_min,args.score_max+1):
                        if args.score_dis == "Beta":
                            value = torch.tensor(betaNormalize(k,args.score_min,args.score_max)).float()
                            probs[k-1] = torch.exp(scoresDis.log_prob(value))
                        else:
                            probs[k-1] = torch.exp(scoresDis.log_prob(k))

                    discDis = Multinomial(probs=probs)
                    score = (discDis.sample().max(0)[1]+1).float()

                    if args.score_dis == "Beta":
                        scores[i,j] = betaNormalize(score,args.score_min,args.score_max)
                    else:
                        scores[i,j] = score
                else:
                    scores[i,j] = scoresDis.sample()

                cdf = lambda x: torch.exp(scoresDis.log_prob(x))
                #print("video",i,"annot",j)

                plt.plot(x_coord.numpy(),cdf(x_coord).cpu().detach().numpy()[0],color=colors[i],marker=markers[j])

                meanList.append(mean)

                scores[i,j] = postProcessingFunc(scores[i,j])

            #Plot score histograms
            scoreNorm = (scores[i] - args.score_min)/(args.score_max+1-args.score_min)
            plt.hist(scoreNorm,color=colors[i],width=1/(args.score_max-args.score_min+1),alpha=0.5,range=(0,1))

        #Plot the legend
        handlesVid = []
        for i in range(nb_videos):
            handlesVid += plt.plot((0,0),marker='',color=colors[i],label=i)
        legVid = plt.legend(handles=handlesVid, loc='upper right' ,title="Videos")
        plt.gca().add_artist(legVid)
        handlesAnnot = []
        for j in range(args.nb_annot):
            handlesAnnot += plt.plot((0,0),marker=markers[j],color="black",label=j)
        legAnnot = plt.legend(handles=handlesAnnot, loc='lower right' ,title="Annotators")
        plt.gca().add_artist(legAnnot)

        plt.savefig("../vis/{}_scoreDis.png".format(args.dataset_id))

        #scores = scores.int()
        csv = "videos\tencode"+"".join(["\t{}".format(i) for i in range(args.nb_annot)])+"\n"

        for i in range(0,nb_videos):
            csv += str(i//args.nb_video_per_content+1)+"\tvid{}".format(i)+"".join("\t{}".format(scores[i,j]) for j in range(args.nb_annot))+"\n"

        with open("../data/{}_scores.csv".format(args.dataset_id),"w") as text_file:
            print(csv,file=text_file)

        np.savetxt("../data/{}_trueScores.csv".format(args.dataset_id),trueScores.numpy(),delimiter="\t")
        np.savetxt("../data/{}_diffs.csv".format(args.dataset_id),diffs.numpy(),delimiter="\t")
        np.savetxt("../data/{}_incons.csv".format(args.dataset_id),incons.numpy(),delimiter="\t")
        np.savetxt("../data/{}_bias.csv".format(args.dataset_id),bias[:,0,0].numpy(),delimiter="\t")

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

        scores = np.genfromtxt("../data/{}_scores.csv".format(args.init_from),delimiter="\t",dtype=str)
        scores[:,2:] = np.transpose(np.transpose(scores[:,2:])[perm])

        #Reshaping the matrix score as a 3D tensor with shape (NB_CONTENT,NB_VIDEOS_PER_CONTENT,NB_ANNOTATORS)
        #With sush a shape, it is easy to remove all videos corresponding to specific contents
        if constantVideoPerRef:
            scores = scores[1:].reshape(nb_content_old,nb_video_per_content_old,nb_annot_old+2)
            scores = scores[:args.nb_content,:args.nb_video_per_content,:args.nb_annot+2]
            scores = scores.reshape(scores.shape[0]*scores.shape[1],scores.shape[2])

        else:
            scores = scores[1:,:args.nb_annot+2]

        header = "videos\tencode"+"".join(["\t{}".format(i) for i in range(args.nb_annot)])
        np.savetxt("../data/{}{}_scores.csv".format(args.init_from,args.dataset_id),scores.astype(str),header=header,delimiter="\t",fmt='%s',comments='')

        print("Finished generating {}{}".format(args.init_from,args.dataset_id))

def meanvar_to_alphabeta(mean,var):
    ''' Compute alpha and beta parameters of a beta distribution having some mean and variance

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

def betaNormalize(x,score_min,score_max):
    ''' Normalize scores between 1 and 5 to score between 0 and 1
    Args:
        x (torch.tensor or numpy.array): the value to normalise
        score_min (int): the mininmum unnormalised score
        score_max (int): the maximum unnormalised score
    Returns:
        the scores normalised
    '''

    #The 0.1 value helps for numerical stability, because beta distribution can go up to
    #plus infinity when x goes to 0 or 1
    low = score_min-0.1
    high = score_max+0.1

    return (x-low)/(high-low)

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
    low = score_min-0.1
    high = score_max+0.1
    return x*(high-low)+low

if __name__ == "__main__":
    main()
