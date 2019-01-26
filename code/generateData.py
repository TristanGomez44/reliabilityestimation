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

    argreader.parser.add_argument('--dataset_id', metavar='STD',type=int,default=0,help='The dataset id')
    argreader.parser.add_argument('--continuous',action='store_true',help='To generate continuous scores instead of discrete ones')

    argreader.parser.add_argument('--init_from',metavar='DATASETNAME',type=str,help='A new dataset can be created by just removing parameters of an older one.\
                                  This argument allow this and its value should be the name of the older dataset.')

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    #Write the arguments in a config file so the experiment can be re-run
    argreader.writeConfigFile("../data/{}{}.ini".format(args.init_from,args.dataset_id))

    if not args.init_from:

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

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

                if not args.continuous:
                    probs = torch.zeros((args.score_max+1-args.score_min))
                    for k in range(args.score_min,args.score_max+1):

                        value = torch.tensor(betaNormalize(k,args.score_min,args.score_max)).float()
                        probs[k-1] = torch.exp(scoresDis.log_prob(value))

                    discDis = Multinomial(probs=probs)

                    scores[i,j] = betaNormalize((discDis.sample().max(0)[1]+1).float(),args.score_min,args.score_max)
                else:
                    scores[i,j] = scoresDis.sample()

                cdf = lambda x: torch.exp(scoresDis.log_prob(x))
                #print("video",i,"annot",j)
                plt.plot(x_coord.numpy(),cdf(x_coord).cpu().detach().numpy()[0],color=colors[i],marker=markers[j])


                meanList.append(mean)

                scores[i,j] = postProcessingFunc(scores[i,j])

        plt.savefig("../vis/artifData{}_scoreDis.png".format(args.dataset_id))

        #scores = scores.int()
        csv = "videos\tencode"+"".join(["\t{}".format(i) for i in range(args.nb_annot)])+"\n"

        for i in range(0,nb_videos):
            csv += str(i//args.nb_video_per_content+1)+"\tvid{}".format(i)+"".join("\t{}".format(scores[i,j]) for j in range(args.nb_annot))+"\n"

        with open("../data/artifData{}_scores.csv".format(args.dataset_id),"w") as text_file:
            print(csv,file=text_file)

        np.savetxt("../data/artifData{}_trueScores.csv".format(args.dataset_id),trueScores.numpy(),delimiter="\t")
        np.savetxt("../data/artifData{}_diffs.csv".format(args.dataset_id),diffs.numpy(),delimiter="\t")
        np.savetxt("../data/artifData{}_incons.csv".format(args.dataset_id),incons.numpy(),delimiter="\t")
        np.savetxt("../data/artifData{}_bias.csv".format(args.dataset_id),bias[:,0,0].numpy(),delimiter="\t")
    else:

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

        #If the dataset to use is an artififical one, ground truth parameters are known
        #and should be processed too
        if args.init_from.find("artif") != -1:

            trueScores = np.genfromtxt("../data/{}_trueScores.csv".format(args.init_from),delimiter="\t")

            if constantVideoPerRef:
                trueScores = trueScores.reshape(nb_content_old,nb_video_per_content_old)
                trueScores = trueScores[:args.nb_content,:args.nb_video_per_content]
                trueScores = trueScores.reshape(-1)

            bias = np.genfromtxt("../data/{}_bias.csv".format(args.init_from),delimiter="\t")[:args.nb_annot]
            incons = np.genfromtxt("../data/{}_incons.csv".format(args.init_from),delimiter="\t")[:args.nb_annot]
            diffs = np.genfromtxt("../data/{}_diffs.csv".format(args.init_from),delimiter="\t")[:args.nb_content]

            np.savetxt("../data/{}{}_trueScores.csv".format(args.init_from,args.dataset_id),trueScores,delimiter="\t")
            np.savetxt("../data/{}{}_bias.csv".format(args.init_from,args.dataset_id),bias,delimiter="\t")
            np.savetxt("../data/{}{}_incons.csv".format(args.init_from,args.dataset_id),incons,delimiter="\t")
            np.savetxt("../data/{}{}_diffs.csv".format(args.init_from,args.dataset_id),diffs,delimiter="\t")

        scores = np.genfromtxt("../data/{}_scores.csv".format(args.init_from),delimiter="\t",dtype=str)[:,:args.nb_annot]
        if constantVideoPerRef:
            scores = scores[1:].reshape(nb_content_old,nb_video_per_content_old,nb_annot_old)
            scores = scores[:args.nb_content,:args.nb_video_per_content,:args.nb_annot+2]
            scores = scores.reshape(scores.shape[0]*scores.shape[1],scores.shape[2])
        else:
            scores = scores[:,:args.nb_annot+2]

        np.savetxt("../data/{}{}_scores.csv".format(args.init_from,args.dataset_id),scores.astype(str),delimiter="\t",fmt='%s')

def meanvar_to_alphabeta(mean,var):

    #print(torch.pow(mean,2))
    alpha = ((1-mean)/var-1/mean)*torch.pow(mean,2)
    #print(alpha)
    beta = alpha*(1/mean-1)

    return alpha,beta

def identity(x,score_min,score_max):
    return x

def betaNormalize(x,score_min,score_max):

    low = score_min-0.1
    high = score_max+0.1

    return (x-low)/(high-low)

def betaDenormalize(x,score_min,score_max):
    low = score_min-0.1
    high = score_max+0.1
    return x*(high-low)+low

if __name__ == "__main__":
    main()
