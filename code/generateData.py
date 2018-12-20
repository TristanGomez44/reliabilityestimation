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

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    #Write the arguments in a config file so the experiment can be re-run
    argreader.writeConfigFile("../data/artifData{}.ini".format(args.dataset_id))

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    trueScoreDis = Uniform(1,5)
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

    for i in range(nb_videos):
        for j in range(args.nb_annot):

            normDis = Normal(trueScores[i]+bias[j], diffs[i//args.nb_video_per_content]+incons[j])

            if not args.continuous:
                probs = torch.zeros((5))
                for k in range(1,6):
                    probs[k-1] = normDis.cdf(torch.tensor(k))
                discDis = Multinomial(probs=probs)

                scores[i,j] = discDis.sample().max(0)[1]+1
            else:
                scores[i,j] = normDis.sample()

    #scores = scores.int()
    csv = "videos\tencode"+"".join(["\t{}".format(i) for i in range(args.nb_annot)])+"\n"

    for i in range(0,nb_videos):
        csv += str(i//args.nb_video_per_content+1)+"\tvid{}".format(i)+"".join("\t{}".format(scores[i-1,j]) for j in range(args.nb_annot))+"\n"

    with open("../data/artifData{}_scores.csv".format(args.dataset_id),"w") as text_file:
        print(csv,file=text_file)

    np.savetxt("../data/artifData{}_trueScores.csv".format(args.dataset_id),trueScores.numpy(),delimiter="\t")
    np.savetxt("../data/artifData{}_diffs.csv".format(args.dataset_id),diffs.numpy(),delimiter="\t")
    np.savetxt("../data/artifData{}_incons.csv".format(args.dataset_id),incons.numpy(),delimiter="\t")
    np.savetxt("../data/artifData{}_bias.csv".format(args.dataset_id),bias[:,0,0].numpy(),delimiter="\t")

if __name__ == "__main__":
    main()
