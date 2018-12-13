from args import ArgReader
from args import str2bool

from torch.distributions.beta import Beta
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.multinomial import Multinomial

import torch
import glob
import numpy as np
def main(argv=None):


    parser = argparse.ArgumentParser(description='Plot the accuracy across epoch')

    parser.add_argument('--seed', metavar='STD',type=int,default=0,help='The seed to generate random numbers')

    parser.add_argument('--bias_std', metavar='STD',type=float,default=0.25,help='The upper limit for the y axis ')
    parser.add_argument('--incons_alpha', metavar='STD',type=float,default=2,help='The alpha parameter of the inconsistency distribution')
    parser.add_argument('--incons_beta', metavar='STD',type=float,default=2,help='The beta parameter of the inconsistency distribution')
    parser.add_argument('--diff_alpha', metavar='STD',type=float,default=2,help='The alpha parameter of the difficulty distribution')
    parser.add_argument('--diff_beta', metavar='STD',type=float,default=2,help='The beta parameter of the difficulty distribution')

    parser.add_argument('--nb_annot', metavar='STD',type=int,default=30,help='The number of annotators')
    parser.add_argument('--nb_video_per_content', metavar='STD',type=float,default=8,help='The number of videos per content')
    parser.add_argument('--nb_content', metavar='STD',type=float,default=25,help='The number of content')

    parser.add_argument('--dataset_id', metavar='STD',type=int,default=0,help='The dataset id')

    #Getting the args from command line and config file
    args = parser.parse_args()

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

            probs = torch.zeros((5))
            for k in range(1,6):
                probs[k-1] = normDis.cdf(torch.tensor(k))
            discDis = Multinomial(probs=probs)

            scores[i,j] = discDis.sample().max(0)[1]+1

    scores = scores.int()
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
