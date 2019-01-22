import sys
import argparse
import configparser

def str2bool(v):
    '''Convert a string to a boolean value'''
    if v == 'True':
        return True
    elif v == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def str2FloatList(x):

    '''Convert a formated string to a list of float value'''
    if len(x.split(",")) == 1:
        return float(x)
    else:
        return [float(elem) for elem in x.split(",")]
def strToStrList(x):
    if x == "None":
        return []
    else:
        return x.split(",")

def str2StrList(x):
    '''Convert a string to a list of string value'''
    return x.split(" ")

class ArgReader():
    """
    This class build a namespace by reading arguments in both a config file
    and the command line.

    If an argument exists in both, the value in the command line overwrites
    the value in the config file

    This class mainly comes from :
    https://stackoverflow.com/questions/3609852/which-is-the-best-way-to-allow-configuration-options-be-overridden-at-the-comman
    Consulted the 18/11/2018

    """

    def __init__(self,argv):
        ''' Defines the arguments used in several scripts of this project.
        It reads them from a config file
        and also add the arguments found in command line.

        If an argument exists in both, the value in the command line overwrites
        the value in the config file
        '''

        # Do argv default this way, as doing it in the functional
        # declaration sets it at compile time.
        if argv is None:
            argv = sys.argv

        # Parse any conf_file specification
        # We make this parser with add_help=False so that
        # it doesn't parse -h and print help.
        conf_parser = argparse.ArgumentParser(
            description=__doc__, # printed with -h/--help
            # Don't mess with format of description
            formatter_class=argparse.RawDescriptionHelpFormatter,
            # Turn off help, so we print all options in response to -h
            add_help=False
            )
        conf_parser.add_argument("-c", "--conf_file",
                            help="Specify config file", metavar="FILE")
        args, self.remaining_argv = conf_parser.parse_known_args()

        defaults = {}

        if args.conf_file:
            config = configparser.SafeConfigParser()
            config.read([args.conf_file])
            defaults.update(dict(config.items("default")))

        # Parse rest of arguments
        # Don't suppress add_help here so it will handle -h
        self.parser = argparse.ArgumentParser(
            # Inherit options from config_parser
            parents=[conf_parser]
            )
        self.parser.set_defaults(**defaults)

        # Training settings
        #parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

        self.parser.add_argument('--stop_crit', type=float, metavar='M',
                            help='The threshold value under which the training stops')
        self.parser.add_argument('--epochs', type=int, metavar='N',
                            help='number of epochs to train')

        self.parser.add_argument('--lr', type=str2FloatList,metavar='LR',
                            help='learning rate (it can be a schedule : --lr 0.01,0.001,0.0001)')
        self.parser.add_argument('--num_workers', type=int,metavar='NUMWORKERS',
                            help='the number of processes to load the data. num_workers equal 0 means that it’s \
                            the main process that will do the data loading when needed, num_workers equal 1 is\
                            the same as any n, but you’ll only have a single worker, so it might be slow')
        self.parser.add_argument('--momentum', type=float, metavar='M',
                            help='SGD momentum')
        self.parser.add_argument('--seed', type=int, metavar='S',
                            help='random seed')
        self.parser.add_argument('--log_interval', type=int, metavar='N',
                            help='how many batches to wait before logging training status')
        self.parser.add_argument('--ind_id', type=int, metavar='IND_ID',
                            help='the id of the individual')
        self.parser.add_argument('--exp_id', type=str, metavar='EXP_ID',
                            help='the id of the experience')
        self.parser.add_argument('--dataset', type=str, metavar='N',help='the dataset to use. Can be \'NETFLIX\', \'IRCCYN\' or \'VQEG\'.')


        self.parser.add_argument('--poly_deg', type=int, metavar='S',
                            help='The polynomial degree to use to model video ambiguity as a function of video mean score')

        self.parser.add_argument('--erase_results', type=str2bool, metavar='S',
                            help='To erase the convergence speed results already computed')
        self.parser.add_argument('--cuda', type=str2bool, metavar='S',
                            help='To run computations on the gpu')
        self.parser.add_argument('--optim', type=str, metavar='OPTIM',
                            help='the optimizer algorithm to use (default: \'LBFGS\')')
        self.parser.add_argument('--rob_param', type=str, metavar='S',
                            help='The name of the parameter to vary during \
                                 robustness evaluation. Can be \'nb_annot\' or \'nb_corr\'')

        self.parser.add_argument('--rob_param_values', type=str2FloatList, nargs='+',metavar='V',
                            help='The values the varying parameter has to have during robustness evaluation')

        self.parser.add_argument('--nb_rep', type=int, metavar='S',
                            help='The number of repetition for each parameter value')

        self.parser.add_argument('--model_param', type=str,metavar='P',
                            help='The model parameter to vary during robustness evaluation')
        self.parser.add_argument('--model_values', type=str2FloatList, nargs='+',metavar='V',
                            help='The values the varying model parameter has to have during robustness evaluation')

        self.parser.add_argument('--start_mode', type=str,metavar='SM',
                    help='The mode to use to initialise the model. Can be \'init\' or \'fine_tune\'.')
        self.parser.add_argument('--train_mode', type=str,metavar='TM',
                    help='The mode to use to train the model. Can be \'joint\' or \'alternate\'.')
        self.parser.add_argument('--alt_epoch_nb', type=int,metavar='TM',
                    help='The number of epoch during which train each parameter. Ignored if using \'joint\' training mode.')

        self.parser.add_argument('--perc_noise', type=float,metavar='S',
                        help='For \'init_oracle\' initialisation mode, the norm of the gaussian noise added to params, relative to their norm.')
        self.parser.add_argument('--prior', type=str,metavar='S',\
                        help='The prior to use. Can be \'uniform\' or \'oracle\'.')
        self.parser.add_argument('--prior_weight', type=float,metavar='S',\
                        help='The weight of the prior term in the loss function')

        self.parser.add_argument('--param_to_opti', type=strToStrList,metavar='V',
                            help="The parameters to optimise. Can be a list with elements among 'bias','incons','diffs','trueScores'")
        self.parser.add_argument('--param_not_gt',type=strToStrList,metavar='V',
                            help="The parameters to set to ground truth when not starting the training witha pre-trained net \
                                (i.e. choosing option 'init' for --start_mode). Can be a list (possibly empty) with elements among 'bias','incons','diffs','trueScores'")

        self.parser.add_argument('--note', type=str,metavar='NOTE',
                            help="A note on the model")

        self.parser.add_argument('--norm_sum', type=str2bool, metavar='S',
                            help='To add the sum of the prob for all possible labels in the loss function. This is a term to minimise.\
                                When using integer scores, this term should be used to compute correctly the probability of a score.')

        self.parser.add_argument('--score_dis', type=str, metavar='S',
                            help='The distribution to use to model the scores')

        self.parser.add_argument('--score_min', type=int, metavar='S',
                            help='The minimum score that can be given by an annotator')
        self.parser.add_argument('--score_max', type=int, metavar='S',
                            help='The maximum score that can be given by an annotator')
        self.parser.add_argument('--div_beta_var', type=float, metavar='S',
                            help='The coefficient with which to rescale down the variances (difficulties and inconsistencies) \
                            sampled from the beta distribution')

        self.args = None

    def getRemainingArgs(self):
        ''' Reads the comand line arg'''

        self.args = self.parser.parse_args(self.remaining_argv)

    def writeConfigFile(self,filePath):
        """ Writes a config file containing all the arguments and their values"""

        config = configparser.SafeConfigParser()
        config.add_section('default')

        for k, v in  vars(self.args).items():
            config.set('default', k, str(v))

        with open(filePath, 'w') as f:
            config.write(f)
