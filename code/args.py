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
    if len(x.split(" ")) == 1:
        return float(x)
    else:
        return [float(elem) for elem in x.split(" ")]

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
        self.parser.add_argument('--lr', type=str2FloatList, nargs='+',metavar='LR',
                            help='learning rate (it can be a schedule : --lr 0.01 0.001 0.0001)')
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
        self.parser.add_argument('--dataset', type=str, metavar='N',help='the dataset to use. Can be \'NETFLIX\' or \'IRCCYN\'')

        self.parser.add_argument('--annotNb', type=int, metavar='S',
                            help='The number of annotator in the dataset')
        self.parser.add_argument('--erase_results', type=str2bool, metavar='S',
                            help='To erase the convergence speed results already computed')
        self.parser.add_argument('--cuda', type=str2bool, metavar='S',
                            help='To run computations on the gpu')
        self.parser.add_argument('--optim', type=str, metavar='OPTIM',
                            help='the optimizer algorithm to use (default: \'LBFGS\')')
        self.parser.add_argument('--param_name', type=str, metavar='S',
                            help='The name of the parameter to vary during \
                                 robustness evaluation. Can be \'nb_annot\' or \'nb_corr\'')

        self.parser.add_argument('--param_min', type=int, metavar='S',
                            help='The minimum value of the parameter to vary')
        self.parser.add_argument('--param_max', type=int, metavar='S',
                            help='The maximum value of the parameter to vary')

        self.parser.add_argument('--nb_rep', type=int, metavar='S',
                            help='The number of repetition for each parameter value')

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
