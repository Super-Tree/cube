"""
PYTHONPATH=/opt/ros/indigo/lib/python2.7/dist-packages;
PYTHONUNBUFFERED=1;
LD_LIBRARY_PATH=/usr/local/cuda/lib64
CUDA_VISIBLE_DEVICES = 0
"""
import _init_paths
import argparse
import sys

from dataset.dataset import get_data
from network.train_net import train_net
from network.test_net import test_net

from cubicnet.cubic_train import network_training
from cubicnet.cubic_test import network_testing


def parse_args():
    parser = argparse.ArgumentParser(description='Train a CombineNet network')
    parser.add_argument('--gpu_id', dest='gpu_id',help=' which gpu to use',
                        default=[0,1,2,3], type=list)
    parser.add_argument('--method', dest='method',help=' train or test',choices=['train', 'test'],
                        default="train", type=str)
    parser.add_argument('--weights', dest='weights',help='which network weights',
                        default='/home/hexindong/ws_dl/pyProj/cubic-local/MODEL_weights/CUBIC_2/weights/CubicNet_iter_85000.ckpt', type=str)
    parser.add_argument('--epoch_iters', dest='epoch_iters',help='number of iterations to train',
                        default=18, type=int)
    parser.add_argument('--imdb_type', dest='imdb_type',help='dataset to train on(sti/kitti)', choices=['kitti', 'sti'],
                        default='kitti', type=str)

    parser.add_argument('--useDemo', dest='useDemo',help='whether use continues frame demo',
                        default="False", type=str)
    parser.add_argument('--fineTune', dest='fineTune',help='whether finetune the existing network weight',
                        default='True', type=str)

    parser.add_argument('--use_demo', dest='use_demo', default=False, type=bool)
    parser.add_argument('--fine_tune', dest='fine_tune', default=True, type=bool)

    # if len(sys.argv) == 1:
    #   parser.print_help()
    #    sys.exit(1)
    return parser.parse_args()


def checkArgs(Args):
    print('Called with args:')
    print(args)
    # print('Using config:')
    # pprint.pprint(cfg)
    print "Checking the args ..."

    if Args.fineTune == 'True':
        Args.fine_tune = True
    else:
        Args.fine_tune = False

    if Args.useDemo == 'True':
        Args.use_demo = True
    else:
        Args.use_demo = False

    if Args.method == 'test':
        if Args.weights is None:
            print "  Specify the testing network weights!"
            sys.exit(3)
        else:
            print "  Test the weight: \n {}".format(Args.weights)
    elif Args.fine_tune:
            if Args.weights is None:
                print "  Specify the finetune network weights!"
                sys.exit(4)
            else:
                print "  Finetune the weight: \n     {}".format(Args.weights)
    else:
            print "  The network will RETRAIN from empty ! ! "


def get_network(arguments):
    """Get a network by name."""
    if arguments.method == 'train':
        return train_net(arguments.gpu_id)
    else:
        return test_net(arguments.gpu_id,trainable=False)
        # print "Loading model from .meta ...."
        # return None # hxd: when testting,we needn't to reload the model,using .meta file to restore the graph


if __name__ == '__main__':
    args = parse_args()
    checkArgs(args)

    data_set = get_data(args)  # load  dataset

    network = get_network(args)  # load network model

    if args.method == 'train':
        network_training(network, data_set, args)
    elif args.method == 'test':
        network_testing(network, data_set, args)
