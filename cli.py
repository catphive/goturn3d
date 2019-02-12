import argparse


def parse_sets(data_str):

    idx_list = data_str.split(':')

    if len(idx_list) == 1:
        return int(idx_list[0])
    elif len(idx_list) == 2:
        return (int(idx_list[0]), int(idx_list[1]))
    else:
        raise RuntimeError(f'Could not parse -sets arg "{data_str}"')

def get_args():

    # based on pygoturn args
    # TODO: remove unneeded args.

    parser = argparse.ArgumentParser(description='my goturn impl')

    parser.add_argument('-data', '--data-directory', type=str,
                        # default='../data/',
                        required=True,
                        help='path to data directory')


    parser.add_argument('-sets', '--data-sets', type=parse_sets,
                        required=True,
                        help='either idx or begin:end')

    parser.add_argument('-epochs', '--num-epochs', type=int, default=10,
                        help='number of epochs to run')

    # parser.add_argument('-n', '--num-batches', default=500000, type=int,
    #                     help='number of total batches to run')
    # parser.add_argument('-lr', '--learning-rate', default=1e-5, type=float,
    #                     help='initial learning rate')
    # parser.add_argument('--gamma', default=0.1, type=float,
    #                     help='learning rate decay factor')
    # parser.add_argument('--momentum', default=0.9, type=float,
    #                     help='optimizer momentum')
    # parser.add_argument('--weight_decay', default=0.0005, type=float,
    #                     help='weight decay in optimizer')
    # parser.add_argument('--lr-decay-step', default=100000, type=int,
    #                     help='number of steps after which learning rate decays')
    #
    # parser.add_argument('-save', '--save-directory', type=str,
    #                     default='../saved_checkpoints/exp3/',
    #                     help='path to save directory')
    # parser.add_argument('-lshift', '--lambda-shift-frac', default=5, type=float,
    #                     help='lambda-shift for random cropping')
    # parser.add_argument('-lscale', '--lambda-scale-frac', default=15, type=float,
    #                     help='lambda-scale for random cropping')
    # parser.add_argument('-minsc', '--min-scale', default=-0.4, type=float,
    #                     help='min-scale for random cropping')
    # parser.add_argument('-maxsc', '--max-scale', default=0.4, type=float,
    #                     help='max-scale for random cropping')
    # parser.add_argument('-seed', '--manual-seed', default=800, type=int,
    #                     help='set manual seed value')
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none)')
    # parser.add_argument('-b', '--batch-size', default=50, type=int,
    #                     help='number of samples in batch (default: 50)')
    # parser.add_argument('--save-freq', default=20000, type=int,
    #                     help='save checkpoint frequency (default: 20000)')


    args = parser.parse_args()
    return args

