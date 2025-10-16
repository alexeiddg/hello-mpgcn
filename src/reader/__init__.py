import logging

from .nba_reader import NBA_Reader
from .k400_reader import K400_HRNet_Reader

__generator = {
    'nba': NBA_Reader,
    'kinetics': K400_HRNet_Reader
}


def create(args):
    dataset = args.dataset
    dataset_args = args.dataset_args
    if dataset not in __generator.keys():
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError()
    return __generator[dataset](**dataset_args)
