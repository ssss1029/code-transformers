
"""

Binary dataloading. Example usage: 

This loads binaries from one dir
python3 train.py \
    --savedir=checkpoints/TEMP/ \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/1/binary/*

This loads binaries from several dirs
python3 train.py \
    --savedir=checkpoints/TEMP/ \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/*/binary/*

This loads binaries from even _more_ dirs
python3 train.py \
    --savedir=checkpoints/TEMP/ \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/*/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_32/*/binary/*
"""

import numpy as np
import os
import argparse
import pprint
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data.BinaryDataset import BinaryDataset

parser = argparse.ArgumentParser(description='Code Transformer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataroot', type=str, action='append')
parser.add_argument('--savedir', type=str)

args = parser.parse_args()

def prologue():
    """
    Bookkeeping stuff
    """
    if os.path.exists(args.savedir):
        resp = "None"
        while resp.lower() not in {'y', 'n'}:
            resp = input("Save directory {0} exits. Continue? [Y/n]: ".format(args.savedir))
            if resp.lower() == 'y':
                break
            elif resp.lower() == 'n':
                exit(1)
            else:
                pass
    else:
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)

        if not os.path.isdir(args.savedir):
            raise Exception('%s is not a dir' % args.savedir)
        else:
            print("Made save directory", args.savedir)

    with open(os.path.join(args.savedir, 'command.txt'), 'w') as f:
        to_print = vars(args)
        to_print['FILENAME'] = __file__
        pprint.pprint(to_print, stream=f)

def main():
    
    all_datasets = []
    for dataroot in args.dataroot:
        curr_dataset = BinaryDataset(
            root_dir=dataroot,
            binary_format='elf'
        )
        all_datasets.append(curr_dataset)

    print(all_datasets[0][0])


if __name__ == "__main__":
    # prologue()
    main()