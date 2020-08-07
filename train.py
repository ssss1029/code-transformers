
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

from transformers import BertConfig, BertForTokenClassification

parser = argparse.ArgumentParser(description='Code Transformer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataroot', type=str, action='append', help="Add multiple dataroots in glob format (e.g. '/var/tmp/sauravkadavath/binary/byteweight/elf_64/*/binary/*')")
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
    
    ####################################################################
    ## Data
    ####################################################################
    
    all_datasets = []
    for dataroot in args.dataroot:
        curr_dataset = BinaryDataset(
            root_dir=dataroot,
            binary_format='elf',
            targets='start', 
            mode='random-chunks', 
            chunk_length=1000
        )
        all_datasets.append(curr_dataset)

    # TODO: ConcatDataset. This requires the __len__() to be implemented.
    dataset = torch.utils.data.ConcatDataset(all_datasets)
    print("Dataset len() = {0}".format(len(dataset)))
    print(dataset[0]['X'].shape, dataset[0]['y'].shape)

    ####################################################################
    ## Model
    ####################################################################

    config = BertConfig(
        vocab_size=256, 
        hidden_size=768, 
        num_hidden_layers=12, 
        num_attention_heads=12, 
        intermediate_size=3072, 
        hidden_act='gelu', 
        hidden_dropout_prob=0.1, 
        attention_probs_dropout_prob=0.1, 
        max_position_embeddings=512, 
        type_vocab_size=1, 
        initializer_range=0.02, 
        layer_norm_eps=1e-12, 
        pad_token_id=0, 
        gradient_checkpointing=False
    )

    model = BertForTokenClassification(config=config)

    



if __name__ == "__main__":
    # prologue()
    main()