
"""

Transformer for Function ID tasks

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

from transformers import BertConfig, BertForTokenClassification, BertTokenizer

parser = argparse.ArgumentParser(description='Code Transformer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataroot', type=str, action='append', help="Add multiple dataroots in glob format (e.g. '/var/tmp/sauravkadavath/binary/byteweight/elf_64/*/binary/*')")
parser.add_argument('--savedir', type=str)

# Model settings
parser.add_argument('--sequence-len', type=int, default=1024, help='Length of sequence fed into transformer')
parser.add_argument('--hidden-size', type=int, default=256)
parser.add_argument('--hidden-layers', type=int, default=8)
parser.add_argument('--num-attn-heads', type=int, default=8)

# Opt settings
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=10)

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
            chunk_length=args.sequence_len
        )
        all_datasets.append(curr_dataset)

    # TODO: ConcatDataset. This requires the __len__() to be implemented.
    dataset = torch.utils.data.ConcatDataset(all_datasets)
    print("Dataset len() = {0}".format(len(dataset)))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )


    ####################################################################
    ## Model
    ####################################################################

    config = BertConfig(
        vocab_size=256, 
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.hidden_layers, 
        num_attention_heads=args.num_attn_heads, 
        intermediate_size=args.hidden_size * 4, # BERT originally uses 4x hidden size for this, so copying that. 
        hidden_act='gelu', 
        hidden_dropout_prob=0.1, 
        attention_probs_dropout_prob=0.1, 
        max_position_embeddings=args.sequence_len, # Sequence length max 
        type_vocab_size=1, 
        initializer_range=0.02, 
        layer_norm_eps=1e-12, 
        pad_token_id=0, 
        gradient_checkpointing=False
    )

    model = BertForTokenClassification(config=config).cuda()
    # model = torch.nn.DataParallel(model, dim=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    lossfn = torch.nn.CrossEntropyLoss()

    print("Beginning training")
    for epoch in range(args.epochs):
        train_loss, train_acc = train(
            model, lossfn, optimizer, dataloader, epoch
        )

        print(f"Train Loss: {train_loss} | Test Loss: {test_loss} | Test Acc: {test_acc}")
        
        # torch.save(
        #     model.state_dict(),
        #     os.path.join(save_dir, "model.pth")
        # )



def train(model, lossfn, optimizer, dataloader, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    end = time.time()
    for i, batch in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        sequences = batch['X'].to(torch.int64).cuda() # batch_size x sequence_len
        labels    = batch['y'].to(torch.int64).cuda() # batch_size x sequence_len

        # Forward
        optimizer.zero_grad()
        logits = model(sequences)[0]
        logits = logits.permute(0, 2, 1)
        loss = lossfn(logits, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Bookkeeping
        losses.update(loss.item(), sequences.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            progress.display(i)
        
    return losses.avg



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    prologue()
    main()