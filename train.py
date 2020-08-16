
"""

Transformers for Binary Analysis

"""

import numpy as np
import os
import argparse
import logging
import pprint
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data.BinaryDataset import BinaryDataset
from utils.evaluation import calc_f1
from models.rnn import RNN

from transformers import BertConfig, BertForTokenClassification, BertTokenizer

import sklearn

# TODO: Make this a command line arg
logging.basicConfig(level = logging.DEBUG)

parser = argparse.ArgumentParser(description='Code Transformer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataroot', type=str, action='append', help="Add multiple dataroots in glob format (e.g. '/var/tmp/sauravkadavath/binary/byteweight/elf_64/*/binary/*')")
parser.add_argument('--val-dataroot', type=str, action='append', help="Add multiple dataroots in glob format (e.g. '/var/tmp/sauravkadavath/binary/byteweight/elf_64/*/binary/*')")
parser.add_argument('--targets', type=str, choices=['start', 'end', 'both'], default='start')
parser.add_argument('--savedir', type=str)
parser.add_argument('--test', action='store_true', help="Add --test to only do validation and exit.")

# Model settings
parser.add_argument('--arch', choices=['gru', 'bert'], required=True)
parser.add_argument('--sequence-len', type=int, default=1024, help='Length of sequence fed into transformer')
parser.add_argument('--hidden-size', type=int, default=16)
parser.add_argument('--num-layers', type=int, default=2)
parser.add_argument('--num-attn-heads', type=int, default=8) # Only for BERT

# Optimizer settings
parser.add_argument('--optimizer', choices=['rmsprop', 'adam'], default='adam', type=str)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr-scheduler', type=str, choices=['none', 'cosine'], default='none')

# Loss settings
parser.add_argument('--weight-loss', '-wl', type=int, default=1, help='downweights background by 1/w. default is does nothing')
parser.add_argument('--weight-loss-rcf', action='store_true', help='Weights losses according to eq. 1 and 2 from https://arxiv.org/pdf/1612.02103.pdf')

args = parser.parse_args()

def check_args():
    """
    Sanity checks on arguments
    """

    if args.weight_loss != 1.0 and args.weight_loss_rcf == True:
        raise Exception("Either choose manual weight loss or RCF weight loss, not both")
    
    return True


def prologue():
    """
    Bookkeeping stuff
    """

    check_args()

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
            logging.info("Made save directory", args.savedir)

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
            targets=args.targets, 
            mode='random-chunks', 
            chunk_length=args.sequence_len
        )
        all_datasets.append(curr_dataset)

    train_data = torch.utils.data.ConcatDataset(all_datasets)
    logging.info("Train dataset len() = {0}".format(len(train_data)))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    val_datasets = []
    for dataroot in args.val_dataroot:
        curr_dataset = BinaryDataset(
            root_dir=dataroot,
            binary_format='elf',
            targets=args.targets, 
            mode='random-chunks', 
            chunk_length=args.sequence_len
        )
        val_datasets.append(curr_dataset)
    
    val_data = torch.utils.data.ConcatDataset(val_datasets)
    logging.info("Validation dataset len() = {0}".format(len(val_data)))
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=True, num_workers=2
    )


    ####################################################################
    ## Model
    ####################################################################

    if args.targets == 'start' or args.targets == 'end':
        num_classes = 2 
    elif args.targets == 'both':
        # TODO: Make sure if this really is 4 or if it is only 3 in practice
        num_classes = 4
    else:
        raise NotImplementedError()

    # Define model
    # For now, embedding dimension = hidden dimension

    if args.arch == 'gru':
        gru = torch.nn.GRU(
            input_size=args.hidden_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            bias=True,
            batch_first=True,
            bidirectional=True
        )

        embedder = torch.nn.Embedding(
            num_embeddings=256,
            embedding_dim=args.hidden_size
        )

        model = RNN(
            rnn=gru, 
            embedder=embedder,
            output_size=num_classes
        ).cuda()
    elif args.arch == 'bert':
        config = BertConfig(
            vocab_size=256, 
            hidden_size=args.hidden_size, 
            num_hidden_layers=args.num_layers, 
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
            gradient_checkpointing=False,
            num_labels=num_classes
        )

        model = BertForTokenClassification(config=config).cuda()
    else:
        raise NotImplementedError()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    else:
        raise NotImplementedError()

    if args.lr_scheduler == 'cosine':
        def cosine_annealing(step, total_steps, lr_max, lr_min):
                return lr_min + (lr_max - lr_min) * 0.5 * (
                        1 + np.cos(step / total_steps * np.pi))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                args.epochs * len(train_loader),
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / (args.lr * args.batch_size / 256.)
            )
        )
    elif args.lr_scheduler == 'none':
        scheduler = None
    else:
        raise NotImplementedError()

    with open(os.path.join(args.savedir, 'training_log.csv'), 'w') as f:
        f.write('epoch,train_loss,train_f1_average,val_loss,val_f1_average\n')
    
    logging.info("Beginning training")
    for epoch in range(args.epochs):
        train_loss_avg, train_f1_avg = train(
            model, optimizer, scheduler, train_loader, epoch, num_classes
        )

        val_loss_avg, val_f1_avg = validate(
            model, val_loader, num_classes
        )
        
        # torch.save(
        #     model.state_dict(),
        #     os.path.join(save_dir, "model.pth")
        # )

        # TODO: Save results and model

        with open(os.path.join(args.savedir, 'training_log.csv'), 'a') as f:
            f.write('%03d,%0.5f,%0.5f,%0.5f,%0.5f\n' % (
                (epoch + 1), train_loss_avg, train_f1_avg, val_loss_avg, val_f1_avg
            ))

def validate(model, test_loader, num_classes):
    losses = AverageMeter('Loss', ':.4e')
    f1 = AverageMeter('F1', ':.4e')

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            sequences = batch['X'].to(torch.int64).cuda() # batch_size x sequence_len
            labels    = batch['y'].to(torch.int64).cuda() # batch_size x sequence_len

            # Forward
            if args.arch == 'gru':
                logits = model(sequences)
            elif args.arch == 'bert':
                logits = model(sequences)[0]
            else:
                raise NotImplementedError()
            
            if args.weight_loss_rcf == True:
                num_background = torch.sum(labels == 0).item()
                num_foreground = torch.sum(labels != 0).item()
                weight_background = num_foreground / (num_foreground + num_background)
                weight_foreground = num_background / (num_foreground + num_background)
                
                weight = torch.ones(num_classes).cuda() * weight_foreground
                weight[0] = weight_background
            else:
                # Default to manual --weight-loss
                weight = torch.ones(num_classes).cuda()
                weight[0] = weight[0] / args.weight_loss

            # logging.info(weight)

            logits = logits.permute(0, 2, 1) # torch.Size([batch_size, N, sequence_len]); N = softmax dim
            loss = torch.nn.functional.cross_entropy(logits, labels, weight)
            
            # Bookkeeping
            losses.update(loss.item(), sequences.size(0))
            
            # TODO: Maybe keep track of a moving average of F1 during training?
            if args.targets == 'start' or args.targets == 'end':
                f1_curr = calc_f1(logits.detach().cpu(), labels.detach().cpu())
                f1.update(f1_curr, sequences.size(0))
            else:
                # TODO: Implement F1 for 'both' targets
                raise NotImplementedError()
        
    return losses.avg, f1.avg




def train(model, optimizer, scheduler, train_loader, epoch, num_classes):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    f1 = AverageMeter('F1', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, f1],
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    end = time.time()
    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        sequences = batch['X'].to(torch.int64).cuda() # batch_size x sequence_len
        labels    = batch['y'].to(torch.int64).cuda() # batch_size x sequence_len

        # Forward
        optimizer.zero_grad()
        if args.arch == 'gru':
            logits = model(sequences)
        elif args.arch == 'bert':
            logits = model(sequences)[0]
        else:
            raise NotImplementedError()
        
        if args.weight_loss_rcf == True:
            num_background = torch.sum(labels == 0).item()
            num_foreground = torch.sum(labels != 0).item()
            weight_background = num_foreground / (num_foreground + num_background)
            weight_foreground = num_background / (num_foreground + num_background)
            
            weight = torch.ones(num_classes).cuda() * weight_foreground
            weight[0] = weight_background
        else:
            # Default to manual --weight-loss
            weight = torch.ones(num_classes).cuda()
            weight[0] = weight[0] / args.weight_loss

        # logging.info(weight)

        logits = logits.permute(0, 2, 1) # torch.Size([batch_size, N, sequence_len]); N = softmax dim
        loss = torch.nn.functional.cross_entropy(logits, labels, weight)

        # Backward
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()

        # Bookkeeping
        losses.update(loss.item(), sequences.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # TODO: Maybe keep track of a moving average of F1 during training?
        if args.targets == 'start' or args.targets == 'end':
            f1_curr = calc_f1(logits.detach().cpu(), labels.detach().cpu())
            f1.update(f1_curr, sequences.size(0))
        else:
            # TODO: Implement F1 for 'both' targets
            raise NotImplementedError()

        if i % args.print_freq == 0:
            # print(logits.shape) # torch.Size([batch_size, N_classes, seq_len])
            # print(logits[:2, :, :7])
            progress.display(i)
        
    return losses.avg, f1.avg



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
