
########################################
# Train BERT
########################################

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 pretrain.py \
    --arch=bert \
    --savedir=checkpoints/TEMP \
    --sequence-len=1024 \
    --batch-size=24 \
    --epochs=20 \
    --lr=1e-2 \
    --multistep-milestone=2 \
    --multistep-milestone=5 \
    --multistep-milestone=10 \
    --multistep-milestone=15 \
    --multistep-gamma=0.2 \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/1/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/2/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/3/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/4/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/5/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/6/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/7/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/8/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/9/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/10/binary/* \

