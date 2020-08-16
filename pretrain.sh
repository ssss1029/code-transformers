
########################################
# Train BERT
########################################

CUDA_VISIBLE_DEVICES=6,7 python3 pretrain.py \
    --arch=bert \
    --savedir=checkpoints/TEMP \
    --sequence-len=1024 \
    --batch-size=32 \
    --epochs=2 \
    --lr=3e-3 \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/1/binary/* \
    # --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/2/binary/* \
    # --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/3/binary/* \
    # --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/4/binary/* \
    # --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/5/binary/* \
    # --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/6/binary/* \
    # --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/7/binary/* \
    # --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/8/binary/* \
    # --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/9/binary/* \

