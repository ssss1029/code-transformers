

########################################
# Train GRU
########################################

# srun --pty -p gpu_jsteinhardt -w shadowfax -c 10 --gres=gpu:1 python3 train.py \
#     --target=start \
#     --arch=gru \
#     --weight-loss-rcf \
#     --batch-size=256 \
#     --sequence-len=2048 \
#     --savedir=checkpoints/TEMP_2 \
#     --optimizer=adam \
#     --lr=1e-3 \
#     --lr-scheduler=none \
#     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/1/binary/* \
#     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/2/binary/* \
#     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/3/binary/* \
#     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/4/binary/* \
#     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/5/binary/* \
#     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/6/binary/* \
#     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/7/binary/* \
#     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/8/binary/* \
#     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/9/binary/* \
#     --val-dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/10/binary/* \


########################################
# Train BERT
########################################

srun --pty -p gpu_jsteinhardt -w shadowfax -c 10 --gres=gpu:4 python3 train.py \
    --target=start \
    --arch=bert \
    --sequence-len=1024 \
    --hidden-size=256 \
    --num-layers=4 \
    --num-attn-heads=16 \
    --optimizer=adam \
    --lr=3e-4 \
    --lr-scheduler=none \
    --batch-size=8 \
    --grad-acc-steps=10 \
    --weight-loss-rcf \
    --savedir=checkpoints/TEMP_2 \
    --val-dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/10/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/1/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/2/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/3/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/4/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/5/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/6/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/7/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/8/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/9/binary/* \
    # --load-pretrained=checkpoints/pretrain_bert_elf64_all/weights/ \



