

########################################
# Train GRU
########################################

CUDA_VISIBLE_DEVICES=7 python3 train.py \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/1/binary/* \
    --target=start \
    --savedir=checkpoints/TEMP \
    --arch=gru \
    --weight-loss-rcf \
    --batch-size=128 \
    --sequence-len=2048 \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/2/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/3/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/4/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/5/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/6/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/7/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/8/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/9/binary/* \
