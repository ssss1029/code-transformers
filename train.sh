

########################################
# Train GRU
########################################

srun --pty -p gpu_jsteinhardt -w shadowfax -c 10 --gres=gpu:1 python3 train.py \
    --target=start \
    --arch=gru \
    --weight-loss-rcf \
    --batch-size=256 \
    --sequence-len=2048 \
    --savedir=checkpoints/TEMP_2 \
    --optimizer=adam \
    --lr=1e-3 \
    --lr-scheduler=none \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/1/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/2/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/3/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/4/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/5/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/6/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/7/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/8/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/9/binary/* \
    --val-dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/10/binary/* \
