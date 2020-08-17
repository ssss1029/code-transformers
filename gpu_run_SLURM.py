# -*- coding: utf-8 -*-

"""
Given a bunch of commands to run, check the available GPUs and run them on the GPUs in separate tmux sessions.
Usage: Just modify the settings in the Config class and then run python3 gpu_run.py
"""

import GPUtil
import subprocess
import sys
import time

class Config:
    """
    Global class that houses all configurations
    """
    
    # Shared args to put onto all of the JOBS
    SHARED_ARGS = " "

    SLURM_HEADER = "srun --pty -p gpu_jsteinhardt -w shadowfax -c 15 --gres=gpu:4"

    # Specifies tasks to run. It maps tmux session name to the command to run in that session.
    JOBS = {
        # "pretrain_bert_elf64_all" : " python3 pretrain.py \
        #     --arch=bert \
        #     --savedir=checkpoints/pretrain_bert_elf64_all \
        #     --sequence-len=1024 \
        #     --batch-size=24 \
        #     --epochs=30 \
        #     --lr=1e-3 \
        #     --multistep-milestone=2 \
        #     --multistep-milestone=5 \
        #     --multistep-milestone=10 \
        #     --multistep-milestone=15 \
        #     --multistep-milestone=20 \
        #     --multistep-milestone=25 \
        #     --multistep-gamma=0.2 \
        #     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/1/binary/* \
        #     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/2/binary/* \
        #     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/3/binary/* \
        #     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/4/binary/* \
        #     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/5/binary/* \
        #     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/6/binary/* \
        #     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/7/binary/* \
        #     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/8/binary/* \
        #     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/9/binary/* \
        #     --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/10/binary/*"

        "tune_bert_elf64_lr3e-5" : "python3 train.py \
            --target=start \
            --arch=bert \
            --weight-loss-rcf \
            --batch-size=32 \
            --sequence-len=1024 \
            --savedir=checkpoints/tune_bert_elf64_lr3e-5 \
            --optimizer=adam \
            --lr=3e-5 \
            --epochs=100 \
            --lr-scheduler=none \
            --grad-acc-steps=10 \
            --val-dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/10/binary/* \
            --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/1/binary/* \
            --load-pretrained=checkpoints/pretrain_bert_elf64_all/weights/ \
            --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/2/binary/* \
            --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/3/binary/* \
            --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/4/binary/* \
            --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/5/binary/* \
            --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/6/binary/* \
            --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/7/binary/* \
            --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/8/binary/* \
            --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/9/binary/*",
        
        "tune_bert_elf64_lr1e-4" : "python3 train.py \
            --master-port=12346 \
            --target=start \
            --arch=bert \
            --weight-loss-rcf \
            --batch-size=32 \
            --sequence-len=1024 \
            --savedir=checkpoints/tune_bert_elf64_lr1e-4 \
            --optimizer=adam \
            --lr=1e-4 \
            --epochs=100 \
            --lr-scheduler=none \
            --grad-acc-steps=10 \
            --val-dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/10/binary/* \
            --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/1/binary/* \
            --load-pretrained=checkpoints/pretrain_bert_elf64_all/weights/ \
            --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/2/binary/* \
            --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/3/binary/* \
            --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/4/binary/* \
            --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/5/binary/* \
            --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/6/binary/* \
            --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/7/binary/* \
            --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/8/binary/* \
            --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/9/binary/*"
    }

    # Time to wait between putting jobs on GPUs (in seconds). This is useful because it might take time 
    # for a process to actually load the network onto the GPU, so we wait until that is done before 
    # selecting the GPU for the next process.
    SLEEP_TIME = 1


# Stick the shared args onto each JOB
for key, value in Config.JOBS.items():
    new_value = value + " " + Config.SHARED_ARGS
    Config.JOBS[key] = new_value


for index, (tmux_session_name, command) in enumerate(Config.JOBS.items()):
    # Get the best available GPU
    print("Running \"{0}\" with SLURM".format(command))

    result = subprocess.run("tmux new-session -d -s {0}".format(tmux_session_name), shell=True)        
    if result.returncode != 0:
        print("Failed to create new tmux session called {0}".format(tmux_session_name))
        sys.exit(result.returncode)

    result = subprocess.run("tmux send-keys '{0} {1}' C-m".format(
        Config.SLURM_HEADER, command
    ), shell=True)
    if result.returncode != 0:
        print("Failed to run {0} in tmux session".format(command, tmux_session_name))
        sys.exit(result.returncode)

    print("---------------------------------------------------------------")

    if index < len(Config.JOBS) - 1:
        time.sleep(Config.SLEEP_TIME)
