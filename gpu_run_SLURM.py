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
    SHARED_ARGS = " \
        --target=start \
        --arch=gru \
        --weight-loss-rcf \
        --batch-size=256 \
        --sequence-len=2048 \
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
    "

    SLURM_HEADER = "srun --pty -p gpu_jsteinhardt -w shadowfax -c 4 --gres=gpu:1"

    # Specifies tasks to run. It maps tmux session name to the command to run in that session.
    JOBS = {
        "gru_elf64_rmsprop_lr1e-4_noSchedule" : "python3 train.py \
            --savedir=checkpoints/gru_elf64_rmsprop_lr1e-4_noSchedule \
            --optimizer=rmsprop \
            --lr=1e-4 \
            --lr-scheduler=none \
            ",

        "gru_elf64_rmsprop_lr3e-3_noSchedule" : "python3 train.py \
            --savedir=checkpoints/gru_elf64_rmsprop_lr3e-3_noSchedule \
            --optimizer=rmsprop \
            --lr=3e-3 \
            --lr-scheduler=none \
            ",

        "gru_elf64_rmsprop_lr1e-3_noSchedule" : "python3 train.py \
            --savedir=checkpoints/gru_elf64_rmsprop_lr1e-3_noSchedule \
            --optimizer=rmsprop \
            --lr=1e-3 \
            --lr-scheduler=none \
            ",

        "gru_elf64_adam_lr3e-5_noSchedule" : "python3 train.py \
            --savedir=checkpoints/gru_elf64_adam_lr3e-5_noSchedule \
            --optimizer=adam \
            --lr=3e-5 \
            --lr-scheduler=none \
            ",

        "gru_elf64_adam_lr1e-5_noSchedule" : "python3 train.py \
            --savedir=checkpoints/gru_elf64_adam_lr1e-5_noSchedule \
            --optimizer=adam \
            --lr=1e-5 \
            --lr-scheduler=none \
            ",
        
        "gru_elf64_adam_lr1e-6_noSchedule" : "python3 train.py \
            --savedir=checkpoints/gru_elf64_adam_lr1e-6_noSchedule \
            --optimizer=adam \
            --lr=1e-6 \
            --lr-scheduler=none \
            ",
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
