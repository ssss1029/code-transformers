# Transformers for Binary Analysis


Currently, `train.py` only supports ELF binaries. 

This loads binaries from one dir
```
python3 train.py \
    --savedir=checkpoints/TEMP/ \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/1/binary/*
```

This loads binaries from several dirs
```
python3 train.py \
    --savedir=checkpoints/TEMP/ \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/*/binary/*
```

You can pass multiple `--dataroot`s to load binaries from even _more_ dirs
```
python3 train.py \
    --savedir=checkpoints/TEMP/ \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_64/*/binary/* \
    --dataroot=/var/tmp/sauravkadavath/binary/byteweight/elf_32/*/binary/*
```

