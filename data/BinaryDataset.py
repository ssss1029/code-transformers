"""

Dataset of binaries

"""

from data import pe_helper, elf_helper

import numpy as np

import bisect
import glob
import logging
import os
import random
import torch

class BinaryDataset(torch.utils.data.Dataset):
    """
    Dataset of compiled binaries.
    
    Returns dict: 
    {
        'X': array([ 
            76,   0,   0,   0,   1,  69,   4, 141,  78, 207, 137,   1, 224,
            ...
            224,  14, 232,   0,   0,  20,  35,  61, 141,  72, 195,  92
            ], dtype=uint8), 
       'y': array([
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ...
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        ])
    }

    Note: Going through this in mode='random-chunks' does NOT guarantee that you see each data 
        point once.
    """

    def __init__(self, root_dir, binary_format, targets='start', mode='random-chunks', chunk_length=1000, reverse=True):
        
        self.binary_format = binary_format
        self.mode          = mode
        self.chunk_length  = chunk_length
        self.reverse       = True

        if targets == 'both':
            self.output_function_starts = True
            self.output_function_ends = True
        elif targets == 'end':
            self.output_function_starts = False
            self.output_function_ends = True
        elif targets == 'start':
            self.output_function_starts = True
            self.output_function_ends = False
        else:
            raise NotImplementedError(f"Targets {targets} not recognized")
        
        self.binaries = load_binaries(
            binary_filenames=glob.glob(root_dir), 
            #binary_format=self.binary_format, 
            chunk_length=self.chunk_length, 
            reverse=self.reverse
        )

        self.num_functions = 0
        for _, boundaries, _ in self.binaries:
            self.num_functions += boundaries.shape[0]

    def __len__(self):
        """
        TODO: This len() is just the number of functions in the dataset. One epoch through the dataset would not
            ensure that one goes through all the functions/chunks. This depends on self.mode alot.
        """
        if self.mode == 'random-chunks':
            return self.num_functions
        else:
            raise NotImplementedError()

    def __getitem__(self, i):
        if self.mode == 'random-chunks':
            # Note this is independent of the arg i passed in
            # Pick a random binary
            random_binary = random.choice(self.binaries)
            text_length = len(random_binary[0])
            # Pick a random start location within the binary
            idx_start = random.randint(0, text_length - self.chunk_length - 1)
            return self.get_chunk(random_binary, idx_start, self.chunk_length)
        else:
            raise NotImplementedError()
    
    def get_chunk(self, binary, idx_start, length):
        text, function_boundaries, _ = binary
        function_starts, function_ends = function_boundaries[:, 0], function_boundaries[:, 1]

        X = np.fromstring(text[idx_start:idx_start + length], dtype=np.uint8)
        y = np.zeros(length)

        # TIL: https://docs.python.org/3/library/bisect.html
        # y consists of softmax indices I think (like used by nn.CrossEntropyLoss):
        # 0: Nothing
        # 1: Start only
        # 2: End only
        # 3: Start and End
        if self.output_function_starts:
            relevant_starts_left = bisect.bisect_left(function_starts, idx_start)
            relevant_starts_right = bisect.bisect_left(function_starts, idx_start + length)
            y[function_starts[relevant_starts_left:relevant_starts_right] - idx_start] += 1
        if self.output_function_ends:
            relevant_ends_left = bisect.bisect_left(function_ends, idx_start)
            relevant_ends_right = bisect.bisect_left(function_ends, idx_start + length)
            y[function_ends[relevant_ends_left:relevant_ends_right] - idx_start] += 1 + self.output_function_starts

        # Richard's code also returned this cw thing, idk why yet.
        # cw = np.ones(length)
        # return {'x': X, 'y': Y, 'cw': cw}

        return {'X': torch.tensor(X), 'y': torch.tensor(y)}

def load_binaries(binary_filenames, binary_format=None, chunk_length=1000, reverse=True):
    """
    Returns the binaries as a list: 
    List [(text:bytes, boundaries:np.array(num_functions, 2), filename:str)]
    """

    bin_helpers = { 'pe': pe_helper, 'elf': elf_helper }#[binary_format]
    if binary_format is not None:
         bin_helper = bin_helpers[binary_format]
            
    binaries = []
    for fn in binary_filenames:
        # Assumes "elf" or "pe" is in the path name
        # Maybe can be better optimized
        if binary_format is None:
            if "elf" in fn:
                bin_helper = bin_helpers["elf"]
            elif "pe" in fn:
                bin_helper = bin_helpers["pe"]
            else:
                logging.warning("unknown binary type")
        binary = bin_helper.open_binary(fn)
        if binary == None:
            logging.warning(f"Unable to open binary {fn}. Skipping this file.")
            continue
        
        # text: bytes, text_offset: int
        text, text_offset = bin_helper.get_text(binary)

        # List of tuples: [(f_name:str, start:int, length:int) ...]
        all_funcs = bin_helper.extract_functions_from_symbol_table(binary)
        function_boundaries = [
            (start - text_offset, start + size - text_offset - 1) for (fn, start, size) in all_funcs
        ]

        if reverse:
            text = text[::-1]
            function_boundaries = [(len(text) - x - 1, len(text) - y - 1) for (x, y) in function_boundaries]
        
        # Sort from smallest start value to largest
        function_boundaries.sort()

        binaries.append((text, np.array(function_boundaries, dtype=np.int32), os.path.basename(fn)))
    
    return binaries
    
