
import torch
import torch.nn as nn 
import torch.nn.functional as F

class RNN(nn.Module):

    def __init__(self, rnn, embedder, output_size):
        super(RNN, self).__init__()
        self.rnn = rnn
        self.embedder = embedder
        self.output_size = output_size

        # Project hidden states onto sofmax dimension
        # 2 * hidden_size since bidirectional RNNs concat each direction for the final output
        self.proj = nn.Linear(2 * self.rnn.hidden_size, self.output_size)
    
    def forward(self, x):
        x = self.embedder(x)
        x, _ = self.rnn(x)
        return self.proj(x)
