import torch
import torch.nn as nn

# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_size, num_layers):
        super(RNNModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # RNN layer
        self.rnn = nn.RNN(input_size=embed_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers, 
                          batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out
        # batch_size, seq_len, hidden_dim = out.shape
        # out = self.fc(out.contiguous().view(-1, hidden_dim)) 
        
        # return out, hidden