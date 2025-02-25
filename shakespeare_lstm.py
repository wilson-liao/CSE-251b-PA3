import torch
import torch.nn as nn

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
                            
        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Linear(hidden_size, vocab_size)


    def forward(self, x, hidden=None):
        x = self.embedding(x)
        
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out