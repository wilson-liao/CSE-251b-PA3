import torch
import torch.nn as nn

class LSTMNoTeacherForcingModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, seq_len):
        super(LSTMNoTeacherForcingModel, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.seq_len = seq_len  # Store sequence length for generation

    def forward(self, input_token, hidden):
        batch_size = input_token.shape[0]
        outputs = []

        for _ in range(self.seq_len):  
            embedded = self.embedding(input_token)  
            lstm_out, hidden = self.lstm(embedded, hidden)  
            output = self.fc(lstm_out)  
            outputs.append(output)

            _, input_token = torch.max(output, dim=2)  # Greedy decoding
            input_token = input_token.detach()  

        outputs = torch.cat(outputs, dim=1)  
        return outputs, hidden

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device  # Get model's device (CUDA or CPU)
        return (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device),
                torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device))