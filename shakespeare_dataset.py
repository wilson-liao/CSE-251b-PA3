import torch
from torch.utils.data import Dataset

class ShakespeareDataset(Dataset):
    def __init__(self, X, y, device):
        self.X = X.to(device)
        self.y = y.to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # i think??? lmao
        return self.X[idx], self.y[idx]