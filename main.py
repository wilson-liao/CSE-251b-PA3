import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import random_split

from util import encode_text, create_sequences
from shakespeare_dataset import ShakespeareDataset
from shakespeare_lstm import LSTMModel
from shakespeare_rnn import RNNModel
from config import load_config
from train import train, eval

from shakespeare_lstm_no_teacher import LSTMNoTeacherForcingModel

import multiprocessing

def main():

    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     'config_file_path',
    #     type=str
    # )

    # args = parser.parse_args()

    ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # num_workers = multiprocessing.cpu_count()
    num_workers = 0
    ###

    input_file_path = 'data/tiny_shakespeare.txt'

    print('ENCODED TEXT DATA')

    encoded_text, vocab_size, char_to_idx, idx_to_char = encode_text(input_file_path)

    config = load_config('configs/base_lstm_config.yaml')

    seq_length = config['seq_len']
    X, y = create_sequences(encoded_text, seq_length)

    print('CREATED SEQUENCES, with VOCAB SIZE ', vocab_size)

    ###
    # Convert to PyTorch tensors and on gpu
    X_tensor = torch.tensor(X, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    ###

    len_data = len(y_tensor)

    train_frac, val_frac, test_frac = 0.8, 0.1, 0.1  
    train_size = int(train_frac * len_data)
    val_size = int(val_frac * len_data)
    test_size = len_data - train_size - val_size  

    torch.manual_seed(0)
    indices = torch.randperm(len_data)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    assert set(train_indices).isdisjoint(set(val_indices)) and set(train_indices).isdisjoint(set(test_indices)) and set(val_indices).isdisjoint(set(test_indices))
    print('PERFORMED TRAIN/VAL/TEST SPLIT')

    # Index tensors to get non-overlapping splits
    X_train, y_train = X_tensor[train_indices], y_tensor[train_indices]
    X_val, y_val = X_tensor[val_indices], y_tensor[val_indices]
    X_test, y_test = X_tensor[test_indices], y_tensor[test_indices]

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")


    train_dataset = ShakespeareDataset(X_train, y_train, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers)

    val_dataset = ShakespeareDataset(X_val, y_val, device=device)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    test_dataset = ShakespeareDataset(X_test, y_test, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers)


    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # RNN  vocab_size, hidden_size, embed_size, num_layers
    # model = RNNModel(vocab_size=vocab_size, embed_size=config['embed_size'],
    #                             hidden_size=config['hidden_size'],
    #                             num_layers=config['num_layers']).to(device)
    # LSTM
    # model = LSTMModel(vocab_size, config['embed_size'],
    #                             config['hidden_size'],
    #                             config['num_layers']).to(device)
    # LSTM no teacher
    model = LSTMNoTeacherForcingModel(vocab_size, 
                                      config['embed_size'],
                                      config['hidden_size'],
                                      config['num_layers'],
                                      config['seq_len']  # Add this argument
                                     ).to(device)

    # config = load_config(args.config_file_path) #why was this here??? why are we loading config again??

    ## TRAIN - will save best model weights
    train(model=model,
            device=device,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config)

    ## INFERENCE
    eval(model=model, device=device,
    val_dataloader=test_dataloader, epoch=config['epochs'], config=config)


if __name__ == '__main__':
    main()