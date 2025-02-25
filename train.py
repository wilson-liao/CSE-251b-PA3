from util import *
from train import *
import numpy as np
import torch.optim as optim
import torch.nn as nn
import os
import time
import random
import torch.nn.functional as F

from tqdm import tqdm

from shakespeare_lstm import LSTMModel
from shakespeare_rnn import RNNModel

def train(model, device, train_dataloader, val_dataloader, config):
    #model is allegedly already on device
    #counting
    best_loss = float('inf')
    best_epoch = None
    teacher_forcing = True #stored here and not passed in for the ~best~ coding practice
    no_improve_count = 0
    
    # func constants
    epochs = config["epochs"]
    patience = config["patience"]
    learning_rate = config['lr']
    num_layers = config['num_layers']
    hidden_size = config['hidden_size']
    seq_len = config['seq_len']
    
    criterion =  torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    #to be plotted
    train_loss_all = []
    val_loss_all = []
    early_stop_epoch = None

    #training loop
    for epoch in range(epochs):
        print('IN EPOCH ', epoch)
        ts = time.time()
        train_losses = []
        for iter, (inputs, labels) in enumerate(train_dataloader):

            optimizer.zero_grad()

            inputs =  inputs.to(device)        # TODO transfer the input to the same device as the model's
            labels =  labels.to(device)         # TODO transfer the labels to the same device as the model's

            # outputs = model.forward(inputs) # Shape should be [batch_size, seq_len, vocab_size]

            ######### NO TEACHER
            hidden = tuple(h.to(device) for h in model.init_hidden(inputs.shape[0]))  # Move hidden state to CUDA
            inputs = inputs.to(device)  # Ensure input is also on CUDA
            labels = labels.to(device)  # Ensure labels are on CUDA
            outputs, _ = model.forward(inputs, hidden)
            ######### 

            loss = criterion(outputs[:, -1, :], labels)

            train_losses.append(loss.item())
            
            loss.backward()

            optimizer.step()

            
        
        print(f"Finish epoch {epoch}, train loss {np.mean(train_losses)}, time elapsed {time.time() - ts}")
        
        curr_loss = eval(model, device, val_dataloader, epoch, config)
        train_loss_all.append(np.mean(train_losses))
        val_loss_all.append(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model_weights.pth') #saving this so we don't have to retrain all the time
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"No improvement in loss for {no_improve_count}/{patience} epochs.")
        
        # Early stopping condition

        if no_improve_count >= patience and early_stop_epoch is None:
            early_stop_epoch = epoch
            print(f"First early stopping condition met at epoch {early_stop_epoch}. Training continues.")
            # break

    fname = model.__class__.__name__ + '_losses' + '.png'
    plot_losses(train_loss_all, val_loss_all, early_stop_epoch, best_epoch, fname)
            
    

def eval(model, device, val_dataloader, epoch, config):
    model.eval()
    losses = []
    criterion = torch.nn.CrossEntropyLoss()
    num_layers = config['num_layers']
    hidden_size = config['hidden_size']
    seq_len = config['seq_len']

    with torch.no_grad():
        for iter, (inputs, labels) in enumerate(val_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # outputs = model(inputs)
            ######### NO TEACHER
            hidden = tuple(h.to(device) for h in model.init_hidden(inputs.shape[0]))  # Move hidden state to CUDA
            inputs = inputs.to(device)  # Ensure input is also on CUDA
            labels = labels.to(device)  # Ensure labels are on CUDA
            outputs, _ = model.forward(inputs, hidden)
            ######### 
            
            # outputs = outputs.view(-1, outputs.size(-1))  # Reshape to [batch_size*seq_len, vocab_size]
            
            loss = criterion(outputs[:, -1, :], labels)
            losses.append(loss.item())
    if epoch < config['epochs']:
        print(f"\nValidation Loss at epoch: {epoch} is {np.mean(losses)}")
    else:
        print(f"\nTest Loss: {epoch} is {np.mean(losses)}")
    model.train()
    return np.mean(losses)