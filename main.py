import random

import numpy as np
import matplotlib as plt
import torch
import torch.nn as nn
import time
import sys
import torch.nn.functional as F # Contains some additional functions such as activations
from torch.autograd import Variable
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
from colorama import Fore, Back, Style
from torchmetrics import MeanSquaredLogError
from torchsummary import summary

from NumpyDataset import *
from Autoencoder import *
from train_val_pred import *
from auxiliary import *


# DEFINE CONSTANTS
# check if there is a gpu available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 20
LEARNING_RATE = 0.03
END_OF_CONTINUOUS = 6

def main():
    # load datasets. TODO: Need to find a way to include the timestamp! It doesn't accept an object as input only numbers
    print('[+] ------ LOADING DATASET ------')
    # dataset = pd.read_csv(r'Dataset\datasetNormalJoined.csv')
    dataset = pd.read_csv(r'Dataset\datasetNormalJoined.csv')

    # ONE HOT FEATURES
    dataset = one_hot(dataset)
    dataset = dataset.values

    # dataset = np.unique(dataset, axis=0)
    # dataset = np.append(dataset, dataset, axis=0)
    # dataset = np.append(np.append(dataset, dataset, axis=0), dataset, axis=0)
    # dataset = np.append(np.append(dataset, dataset, axis=0), dataset, axis=0)
    print(dataset.shape)

    print('[+] ------ PREPROCESSING ------')
    # split into train and validation set
    d_train, d_val = train_test_split(dataset, test_size=0.1, random_state=random.randint(0,100))

    print('Training data shape:', d_train.shape)
    print('Validation data shape:', d_val.shape)
    # TODO: Normalize only numerical features and not one hot encoded ones
    # normalize
    # d_train[:, :-END_OF_CONTINUOUS] = normalize(d_train[:, :-END_OF_CONTINUOUS], axis=0)
    # d_val[:, :-END_OF_CONTINUOUS] = normalize(d_val[:, :-END_OF_CONTINUOUS], axis=0)
    scaler, d_train[:, :-END_OF_CONTINUOUS] = scaleDataset(d_train[:, :-END_OF_CONTINUOUS])
    d_val[:, :-END_OF_CONTINUOUS] = scaler.transform(d_val[:, :-END_OF_CONTINUOUS])
    print(pd.DataFrame(d_train[:,:-END_OF_CONTINUOUS]).describe())

    # d_train[:, :-END_OF_CONTINUOUS] = centring(d_train[:, :-END_OF_CONTINUOUS])
    # d_val[:, :-END_OF_CONTINUOUS] = centring(d_val[:, :-END_OF_CONTINUOUS])



    print('[+] ------ PREPARATION ------')
    # create numpy datasets
    train_dataset = numpy_dataset(d_train)
    # train_dataset = numpy_dataset(normalize(dataset, axis=0))
    val_dataset = numpy_dataset(d_val)

    # create data loaders.
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, drop_last=False)

    # define model
    net = Autoencoder(n_features=dataset.shape[1]).to(DEVICE)

    # Calculate the number of traininable params
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Trainable params: ', params)

    loss_func = MeanSquaredLogError()
    # loss_func = nn.MSELoss()
    optim = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)
    # optim = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    print(summary(net, d_train.shape))

    print(f'[+] ------ START TRAINING FOR {MAX_EPOCHS} EPOCHS ------')
    losses = list()

    # training loop over epochs
    start = time.time()
    for epoch in range(1, MAX_EPOCHS+1):
        try:
            # train_loss = train(net, train_dataloader, optim, loss_func, epoch)
            train_loss = train(net, train_dataloader, optim, loss_func, epoch)
        except KeyboardInterrupt:
            print('KEYBOARD INTERRUPT DETECTED. EXITING...')
            sys.exit(1)
        try:
            val_loss = val(net, val_dataloader, optim, loss_func, epoch)
        except KeyboardInterrupt:
            print('KEYBOARD INTERRUPT DETECTED. EXITING...')
            sys.exit(1)
        losses.append([train_loss, val_loss])

    end = time.time()
    print(Back.GREEN, '[+] ------ TRAINING FINISHED ------', Style.RESET_ALL)
    print('TRAINING TIME: {:.2f}'.format(end-start))
    # plot learning curves
    plotMetrics(losses, MAX_EPOCHS)

    print('[+] ------ TESTING ------')
    test_rans = pd.read_csv(r'Dataset\datasetRANS.csv')
    test_rans = one_hot(test_rans)
    test_rans = test_rans.values
    # test_rans[:, :-6] = normalize(test_rans[:, :-6], axis=0)
    test_rans[:, :-END_OF_CONTINUOUS] = scaler.transform(test_rans[:, :-END_OF_CONTINUOUS])
    test_dataset = numpy_dataset(test_rans)

    test_normal = pd.read_csv(r'Dataset\datasetTest.csv')
    test_normal = one_hot(test_normal)

    # test_normal = np.unique(test_normal.values, axis=0)
    test_normal = test_normal.values
    # test_normal = np.unique(test_normal, axis=0)
    # test_normal[:, :-6] = normalize(test_normal[:, :-6], axis=0)
    test_normal[:, :-END_OF_CONTINUOUS] = scaler.transform(test_normal[:, :-END_OF_CONTINUOUS])
    # test_normal = normalize(test_normal, axis=0)
    test_normal_dataset = numpy_dataset(test_normal)
    print(pd.DataFrame(test_normal[:, :-END_OF_CONTINUOUS]).describe())
    print(pd.DataFrame(test_rans[:, :-END_OF_CONTINUOUS]).describe())
    # test_dataloader = DataLoader(test_dataset, batch_size=test_data.shape[0], shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
    prev_train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=True)
    test_normal_dataloader = DataLoader(test_normal_dataset, batch_size=1, shuffle=False, drop_last=True)
    # make prediction
    pred_rans_store = predict(net, test_dataloader)
    pred_norm_store = predict(net, test_normal_dataloader)
    pred_all_store = predict(net, prev_train_dataloader)

    losses_rans = list()
    losses_norm = list()
    losses_prev_train = list()
    for r in pred_rans_store:
        losses_rans.append(loss_func(r[0], r[1]).item())

    for r in pred_norm_store:
        losses_norm.append(loss_func(r[0], r[1]).item())

    for r in pred_all_store:
        losses_prev_train.append(loss_func(r[0], r[1]).item())


    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Ransomware and Normal (Unseen) Loss distribution')

    its1 = np.linspace(1, len(losses_rans), len(losses_rans))
    its2 = np.linspace(1, len(losses_norm), len(losses_norm))
    its3 = np.linspace(1, len(losses_prev_train), len(losses_prev_train))

    # show on same scale
    # ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax1.plot(its1, losses_rans, color='red')
    ax2.plot(its2, losses_norm, color='blue')
    # ax2.plot(its3, losses_prev_train, color='blue')

    plt.show()
    # multipliers for the threshold
    multiplier = [(1, 'black'), (3,'pink'), (5, 'green'), (10, 'yellow')]
    print('Ransomware samples', len(test_dataset))
    print('Validation samples', len(test_normal))
    print(*losses_rans, sep='\n')
    print('RANSOMWARE Loss:', np.asarray(losses_rans).mean())
    print('NORMAL TEST Loss:', np.asarray(losses_norm).mean())

    joined_lists = losses_norm + losses_rans
    # joined_lists = losses_prev_train + losses_rans
    its = np.linspace(1, len(joined_lists), len(joined_lists))
    plt.figure()
    plt.plot(its, joined_lists)
    # draw some thresholds
    for (m, c) in multiplier:
        plt.axhline(y=np.std(np.asarray(losses_prev_train))*m, color=c, label=f'{m}-sigma')
    plt.axvline(x=len(losses_norm), color='r')
    plt.legend()
    plt.show()

    print(*losses_norm, sep='\n')

    train_std = np.std(np.asarray(losses_prev_train))
    for (m, _) in multiplier:
        print(f"{m}-Sigma Threshold: {train_std*m}")


if __name__ == '__main__':
    main()