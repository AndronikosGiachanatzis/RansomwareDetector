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
from torchmetrics import LogCoshError
from torchsummary import summary

from NumpyDataset import *
from Autoencoder import *
from train_val_pred import *
from auxiliary import *


# DEFINE CONSTANTS AND HYPERPARAMETERS
# check if there is a gpu available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 10
LEARNING_RATE = 0.03
END_OF_CONTINUOUS = len(BINARY_FEATURES)*2
BATCH_SIZE = 16

TRAIN_PATH = r'Dataset\datasetNormalJoined.csv'
TEST_RANS_PATH = r'Dataset\DefenseEvasion\datasetNonWorkingRansKill.csv'
TEST_NORMAL_PATH = r'Dataset\DefenseEvasion\datasetNormalNonWorkingRansKillAuto.csv'
def main():
    # load datasets.
    print('[+] ------ LOADING DATASET ------')
    dataset = pd.read_csv(TRAIN_PATH)
    print(pd.DataFrame(dataset['WORKING_HOUR']).value_counts())

    # ONE HOT FEATURES
    dataset = one_hot(dataset)
    n = dataset.iloc[:, :-END_OF_CONTINUOUS]
    b = dataset.iloc[:, -END_OF_CONTINUOUS:]
    dataset = dataset.values


    # keep only unique values. We want unique values in order not to make our model learn more the duplicate samples
    dataset = np.unique(dataset, axis=0)
    print(pd.DataFrame(dataset[:,0]).value_counts())
    # dataset = np.append(dataset, dataset, axis=0)
    # dataset = np.append(np.append(dataset, dataset, axis=0), dataset, axis=0)
    print('Complete Dataset shape:', dataset.shape)
    print('[+] ------ PREPROCESSING ------')
    # split into train and validation set
    d_train, d_val = train_test_split(dataset, test_size=0.1, random_state=random.randint(0, 100))

    print('Training data shape:', d_train.shape)
    print('Validation data shape:', d_val.shape)

    # normalize
    scaler, d_train[:, :-END_OF_CONTINUOUS] = scaleDataset(d_train[:, :-END_OF_CONTINUOUS])
    d_val[:, :-END_OF_CONTINUOUS] = scaler.transform(d_val[:, :-END_OF_CONTINUOUS])
    print(pd.DataFrame(d_train[:,:-END_OF_CONTINUOUS]).describe())


    print('[+] ------ PREPARATION ------')
    # create numpy datasets
    train_dataset = numpy_dataset(d_train)
    val_dataset = numpy_dataset(d_val)

    # create data loaders.
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # define model
    net = Autoencoder(n_features=dataset.shape[1]).to(DEVICE)

    # Calculate the number of traininable params
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Trainable params: ', params)

    loss_func = MeanSquaredLogError() # MSLE is better for this task: https://builtin.com/data-science/msle-vs-mse
    # loss_func = nn.MSELoss()
    optim = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)
    # optim = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    print(summary(net, d_train.shape))

    print(f'[+] ------ START TRAINING FOR {MAX_EPOCHS} EPOCHS ------')
    losses = list()

    # training loop over epochs
    start = time.time()
    print(start)
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
    print(end)
    print(Back.GREEN, '[+] ------ TRAINING FINISHED ------', Style.RESET_ALL)
    print('TRAINING TIME: {:.2f}'.format(end-start))
    # plot learning curves
    plotMetrics(losses, MAX_EPOCHS)

    print('\n[+] ------ TESTING ------')
    # load ransomware dataset
    test_rans = pd.read_csv(TEST_RANS_PATH)
    test_rans = one_hot(test_rans)
    test_rans = test_rans.values
    # test_rans[:, :-END_OF_CONTINUOUS] = normalize(test_rans[:, :-END_OF_CONTINUOUS], axis=0)
    test_rans[:, :-END_OF_CONTINUOUS] = scaler.transform(test_rans[:, :-END_OF_CONTINUOUS])
    test_dataset = numpy_dataset(test_rans)

    # load normal test dataset
    test_normal = pd.read_csv(TEST_NORMAL_PATH)
    test_normal = one_hot(test_normal)

    # test_normal = np.unique(test_normal.values, axis=0)
    test_normal = test_normal.values
    # test_normal = np.unique(test_normal, axis=0)
    # test_normal[:, :-END_OF_CONTINUOUS] = normalize(test_normal[:, :-END_OF_CONTINUOUS], axis=0)
    test_normal[:, :-END_OF_CONTINUOUS] = scaler.transform(test_normal[:, :-END_OF_CONTINUOUS])
    # test_normal = normalize(test_normal, axis=0)
    test_normal_dataset = numpy_dataset(test_normal)
    print(pd.DataFrame(test_normal[:, :-END_OF_CONTINUOUS]).describe())
    print(pd.DataFrame(test_rans[:, :-END_OF_CONTINUOUS]).describe())
    # test_dataloader = DataLoader(test_dataset, batch_size=test_data.shape[0], shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
    prev_train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=True)
    test_normal_dataloader = DataLoader(test_normal_dataset, batch_size=1, shuffle=False, drop_last=True)

    # make predictions for all sets
    pred_rans_store = predict(net, test_dataloader)
    pred_norm_store = predict(net, test_normal_dataloader)
    pred_all_store = predict(net, prev_train_dataloader)

    # create loss lists
    losses_rans = list()
    losses_norm = list()
    losses_prev_train = list()
    for r in pred_rans_store:
        losses_rans.append(loss_func(r[0], r[1]).item())

    for r in pred_norm_store:
        losses_norm.append(loss_func(r[0], r[1]).item())

    for r in pred_all_store:
        losses_prev_train.append(loss_func(r[0], r[1]).item())

    # plot ransomware and normal test loss plots
    # fig, (ax1, ax2) = plt.subplots(2)
    # fig.suptitle('Ransomware and Normal Test Loss')
    # fig.subplots_adjust(hspace=0.6)

    its1 = np.linspace(1, len(losses_rans), len(losses_rans))
    its2 = np.linspace(1, len(losses_norm), len(losses_norm))
    its3 = np.linspace(1, len(losses_prev_train), len(losses_prev_train))

    # # show on same scale
    # # ax1.set_ylim(bottom=0)
    # ax2.set_ylim(bottom=0)
    # ax1.plot(its1, losses_rans, color='red')
    # ax2.plot(its2, losses_norm, color='blue')
    #
    # # add titles and labels
    # ax1.set_title('Ransomware')
    # ax1.set_ylabel('Loss')
    # ax1.set_xlabel('Sample')
    # ax2.set_title('Normal Test')
    # ax2.set_ylabel('Loss')
    # ax2.set_xlabel('Sample')
    #
    # # ax2.plot(its3, losses_prev_train, color='blue')
    #
    # plt.savefig('Figures/TestSeparate.png', dpi=300, transparent=False)
    # plt.show()

    # multipliers and percentiles for the threshold
    multiplier = [(4, 'green'), (5, 'black')]
    percentiles = [(90, 'yellow'), (95, 'pink'), (99, 'black')]

    print('Ransomware samples', len(test_dataset))
    print('Normal Test samples', len(test_normal))

    print('\n Ransomware Losses: ')
    print(*losses_rans, sep='\n')

    print()
    print('RANSOMWARE Loss:', np.asarray(losses_rans).mean())
    print('NORMAL TEST Loss:', np.asarray(losses_norm).mean())

    # plot ransomware and normal plots in one figure
    joined_lists = losses_norm + losses_rans
    # joined_lists = losses_prev_train + losses_rans
    its = np.linspace(1, len(joined_lists), len(joined_lists))
    fig = plt.figure(figsize=(8,6))
    fig.suptitle(' Normal Test (Left) and Ransomware (Right) Reconstruction Errors')
    plt.xlabel('Sample')
    plt.ylabel('Loss')


    # calculate train statistics
    train_std = np.std(np.asarray(losses_prev_train))
    train_mean = np.mean(np.asarray(losses_prev_train))

    # draw the various thresholds
    for (m, c) in multiplier:
        plt.axhline(y=train_mean + train_std*m, color=c, label=f'{m}-sigma threshold')
    plt.axvline(x=len(joined_lists) - len(losses_rans), color='r', label='Normal/Ransomware samples boundary')
    # random.shuffle(joined_lists)
    plt.plot(its, joined_lists)
    plt.legend()
    plt.savefig('Figures/TestJoinSigma.png', dpi=300, transparent=False)
    plt.show()

    # fig = plt.figure()
    # fig.suptitle('Ransomware (Right) and Normal Test (Left) Loss. Percentile Thresholds')
    # plt.xlabel('Sample')
    # plt.ylabel('Loss')
    #
    # # draw the various thresholds
    # for (p, c) in percentiles:
    #     plt.axhline(y=np.percentile(losses_prev_train, p)+train_std, color=c, label=f'{p}-percentile')
    # plt.axvline(x=len(joined_lists) - len(losses_rans), color='r', label='Normal/Ransomware samples boundary')
    # # random.shuffle(joined_lists)
    # plt.plot(its, joined_lists)
    # plt.legend()
    # plt.savefig('Figures/TestJoinPercentiles.png', dpi=300, transparent=False)
    # plt.show()

    print('\nNormal Test Losses: ')
    print(*np.unique(np.array(losses_norm)), sep='\n')

    # print sigma rules
    print('THRESHOLDS')
    for (m, _) in multiplier:
        print(f"{m}-Sigma Threshold: {train_mean + train_std*m}")

    # print percentile thresholds
    # print()
    # for (p, _) in percentiles:
    #     print(f"{p}-percentile+std Threshold: {np.percentile(losses_prev_train, p)+train_std}")
    #


if __name__ == '__main__':
    main()