import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

BINARY_FEATURES = ['ELEVATED', 'EXISTS_IN_AUTORUN', 'CRYPT_DLLS_LOADED', 'WORKING_HOUR',
               'KILL_DEFENSE', 'AUTORUN_EDITS', 'DELETE_BACKUP']
def plotMetrics(l, epochs):
    l = np.array(l).T
    its = np.linspace(1, epochs, epochs)
    fig = plt.figure()
    fig.suptitle('Training and Validation Loss')
    plt.plot(its, l[0, :])
    plt.plot(its, l[1, :])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.savefig('Figures/learning_curves.png', dpi=300, transparent=False)
    plt.show()


def one_hot(dataset):


    for c in BINARY_FEATURES:
        if c in dataset.columns:
            y = pd.get_dummies(dataset[c], prefix=c, dtype=float)
            if f"{c}_0" not in y.columns:
                y[f"{c}_0"] = 0.0
            if f"{c}_1" not in y.columns:
                y[f"{c}_1"] = 0.0
            dataset = dataset.join(y[f"{c}_0"])
            dataset = dataset.join(y[f"{c}_1"])
            dataset = dataset.drop(columns=[c])

    return dataset


def centring(x):
    # print(X.shape)
    epsilon = 1e-7  # To prevent division by 0
    # standardize columns
    mean = np.mean(x, axis=0, keepdims=True)
    print(mean.shape)
    std = np.std(x, axis=0, keepdims=True)

    return (x - mean) / (std + epsilon)

def KLLoss(pred, target, kl):
    return ((target - pred) ** 2).sum() + kl

def scaleDataset(data, minmax=True):
    if minmax:
        scaler = MinMaxScaler().fit(data)
    else:
        scaler = StandardScaler().fit(data)

    # scaler.fit(data)

    return scaler, scaler.transform(data)