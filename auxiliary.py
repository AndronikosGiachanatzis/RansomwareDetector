import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plotMetrics(l, epochs):
    l = np.array(l).T
    print(l.shape)
    its = np.linspace(1, epochs, epochs)
    plt.figure()
    plt.plot(its, l[0, :])
    plt.plot(its, l[1, :])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.savefig('learning_curves.png')
    plt.show()


def one_hot(dataset):
    y = pd.get_dummies(dataset.ELEVATED, prefix='ELEVATED', dtype=float)
    if 'ELEVATED_0' not in y.columns:
        y['ELEVATED_0'] = 0.0
    dataset = dataset.join(y['ELEVATED_0'])
    dataset = dataset.join(y['ELEVATED_1'])

    y = pd.get_dummies(dataset.ELEVATED, prefix='EXISTS_IN_AUTORUN', dtype=float)
    if 'EXISTS_IN_AUTORUN_0' not in y.columns:
        y['EXISTS_IN_AUTORUN_0'] = 0.0
    dataset = dataset.join(y['EXISTS_IN_AUTORUN_0'])
    dataset = dataset.join(y['EXISTS_IN_AUTORUN_1'])


    dataset = dataset.drop(columns=['ELEVATED', "EXISTS_IN_AUTORUN"])

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