import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


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
    columns = ['ELEVATED', 'EXISTS_IN_AUTORUN', 'CRYPT_DLLS_LOADED', 'WORKING_HOUR']

    for c in columns:
        if c in dataset.columns:
            y = pd.get_dummies(dataset[c], prefix=c, dtype=float)
            if f"{c}_0" not in y.columns:
                y[f"{c}_0"] = 0.0
            if f"{c}_1" not in y.columns:
                y[f"{c}_1"] = 1.0
            dataset = dataset.join(y[f"{c}_0"])
            dataset = dataset.join(y[f"{c}_1"])
            dataset = dataset.drop(columns=[c])
    # if 'ELEVATED' in dataset.columns:
    #     y = pd.get_dummies(dataset.ELEVATED, prefix='ELEVATED', dtype=float)
    #     if 'ELEVATED_0' not in y.columns:
    #         y['ELEVATED_0'] = 0.0
    #     if 'ELEVATED_1' not in y.columns:
    #         y['ELEVATED_1'] = 1.0
    #     dataset = dataset.join(y['ELEVATED_0'])
    #     dataset = dataset.join(y['ELEVATED_1'])
    #     dataset = dataset.drop(columns=['ELEVATED'])
    #
    # if 'EXISTS_IN_AUTORUN' in dataset.columns:
    #     y = pd.get_dummies(dataset.EXISTS_IN_AUTORUN, prefix='EXISTS_IN_AUTORUN', dtype=float)
    #     if 'EXISTS_IN_AUTORUN_0' not in y.columns:
    #         y['EXISTS_IN_AUTORUN_0'] = 0.0
    #     if 'EXISTS_IN_AUTORUN_1' not in y.columns:
    #         y['EXISTS_IN_AUTORUN_1'] = 1.0
    #     dataset = dataset.join(y['EXISTS_IN_AUTORUN_0'])
    #     dataset = dataset.join(y['EXISTS_IN_AUTORUN_1'])
    #     dataset = dataset.drop(columns='EXISTS_IN_AUTORUN')
    #
    # if 'CRYPT_DLLS_LOADED' in dataset.columns:
    #     y = pd.get_dummies(dataset.CRYPT_DLLS_LOADED, prefix='CRYPT_DLLS_LOADED', dtype=float)
    #     if 'CRYPT_DLLS_LOADED_0' not in y.columns:
    #         y['CRYPT_DLLS_LOADED_0'] = 0.0
    #     if 'CRYPT_DLLS_LOADED_1' not in y.columns:
    #         y['CRYPT_DLLS_LOADED_1'] = 1.0
    #     dataset = dataset.join(y['CRYPT_DLLS_LOADED_0'])
    #     dataset = dataset.join(y['CRYPT_DLLS_LOADED_1'])
    #     dataset = dataset.drop(columns='CRYPT_DLLS_LOADED')
    #
    #     if 'WORKING_HOUR' in dataset.columns:
    #         y = pd.get_dummies(dataset.WORKING_HOUR, prefix='WORKING_HOUR', dtype=float)
    #         if 'WORKING_HOUR_0' not in y.columns:
    #             y['WORKING_HOUR_0'] = 0.0
    #         if 'WORKING_HOUR_1' not in y.columns:
    #             y['WORKING_HOUR_1'] = 1.0
    #         dataset = dataset.join(y['WORKING_HOUR_0'])
    #         dataset = dataset.join(y['WORKING_HOUR_1'])
    #         dataset = dataset.drop(columns=['WORKING_HOUR'])


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