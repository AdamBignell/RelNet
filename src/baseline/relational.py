#   CMPT 419 Final Proejct - MLAllStars
#   Script to load in embeddings, construct a (smaller!) dev set, and integrate a Relational Network
#   Fall 2018
import numpy as np
import sys,os
from model import RN, CNN_MLP
import argparse

def loadTrainDev(rootDirectory, labels=False):
    """Load just the training dev data"""
    which = "X"
    if labels:
        which = "Y"
    dataDir = rootDirectory + "train_" + which + "_dev.npy"
    train_dev = np.load(open(dataDir, 'rb'))
    return train_dev

def loadValDev(rootDirectory, labels=False):
    """Load just the validation dev data"""
    which = "X"
    if labels:
        which = "Y"
    dataDir = rootDirectory + "val_" + which + "_dev.npy"
    val_dev = np.load(open(dataDir, 'rb'))
    return val_dev

def loadTestDev(rootDirectory, labels=False):
    """Load just the testing dev data"""
    which = "X"
    if labels:
        which = "Y"
    dataDir = rootDirectory + "test_" + which + "_dev.npy"
    val_dev = np.load(open(dataDir, 'rb'))
    return val_dev

def loadDevData(rootDirectory, labels=False):
    """Load all of the dev-data. Will return Y values instead of X if labels=True"""
    train_dev = loadTrainDev(rootDirectory, labels)
    val_dev = loadValDev(rootDirectory, labels)
    test_dev = loadTestDev(rootDirectory, labels)
    return train_dev, val_dev, test_dev


if __name__ == "__main__":
    print("\n\n\t\t-~*= RUNNING RELNET =*~-\n")

    # Set the below to whatever your machine uses
    DEV_DIR = os.path.realpath(__file__[0:-len('relational.py')]) + "/DevData/"
    print("Loading dev data...")
    trainXDev, valXDev, testXDev = loadDevData(DEV_DIR)
    trainYDev, valYDev, testYDev = loadDevData(DEV_DIR, labels=True)
    trainXDev = trainXDev[:,263:]
    trainYDev = trainYDev[:,263:]
    valXDev = valXDev[:,263:]
    valYDev = valYDev[:,263:]
    testXDev = testXDev[:,263:]
    testYDev = testYDev[:,263:]

    # This is just a peace of mind check
    print("\tTrainX Size \t= ", trainXDev.shape)      # (5000, 1227)
    print("\tValX Size \t= ", valXDev.shape)          # (600, 1227)
    print("\tTestX Size \t= ", testXDev.shape)        # (600, 1227)
    print("\tTrainY Size \t= ", trainYDev.shape)      # (5000, ) -> Just a vector
    print("\tValY Size \t= ", valYDev.shape)          # (600, )
    print("\tTestY Size \t= ", testYDev.shape)        # (600, )

    # ======== Relational Network Goes Below ============

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
    parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP'], default='RN', 
                        help='resume from model stored')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

                        # Attention MLAllStars: I changed 
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str,
                        help='resume from model stored')
    args = parser.parse_args()     

    # Prepare the Relational Network
    model = RN(args)