#   CMPT 419 Final Proejct - MLAllStars
#   Script to load in embeddings, construct a (smaller!) dev set, and integrate a Relational Network
#   Fall 2018
import numpy as np
import model as relnet
import sys,os

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

    # This is just a peace of mind check
    print("\tTrainX Size \t= ", trainXDev.shape)      # (5000, 1227)
    print("\tValX Size \t= ", valXDev.shape)          # (600, 1227)
    print("\tTestX Size \t= ", testXDev.shape)        # (600, 1227)
    print("\tTrainY Size \t= ", trainYDev.shape)      # (5000, ) -> Just a vector
    print("\tValY Size \t= ", valYDev.shape)          # (600, )
    print("\tTestY Size \t= ", testYDev.shape)        # (600, )

    # ======== Relational Network Goes Below ============