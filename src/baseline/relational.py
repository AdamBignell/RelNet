#   CMPT 419 Final Proejct - MLAllStars
#   Script to load in embeddings, construct a (smaller!) dev set, and integrate a Relational Network
#   Fall 2018

from __future__ import print_function

import argparse
import numpy as np
import sys,os
import pickle
import random
import torch
from torch.autograd import Variable
from model import RN
from sklearn.metrics import roc_auc_score

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

def loadConflictData(rootDirectory):
    """Loads specifically connections marked as a conflict"""
    x = np.load(open(rootDirectory + "conflict_dev.npy", 'rb'))
    y = np.load(open(rootDirectory + "conflict_Y_dev.npy", 'rb'))
    ids = open(rootDirectory + "conflictIDs_dev.txt", 'r').readlines()
    return x, y, ids

def loadNonconflictData(rootDirectory):
    """Loads specifically connections marked as a non-conflict"""
    x = np.load(open(rootDirectory + "nonconflict_dev.npy", 'rb'))
    y = np.load(open(rootDirectory + "nonconflict_Y_dev.npy", 'rb'))
    ids = open(rootDirectory + "nonconflictIDs_dev.txt", 'r').readlines()
    return x, y, ids




def tensor_data(data, i, bs, args, leftover=False):
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if leftover:
        input_tensor = torch.from_numpy(np.asarray(data[0][bs * i:]))
        output_tensor = torch.from_numpy(np.asarray(data[1][bs * i:]))
    else:
        input_tensor = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
        output_tensor = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))

    input_tensor.data.resize_(input_tensor.size()).copy_(input_tensor)
    output_tensor.data.resize_(output_tensor.size()).copy_(output_tensor)


    input_tensor = input_tensor.to(device)
    output_tensor = output_tensor.to(device)


    return input_tensor, output_tensor


def cvt_data_axis(data):
    input_data = [e[0] for e in data]
    output_data = [e[1] for e in data]
    return (input_data, output_data)


def train(epoch, train_data, model, bs, args):
    model.train()

    # Uncomment this later to shuffle
    random.shuffle(train_data)

    train_data = cvt_data_axis(train_data)
    N = len(train_data[0])

    for batch_idx in range(N // bs):
        # train data is a list of tuples where the first entry in the tuple is the X values and the second entry is the label Y
        input_tensor, output_tensor = tensor_data(train_data, batch_idx, bs, args)

        accuracy = model.train_(input_tensor, output_tensor, bs, args)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Conflict Accuracy: {:.0f}% '.format(
                epoch, batch_idx * bs, N, batch_idx*bs/N*100, accuracy))

    # TODO : Refactor (low priority, but duplicated code)
    leftover = N - bs*(N//bs)
    if leftover > 0:
        batch_idx = N // bs
        input_tensor, output_tensor = tensor_data(train_data, batch_idx, bs, args, leftover=True)
        accuracy = model.train_(input_tensor, output_tensor, leftover, args)

    return

def test(epoch, test_data, model, bs, args):
    model.eval()

    test_data = cvt_data_axis(test_data)
    N = len(test_data[0])

    allLabels = test_data[1]
    allPredProbs = []
    accuracy = []

    for batch_idx in range(N // bs):
        input_tensor, output_tensor = tensor_data(test_data, batch_idx, bs, args)
        acc, predPos = model.test_(input_tensor, output_tensor, args)

        # Compare labels (output_tensor) to input_tensor0
        # fpr, tpr, _ = roc_curve(output_tensor, pred)

        accuracy.append(acc)
        allPredProbs.extend(predPos.tolist())

    print('\n')

    # TODO : Refactor (low priority, but duplicated code)
    leftover = N - bs*(N//bs)
    if leftover > 0:
        batch_idx = N // bs
        input_tensor, output_tensor = tensor_data(test_data, batch_idx, bs, args, leftover=True)
        acc, predPos = model.test_(input_tensor, output_tensor, args)

        accuracy.append(acc)
        allPredProbs.extend(predPos.tolist())

    # Compute the AUC given ground truth (label) and probabilities for the positive class (i.e. the True class)
    auc = roc_auc_score(allLabels, allPredProbs)
    auc = round(auc, 4)

    accuracy = sum(accuracy) / len(accuracy)
    print('\n Test set: Conflict Accuracy: {:.0f}%\n'.format(accuracy))
    print('\n AUC Score: {}\n'.format(auc))

    return


def main():
    DEFAULT_BS = 64
    DEFAULT_EPOCHS = 5
    TOTAL_FEATURES = 1227
    NUM_HANDCRAFTED = 263
    NUM_FEATURES = TOTAL_FEATURES - NUM_HANDCRAFTED
    test_size = 2000

    print("\n\n\t\t-~*= RUNNING RELNET =*~-\n")

    # Set the below to whatever your machine uses
    DEV_DIR = os.path.realpath(__file__[0:-len('relational.py')]) + "/DevData/"

    print("Loading dev data...")

    # New version of the dev data, sorted by label
    conX, conY, conIDs = loadConflictData(DEV_DIR)
    print("\n\tConflict X Size \t= ", conX.shape)
    print("\tConflict Y Size \t= ", conY.shape)
    print("\tConflict IDs Size \t= ", len(conIDs))
    print()

    nonX, nonY, nonIDs = loadNonconflictData(DEV_DIR)
    print("\tNon-Conflict X Size \t= ", conX.shape)
    print("\tNon-Conflict Y Size \t= ", conY.shape)
    print("\tNon-Conflict IDs Size \t= ", len(conIDs))

    allX = np.concatenate((conX, nonX), axis=0)
    allY = np.concatenate((conY, nonY), axis=0)

    # This is the version of the data with even representation
    prop_all = []
    for i in range(len(allX)):
        prop_all.append((allX[i], allY[i][0]))
    prop_all = np.array(prop_all)

    np.random.shuffle(prop_all)
    prop_test = prop_all[0:test_size]
    prop_train = prop_all[test_size:]

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
    parser.add_argument('--model', type=str, choices=['RN'], default='RN',
                        help='resume from model stored')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BS, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, metavar='N',
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

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    cuda = args.cuda

    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    model = RN(args)

    bs = args.batch_size

    unique, counts = np.unique(allY, return_counts=True)
    print("\nNumber of conflict/non-conflict:")
    print(dict(zip(unique, counts)))
    print("\nTraining...")

    for epoch in range(1, args.epochs + 1):
        train(epoch, prop_train, model, bs, args)
        test(epoch, prop_test, model, bs, args)

    print("Training complete!")

    return


if __name__ == "__main__":
    main()




