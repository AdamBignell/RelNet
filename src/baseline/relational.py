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
from model import RN, CNN_MLP


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

def tensor_data(data, i, bs):
    input_data = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
    output_data = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))
    input_data.data.resize_(input_data.size()).copy_(input_data)
    output_data.data.resize_(output_data.size()).copy_(output_data)
    return input_data, output_data


def cvt_data_axis(data):
    input_data = [e[0] for e in data]
    output_data = [e[1] for e in data]
    return (input_data, output_data)

    
def train(epoch, train_data, model, input_tensor, output_tensor, bs, args):
    model.train()

    random.shuffle(train_data)
    train_data = cvt_data_axis(train_data)
    N = len(train_data[0])

    for batch_idx in range(N // bs):
        # train data is a list of tuples where the first entry in the tuple is the X values and the second entry is the label Y
        input_tensor, output_tensor = tensor_data(train_data, batch_idx, bs)
        # accuracy_rel = model.train_(input_tensor, output_tensor)
        accuracy = model.naive_train_(input_tensor, output_tensor, bs)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Conflict Accuracy: {:.0f}% '.format(epoch, batch_idx * bs, N, \
                                                                                                                           100. * batch_idx * bs/ N, accuracy))
            

def test(epoch, train_data, model, input_tensor, output_tensor, bs, args):
    model.eval()

    train_data = cvt_data_axis(train_data)
    N = len(train_data[0])

    accuracy = []
    for batch_idx in range(N // bs):
        input_tensor, output_tensor = tensor_data(train_data, batch_idx, bs)
        accuracy.append(model.test_(input_tensor, output_tensor))

    accuracy = sum(accuracy) / len(accuracy)
    print('\n Test set: Conflict Accuracy: {:.0f}%\n'.format(accuracy))



def main():
    print("\n\n\t\t-~*= RUNNING RELNET =*~-\n")

    # Set the below to whatever your machine uses
    DEV_DIR = os.path.realpath(__file__[0:-len('relational.py')]) + "/DevData/"

    print("Loading dev data...")
    # trainXDev, valXDev, testXDev = loadDevData(DEV_DIR)
    # trainYDev, valYDev, testYDev = loadDevData(DEV_DIR, labels=True)

    # # THIS CONFLICTS WITH EMBEDDING EXTRACTION
    # # trainXDev = trainXDev[:,263:]
    # # valXDev = valXDev[:,263:]
    # # testXDev = testXDev[:,263:]

    # # This is just a peace of mind check
    # # Old version of the Dev Data
    # print("\tTrainX Size \t= ", trainXDev.shape)      # w/ all: (5000, 1227) w/o handcrafted: (5000, 964) 
    # print("\tValX Size \t= ", valXDev.shape)          # (600, 1227) w/o handcrafted: (5000, 964) 
    # print("\tTestX Size \t= ", testXDev.shape)        # (600, 1227) w/o handcrafted: (5000, 964) 
    # print("\tTrainY Size \t= ", trainYDev.shape)      # (5000, ) -> Just a vector
    # print("\tValY Size \t= ", valYDev.shape)          # (600, )
    # print("\tTestY Size \t= ", testYDev.shape)        # (600, )

    # New version of the dev data, sorted by label
    conX, conY, conIDs = loadConflictData(DEV_DIR)
    print("\n\tConflict X Size \t= ", conX.shape)
    print("\tConflict Y Size \t= ", conY.shape)
    print("\tConflict IDs Size \t= ", len(conIDs))

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

    random.shuffle(prop_all)
    test_size = 2000
    prop_test = prop_all[0:test_size]
    prop_train= prop_all[test_size:]

    # ======== Relational Network Goes Below ============

    DEFAULT_BS = 10 # change to 64

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
    parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP'], default='RN', 
                        help='resume from model stored')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BS, metavar='N',
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
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.model=='CNN_MLP': 
        model = CNN_MLP(args)
    else:
        model = RN(args)

    bs = args.batch_size

    # WARNING: The code below has not yet been modified to work with the Reddit embeddings.
    # The return statement prevents execution of the lines below.

    # ===========================================================
    #    EDIT THE CODE BELOW TO WORK WITH REDDIT EMBEDDINGS!
    # ===========================================================

    # CHANGE THIS TO 1227 when doing the full analysis
    NUM_FEATURES = 1227
    input_tensor = torch.FloatTensor(bs, NUM_FEATURES)
    output_tensor = torch.LongTensor(bs)

    if args.cuda:
        model.cuda()
        input_tensor = input_tensor.cuda()
        output_tensor = output_tensor.cuda()

    input_tensor = Variable(input_tensor)
    output_tensor = Variable(output_tensor)

    # Old version of the Dev Data
    # train_data = []
    # for i, tr in enumerate(trainXDev):
    #     tr = tr[:NUM_FEATURES]
    #     tup = (tr, trainYDev[i])
    #     train_data.append(tup)
    # train_data = np.array(train_data)

    # test_data = []
    # for i, te in enumerate(testXDev):
    #     te = te[:NUM_FEATURES]
    #     tup = (te, testYDev[i])
    #     test_data.append(tup)
    # test_data = np.array(test_data)

    # Count labels in Old training set
    # unique, counts = np.unique(trainYDev, return_counts=True)
    # print(dict(zip(unique, counts)))

    # Count labels in New training set
    unique, counts = np.unique(allY, return_counts=True)
    print("\nNumber of conflict/non-conflict:")
    print(dict(zip(unique, counts)))
    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        # train_data =
        train(epoch, prop_train, model, input_tensor, output_tensor, bs, args)
        test(epoch, prop_test, model, input_tensor, output_tensor, bs, args)
        # model.save_model(epoch)

    print("Training complete!")

    return

    # for epoch in range(1, args.epochs + 1):
    #     train(epoch, rel_train)
    #     test(epoch, rel_test)
    #     model.save_model(epoch)


   



if __name__ == "__main__":
    main()




