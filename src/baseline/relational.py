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


def tensor_data(data, i, bs):
    input_data = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
    output_data = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))

    input_data.data.resize_(input_data.size()).copy_(input_data)
    output_data.data.resize_(output_data.size()).copy_(output_data)


def cvt_data_axis(data):
    input_data = [e[0] for e in data]
    output_data = [e[1] for e in data]
    return (input_data, output_data)

    
def train(epoch, rel, model, input_tensor, output_tensor, bs, args):
    model.train()

    random.shuffle(rel)
    rel = cvt_data_axis(rel)

    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(rel, batch_idx, bs)
        # accuracy_rel = model.train_(input_tensor, output_tensor)
        accuracy_rel = model.naive_train_(input_tensor, output_tensor)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Relations accuracy: {:.0f}% '.format(epoch, batch_idx * bs * 2, len(rel[0]) * 2, \
                                                                                                                           100. * batch_idx * bs/ len(rel[0]), accuracy_rel))
            

def test(epoch, rel, model, input_tensor, output_tensor, bs, args):
    model.eval()

    rel = cvt_data_axis(rel)

    accuracy_rels = []
    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(rel, batch_idx, bs)
        accuracy_rels.append(model.test_(input_tensor, output_tensor))

    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
    print('\n Test set: Relation accuracy: {:.0f}% | Non-relation accuracy: {:.0f}%\n'.format(
        accuracy_rel))



def main():
    print("\n\n\t\t-~*= RUNNING RELNET =*~-\n")

    # Set the below to whatever your machine uses
    DEV_DIR = os.path.realpath(__file__[0:-len('relational.py')]) + "/DevData/"

    print("Loading dev data...")
    trainXDev, valXDev, testXDev = loadDevData(DEV_DIR)
    trainYDev, valYDev, testYDev = loadDevData(DEV_DIR, labels=True)

    # Remove the hand crafted features
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

    DEFAULT_BS = 1 # change to 64

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
    print("Training complete!")

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

    train_data = []
    for i, tr in enumerate(trainXDev):
        tr = tr[:NUM_FEATURES]
        tup = (tr, trainYDev[i])
        train_data.append(tup)
    train_data = np.array(train_data)

    test_data = []
    for i, te in enumerate(testXDev):
        te = te[:NUM_FEATURES]
        tup = (te, testYDev[i])
        test_data.append(tup)
    test_data = np.array(test_data)

    for epoch in range(1, args.epochs + 1):
        # train_data =
        train(epoch, train_data, model, input_tensor, output_tensor, bs, args)
        test(epoch, test_data, model, input_tensor, output_tensor, bs, args)
        model.save_model(epoch)


    return

    # for epoch in range(1, args.epochs + 1):
    #     train(epoch, rel_train)
    #     test(epoch, rel_test)
    #     model.save_model(epoch)


   



if __name__ == "__main__":
    main()




