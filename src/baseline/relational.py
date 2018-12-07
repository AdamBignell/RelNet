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
from model import RN, SimpleAutoEncoder, VariationalAutoEncoder
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

# Define Consts
DEFAULT_BS = 64
TOTAL_FEATURES = 1227
NUM_HANDCRAFTED = 263
NUM_FEATURES = TOTAL_FEATURES - NUM_HANDCRAFTED
USE_LEFTOVERS = True
USE_BCE = False
USE_AUTOENCODERS = False
USE_TESTALL = True

# Change number of epochs and folds here:
DEFAULT_EPOCHS = 10
N_FOLDS = 5

# Set the below to whatever your machine uses
DEV_DIR = os.path.realpath(__file__[0:-len('relational.py')]) + "/DevData/"
MODEL_DIR = os.path.realpath(__file__[0:-len('relational.py')]) + "/model/"
DATA_DIR = "/media/dxguo/exFAT/School/conflict_data/prediction/"
BEST_BCE = "BCE_epoch_100.pth"
BEST_NLL = "NLL_epoch_97.pth"

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

def loadFullData(rootDirectory):
    """Loads the complete data set"""
    prop_all = np.load(open(rootDirectory + "prop_all.npy", 'rb'))
    return prop_all


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


def autoencoder_tensor(data, i, bs, args, leftover=False):
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if leftover:
        input_tensor = torch.from_numpy(data[bs * i:])
    else:
        input_tensor = torch.from_numpy(data[bs*i:bs*(i+1)])

    input_tensor.data.resize_(input_tensor.size()).copy_(input_tensor)

    input_tensor = input_tensor.to(device)

    return input_tensor


def cvt_data_axis(data):
    input_data = [e[0] for e in data]
    output_data = [e[1] for e in data]
    return (input_data, output_data)


def tensor_data_encoded(train_data, batch_idx, bs, args, user_autoencoder, sub_autoencoder, leftover=False):
    input_tensor, output_tensor = tensor_data(train_data, batch_idx, bs, args, leftover)

    # Resize the input tensor
    user_embedding, source_embedding, target_embedding, post_embedding = extract_embeddings(input_tensor)
    # embeds = [first_embedding, second_embedding, third_embedding]
    final_feats = []

    final_feats.append(user_autoencoder.encode(user_embedding.float()))
    final_feats.append(sub_autoencoder.encode(source_embedding.float()))
    final_feats.append(sub_autoencoder.encode(target_embedding.float()))
    final_feats.append(post_embedding.float())

    # minibatch * 256 (64x4)
    input_tensor = torch.cat(final_feats, 1)

    return input_tensor, output_tensor


def train(epoch, train_data, model, bs, args, user_autoencoder, sub_autoencoder):
    model.train()

    random.shuffle(train_data)

    train_data = cvt_data_axis(train_data)
    N = len(train_data[0])

    for batch_idx in range(N // bs):
        # train data is a list of tuples where the first entry in the tuple is the X values and the second entry is the label Y
        if args.autoencoder:
            input_tensor, output_tensor = tensor_data_encoded(train_data, batch_idx, bs, args, user_autoencoder, sub_autoencoder)
        else:
            input_tensor, output_tensor = tensor_data(train_data, batch_idx, bs, args)

        accuracy = model.train_(input_tensor, output_tensor, bs, args)

        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)] Conflict Accuracy: {:.0f}% '.format(
        #         epoch, batch_idx * bs, N, batch_idx*bs/N*100, accuracy))

    if args.leftovers:
        # TODO : Refactor (low priority, but duplicated code)
        leftover = N - bs*(N//bs)
        if leftover > 0:
            batch_idx = N // bs
            if args.autoencoder:
                input_tensor, output_tensor = tensor_data_encoded(train_data, batch_idx, bs, args, \
                                                                  user_autoencoder, sub_autoencoder, leftover=True)
            else:
                input_tensor, output_tensor = tensor_data(train_data, batch_idx, bs, args)
            accuracy = model.train_(input_tensor, output_tensor, leftover, args)

    return


def test(epoch, test_data, model, bs, args, user_autoencoder=None, sub_autoencoder=None):
    model.eval()

    test_data = cvt_data_axis(test_data)
    N = len(test_data[0])

    allLabels = test_data[1]
    allPredProbs = []
    accuracy = []

    for batch_idx in range(N // bs):
        if args.autoencoder:
            input_tensor, output_tensor = tensor_data_encoded(test_data, batch_idx, bs, args, user_autoencoder, sub_autoencoder)
        else:
            input_tensor, output_tensor = tensor_data(test_data, batch_idx, bs, args)
        acc, predPos, pred = model.test_(input_tensor, output_tensor, args)

        # Compare labels (output_tensor) to input_tensor0
        # fpr, tpr, _ = roc_curve(output_tensor, pred)

        accuracy.append(acc)
        allPredProbs.extend(predPos.tolist())

    print('\n')

    if args.leftovers:
        # TODO : Refactor (low priority, but duplicated code)
        leftover = N - bs*(N//bs)
        if leftover > 0:
            batch_idx = N // bs
            if args.autoencoder:
                input_tensor, output_tensor = tensor_data_encoded(test_data, batch_idx, bs, args, \
                                                                  user_autoencoder, sub_autoencoder, leftover=True)
            else:
                input_tensor, output_tensor = tensor_data(test_data, batch_idx, bs, args)
            acc, predPos, pred = model.test_(input_tensor, output_tensor, args)

            accuracy.append(acc)
            allPredProbs.extend(predPos.tolist())

    allLabels = allLabels[:len(allPredProbs)]
    # Compute the AUC given ground truth (label) and probabilities for the positive class (i.e. the True class)
    auc = roc_auc_score(allLabels, allPredProbs)
    auc = round(auc, 4)

    accuracy = sum(accuracy) / len(accuracy)
    if args.test_all:
        print('Conflict Accuracy: {:.0f}%'.format(accuracy))
        print('AUC Score: {}'.format(auc))
    else:
        print('Test set on epoch {}: Conflict Accuracy: {:.0f}%'.format(epoch, accuracy))
        print('AUC Score: {}'.format(auc))
    
    return


def train_autoencoder(epoch, train_data, user_autoencoder, sub_autoencoder, bs, args):
    user_autoencoder.train()
    sub_autoencoder.train()

    train_data = train_data[:, 0]
    # train_data = train_data[:, 0]
    random.shuffle(train_data)
    train_data = np.array(list(train_data[:][:]), dtype=np.float32)

    # train_data = cvt_data_axis(train_data)
    N = len(train_data)

    for batch_idx in range(N // bs):
        # train data is a list of tuples where the first entry in the tuple is the X values and the second entry is the label Y
        input_tensor = autoencoder_tensor(train_data, batch_idx, bs, args)

        #     data = data[0]
        #     data = Variable(torch.from_numpy(data)).float()
        #
        user_embedding, source_embedding, target_embedding, post_embedding = extract_embeddings(input_tensor)

        loss1 = user_autoencoder.train_(user_embedding, args)
        loss2 = sub_autoencoder.train_(source_embedding, args)
        loss3 = sub_autoencoder.train_(target_embedding, args)

        # if batch_idx % args.log_interval == 0:
        #     print('Autoencoder training Epoch: {}, [{}/{} ({:.0f}%)] '.format(
        #         epoch, batch_idx * bs, N, batch_idx*bs/N*100))
            # code = autoencoder.encode(embed)

    print("User autoencoder loss:", loss1)
    print("Source embedding loss:", loss2)
    print("Target embedding loss:", loss3)
    print("====================================")

        #     if (i+1) % (len(prop_all)//100) == 0:
        #         print('[{}/{} ({:.0f}%)]'.format(i, len(prop_all), i/len(prop_all)*100))
        #     # else:
        #     #     print("{}/{}".format(i+1, (len(prop_all)/1000)))
        #
    # print('epoch {} complete'.format(epoch+1))

    # if args.leftovers:
    #     # TODO : Refactor (low priority, but duplicated code)
    #     leftover = N - bs*(N//bs)
    #     if leftover > 0:
    #         batch_idx = N // bs
    #         input_tensor, output_tensor = tensor_data(train_data, batch_idx, bs, args, leftover=True)
    #         accuracy = model.train_(input_tensor, output_tensor, leftover, args)

    return


def extract_embeddings(input_feats):
    """Extract embeddings from 1227 long input vector"""
    INPUT_FEAT_LENGTH = 1227
    HANDCRAFTED_FEATURES = 263
    batch_size = input_feats.shape[0]

    # input_feats = input_feats.view(NUM_FEATURES)

    input_feats = input_feats[:, HANDCRAFTED_FEATURES:]

    first_embedding = input_feats[:, :300]
    second_embedding = input_feats[:, 300:600]
    third_embedding = input_feats[:, 600:900]
    post_embedding = input_feats[:, 900:]

    embeddings = [first_embedding, second_embedding, third_embedding, post_embedding]

    return embeddings

def main():

    print("\n\n\t\t-~*= RUNNING RELNET =*~-\n")

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
    parser.add_argument('--BCE', action='store_true', default=USE_BCE,
                        help='use Binary Cross Entropy loss function')
    parser.add_argument('--leftovers', action='store_true', default=USE_LEFTOVERS,
                        help='train on leftovers after mini-batches')
    parser.add_argument('--autoencoder', action='store_true', default=USE_AUTOENCODERS,
                        help='train on leftovers after mini-batches')
    parser.add_argument('--test-all', action='store_true', default=USE_TESTALL,
                        help='train on leftovers after mini-batches')
    args = parser.parse_args()


    """Prepare the Relational Network"""

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    cuda = args.cuda

    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    model = RN(args)

    bs = args.batch_size

    if args.cuda:
        model.cuda()

    """Train or Test"""
    if args.test_all:
        test_manager(model, bs, args)
    else:
        train_manager(model, bs, args)


    return


def test_manager(model, bs, args):

    print("Loading test data...")
    prop_all = loadFullData(DATA_DIR)
    print("\tFull Data Shape \t= ", prop_all.shape)

    print("\nLoading saved state...")
    if args.BCE:
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, BEST_BCE)))
    else:
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, BEST_NLL)))

    print("\nTesting...")
    test(0, prop_all, model, bs, args)

    print("\nTesting complete!")

    return


def train_manager(model, bs, args):
    # Change test set size here:
    test_size = 2000

    print("Loading dev data...")

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

    np.random.shuffle(prop_all)
    prop_test = prop_all[0:test_size]
    prop_train = prop_all[test_size:]

    """ 
    Train the AutoEncoder
    
    The autoencoder will work on 300-length vectors: the user embedding, the source subreddit embedding,
    and the target subreddit embedding. These can be treated as individual objects.
    
    Note that the post embeddings are already of length 64, hence they do not need to be passed through
    the autoencoder.
    """
    if args.autoencoder:
        user_autoencoder = SimpleAutoEncoder()
        sub_autoencoder = SimpleAutoEncoder()
        user_autoencoder = VariationalAutoEncoder()
        sub_autoencoder = VariationalAutoEncoder()

        if os.path.isfile('./user_autoencoder.pth'):
            user_autoencoder.load_state_dict(torch.load('./user_autoencoder.pth'))
            sub_autoencoder.load_state_dict(torch.load('./sub_autoencoder.pth'))
        else:
            autoEpochs = 10
            ae_bs = 64
            print("~~~ Starting autoencoder training! ~~~")
            for epoch in range(1, autoEpochs + 1):
                print("Training autoencoder: epoch {}".format(epoch))
                train_autoencoder(epoch, prop_all, user_autoencoder, sub_autoencoder, ae_bs, args)

            torch.save(user_autoencoder.state_dict(), './user_autoencoder.pth')
            torch.save(sub_autoencoder.state_dict(), './sub_autoencoder.pth')
    else:
        user_autoencoder, sub_autoencoder = None, None

    # Count labels in New training set
    unique, counts = np.unique(allY, return_counts=True)
    print("\nNumber of conflict/non-conflict:")
    print(dict(zip(unique, counts)))

    print("\nTraining...")

    for epoch in range(1, args.epochs + 1):
        # Split into training set and validation set
        kfold = KFold(n_splits = N_FOLDS)
        total_accuracy = 0
        for train_idx, test_idx in kfold.split(prop_all):
            prop_train = prop_all[train_idx, :]
            prop_test = prop_all[test_idx, :]
            train(epoch, prop_train, model, bs, args, user_autoencoder, sub_autoencoder)
            test(epoch, prop_test, model, bs,
                               args, user_autoencoder, sub_autoencoder)
        if epoch % 5 == 0:
            model.save_model(epoch, args)

    print("Training complete!")

    return


if __name__ == "__main__":
    main()




