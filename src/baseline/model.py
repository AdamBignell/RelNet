import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
import os


class FCOutputModel(nn.Module):
    def __init__(self):
        super(FCOutputModel, self).__init__()
        self.fc2 = nn.Linear(256, 256)

        # Conflict = {True or False}
        N_CLASSES = 2
        self.fc3 = nn.Linear(256, N_CLASSES)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=0)


class FCOutputModelBCE(nn.Module):
    def __init__(self):
        super(FCOutputModelBCE, self).__init__()
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return torch.sigmoid(x)


class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name = name

    def train_(self, input_feats, label, batch_size, args):
        """Naive train using the naive forward method"""
        self.optimizer.zero_grad()
        # Run input batch across relational net
        output = self.forward(input_feats, args)

        # Calculate loss on this batch
        if not args.BCE:
            loss = F.nll_loss(output, label)
        else:
            loss = F.binary_cross_entropy(output.view(batch_size), label.float())

        loss.backward()
        self.optimizer.step()

        # Get the actual argmax that indicates the label
        if not args.BCE:
            pred = output.data.max(1)[1]
            correct = pred.eq(label.data).cpu().sum()
        else:
            pred = output.view(output.shape[0]).round()
            correct = pred.eq(label.data.float()).cpu().sum()

        accuracy = correct * 100. / len(label)
        return accuracy


    def test_(self, input_feats, label, args):
        output = self.forward(input_feats, args)

        # Get the actual argmax that indicates the label
        if not args.BCE:
            pred = output.data.max(1)[1]
            correct = pred.eq(label.data).cpu().sum()
            # predictions are log probabilities, so convert back to actual probs
            if args.cuda:
                posClassProbs = torch.exp(output)[:, 1].cpu().detach().numpy()
            else:
                posClassProbs = torch.exp(output)[:, 1].detach().numpy()
        else:
            pred = output.view(output.shape[0]).round()
            correct = pred.eq(label.data.float()).cpu().sum()
            if args.cuda:
                posClassProbs = output[:, 0].cpu().detach().numpy()
            else:
                posClassProbs = output[:, 0].detach().numpy()

        accuracy = correct * 100. / len(label)

        return accuracy, posClassProbs, pred

    def save_model(self, epoch, args):
        torch.save(self.state_dict(), 'model/{}_epoch_{:02d}.pth'.format('BCE' if args.BCE else 'NLL', epoch))



"""Default args
Model:          RN
Batch-size:     64
Epochs:         20
Learning rate:  0.0001
No-cuda:        False
Seed:           1
Log-interval:   10
"""

OBJ_LENGTH = 64
HIDDEN_LAYER_UNITS = 256
NUM_FEATURES = 964
REG_PARAM = 0.00001


class VariationalAutoEncoder(nn.Module):

    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()

        self.input_length = 300
        # self.input_length = 2 * OBJ_LENGTH

        self.fc1 = nn.Linear(300, 150)
        self.fc21 = nn.Linear(150, 64)
        self.fc22 = nn.Linear(150, 64)
        self.fc3 = nn.Linear(64, 150)
        self.fc4 = nn.Linear(150, 300)

        self.criterion = nn.MSELoss()
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        self.reconstruction_function = nn.MSELoss(size_average=False)


    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = self.reconstruction_function(recon_x, x)  # mse loss
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        # KL divergence
        return BCE + KLD


    def train_(self, embed, args):

        self.optimizer.zero_grad()
        recon_batch, mu, logvar = self.forward(embed, args)

        loss = self.loss_function(recon_batch, embed, mu, logvar)
        loss.backward()
        # train_loss += loss.data[0]
        self.optimizer.step()

        return loss.data[0]

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x, args):
        if not args.no_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar



class SimpleAutoEncoder(nn.Module):
    def __init__(self):
        super(SimpleAutoEncoder, self).__init__()
        self.input_length = 300
        # self.input_length = 2 * OBJ_LENGTH
        self.encoder = nn.Sequential(
            nn.Linear(self.input_length, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64))

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, self.input_length),
            nn.Tanh())

        self.criterion = nn.MSELoss()
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    def train_(self, embed, args):
        self.optimizer.zero_grad()
        output = self.forward(embed, args)
        loss = self.criterion(output, embed)
        loss.backward()
        self.optimizer.step()
        return loss.data[0]

    def forward(self, x, args):
        if not args.no_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)


class RN(BasicModel):
    def __init__(self, args, reg_p):
        super(RN, self).__init__(args, 'RN')
        
        """nn.Linear(in_features, out_features, bias=True)
        in_features – size of each input sample
        out_features – size of each output sample
        bias – If set to False, the layer will not learn an additive bias. Default: True
        """

        # g_fc1 should take 2 * 64 long objects
        self.g_fc1 = nn.Linear(2 * OBJ_LENGTH, HIDDEN_LAYER_UNITS)
        self.g_fc2 = nn.Linear(HIDDEN_LAYER_UNITS, HIDDEN_LAYER_UNITS)
        self.g_fc3 = nn.Linear(HIDDEN_LAYER_UNITS, HIDDEN_LAYER_UNITS)
        self.g_fc4 = nn.Linear(HIDDEN_LAYER_UNITS, HIDDEN_LAYER_UNITS)

        self.f_fc1 = nn.Linear(HIDDEN_LAYER_UNITS, HIDDEN_LAYER_UNITS)

        if not args.BCE:
            self.fcout = FCOutputModel()
        else:
            self.fcout = FCOutputModelBCE()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=reg_p)


    def forward(self, input_feats, args):
        if args.autoencoder:
            first_embedding, second_embedding, third_embedding, post_embedding = \
                input_feats[:, :64], input_feats[:, 64:128], input_feats[:, 128:192], input_feats[:, 192:]
        else:
            first_embedding, second_embedding, third_embedding, post_embedding = self.extract_embeddings(
                input_feats)

        # Save ourselves an argument and make it failsafe
        batch_size = input_feats.shape[0]

        POSSIBLE_PAIRINGS = 6
        
        # Define object-pairs
        # Each embedding is a mbx64 tensor. Concatenating them along the 1-axis yeids mbx128 tensors.
        first_second = torch.cat([first_embedding, second_embedding], dim=1)
        first_third = torch.cat([first_embedding, third_embedding], dim=1)
        first_post = torch.cat([first_embedding, post_embedding], dim=1)
        second_third = torch.cat([second_embedding, third_embedding], dim=1)
        second_post = torch.cat([second_embedding, post_embedding], dim=1)
        third_post = torch.cat([third_embedding, post_embedding], dim=1)

        if not args.no_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # Hold inputs to g
        # g_input should be a  6 * mb * 128 tensor (since there are 6 of mb*128 tensors)
        g_input = torch.empty(POSSIBLE_PAIRINGS, batch_size, 2 *
                              OBJ_LENGTH, dtype=torch.float)
                              
        g_input[0, :, :] = first_second
        g_input[1, :, :] = first_third
        g_input[2, :, :] = first_post
        g_input[3, :, :] = second_third
        g_input[4, :, :] = second_post
        g_input[5, :, :] = third_post

        # now g_input is mb * 6 * 128, by swapping the first two rows
        g_input = g_input.permute(1, 0, 2)

        # now reshape  (mb*6) x 128 =>
        g_input = g_input.contiguous().view(batch_size*POSSIBLE_PAIRINGS, 2*OBJ_LENGTH)

        if args.cuda:
            x_ = g_input.cuda()
        else:
            x_ = g_input

        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)

        # Hold outputs of g
        # g_output will have mb x 6 (possible pairs) x 256 (# nodes in hidden layer)
        g_output = x_.view(batch_size, POSSIBLE_PAIRINGS, HIDDEN_LAYER_UNITS)
        # g_output = torch.empty(mb, POSSIBLE_PAIRINGS, HIDDEN_LAYER_UNITS, dtype=torch.float)

        # Sum output pairings elementwise
        # f_input has size: mb x 256 (since we sum along the possible pairings)
        f_input = g_output.sum(1).squeeze()

        """f"""
        x_f = self.f_fc1(f_input)
        x_f = F.relu(x_f)

        # Each example gets two values (a LOG probability for False, and a LOG probability for True)
        output = self.fcout(x_f)

        return output


    def extract_embeddings(self, input_feats):
        """Extract embeddings from 1227 long input vector"""

        INPUT_FEAT_LENGTH = 1227
        HANDCRAFTED_FEATURES = 263
        numExamples = input_feats.shape[0]

        # input_feats = input_feats.view(NUM_FEATURES)

        input_feats = input_feats[:, HANDCRAFTED_FEATURES:].view(
            numExamples, NUM_FEATURES)  # remove features and flatten
        
        first_embedding = input_feats[:, :300]
        second_embedding = input_feats[:, 300:600]
        third_embedding = input_feats[:, 600:900]
        post_embedding = input_feats[:, 900:]

        # Dimensionality reduction by averaging
        first_embedding = (first_embedding[:, ::3] + first_embedding[:, 1::3] + first_embedding[:, 2::3]) / 3
        second_embedding = (second_embedding[:, ::3] + second_embedding[:, 1::3] + second_embedding[:, 2::3]) / 3
        third_embedding = (third_embedding[:, ::3] + third_embedding[:, 1::3] + third_embedding[:, 2::3]) / 3

        # For now, just take the first 64
        first_embedding = first_embedding[:, :OBJ_LENGTH]
        second_embedding = second_embedding[:, :OBJ_LENGTH]
        third_embedding = third_embedding[:, :OBJ_LENGTH]
        post_embedding = post_embedding[:, :OBJ_LENGTH]

        embeddings = [first_embedding, second_embedding, third_embedding, post_embedding]

        return embeddings
