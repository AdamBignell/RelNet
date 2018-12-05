import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable



class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x


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

        return F.log_softmax(x)


class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name = name

    def train_(self, input_feats, label):
        self.optimizer.zero_grad()
        output = self.forward(input_feats)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy

    def test_(self, input_feats, label):
        output = self.forward(input_feats)
        pred = output.data.max(1)[1]

        # predictions are log probabilities, so convert back to actual probs
        posClassProbs = torch.exp(output)[:, 1].detach().numpy()

        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, posClassProbs

    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))



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

class RN(BasicModel):
    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')

        # Let's assume we aren't needing a convolutional model
        # self.conv = ConvInputModel()
        
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

        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, input_feats):
        mb = input_feats.shape[0]

        first_embedding, second_embedding, third_embedding, post_embedding = self.extract_embeddings(
            input_feats)

        POSSIBLE_PAIRINGS = 6
        
        # Define object-pairs
        # Each embedding is a mbx64 tensor. Concatenating them along the 1-axis yeids mbx128 tensors.
        first_second = torch.cat([first_embedding, second_embedding], dim=1)
        first_third = torch.cat([first_embedding, third_embedding], dim=1)
        first_post = torch.cat([first_embedding, post_embedding], dim=1)
        second_third = torch.cat([second_embedding, third_embedding], dim=1)
        second_post = torch.cat([second_embedding, post_embedding], dim=1)
        third_post = torch.cat([third_embedding, post_embedding], dim=1)

        # Hold inputs to g
        # g_input should be a  6 * mb * 128 tensor (since there are 6 of mb*128 tensors)
        g_input = torch.empty(POSSIBLE_PAIRINGS, mb, 2 *
                              OBJ_LENGTH, dtype=torch.float)

        g_input[0, :, :] = first_second
        g_input[1, :, :] = first_third
        g_input[2, :, :] = first_post
        g_input[3, :, :] = second_third
        g_input[4, :, :] = second_post
        g_input[5, :, :] = third_post

        # now g_input is mb * 6 * 128, by swapping the first two rows
        g_input = g_input.permute(1, 0, 2)

        # now reshape
        g_input = g_input.contiguous().view(mb*POSSIBLE_PAIRINGS, 2*OBJ_LENGTH)

        # Hold outputs of g
        # g_output will have mb x 6 (possible pairs) x 256 (# nodes in hidden layer)

        x_ = g_input
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)

        g_output = x_.view(mb, POSSIBLE_PAIRINGS, HIDDEN_LAYER_UNITS)
        # g_output = torch.empty(mb, POSSIBLE_PAIRINGS, HIDDEN_LAYER_UNITS, dtype=torch.float)
        
        """g"""
        # for i in range(POSSIBLE_PAIRINGS):
        #     x_ = self.g_fc1(g_input[i,:])
        #     x_ = F.relu(x_)
        #     x_ = self.g_fc2(x_)
        #     x_ = F.relu(x_)
        #     x_ = self.g_fc3(x_)
        #     x_ = F.relu(x_)
        #     x_ = self.g_fc4(x_)
        #     x_ = F.relu(x_)
        #     g_output[i,:] = x_
        
        # Sum output pairings elementwise
        # f_input has size: mb x 256 (since we sum along the possible pairings
        f_input = g_output.sum(1).squeeze()

        """f"""
        x_f = self.f_fc1(f_input)
        x_f = F.relu(x_f)

        # Each example gets two values (a LOG probability for False, and a LOG probability for True)
        output = self.fcout(x_f)

        return output


    def extract_embeddings(self, input_feats):
        """Extract embeddings from 1227 long input vector"""

        # INPUT_FEAT_LENGTH = 1227
        # HANDCRAFTED_FEATURES = 263

        # input_feats = input_feats.view(NUM_FEATURES)

        # input_feats = input_feats[:, HANDCRAFTED_FEATURES:].view(
        #     INPUT_FEAT_LENGTH - HANDCRAFTED_FEATURES)  # remove features and flatten
        
        first_embedding = input_feats[:, :300]
        second_embedding = input_feats[:, 300:600]
        third_embedding = input_feats[:, 600:900]
        post_embedding = input_feats[:, 900:]

        # For now, just take the first 64
        first_embedding = first_embedding[:, :OBJ_LENGTH]
        second_embedding = second_embedding[:, :OBJ_LENGTH]
        third_embedding = third_embedding[:, :OBJ_LENGTH]
        post_embedding = post_embedding[:, :OBJ_LENGTH]

        # Todo: Implement autoencoder or other dim-reduction technique

        return first_embedding, second_embedding, third_embedding, post_embedding



class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')

        self.conv = ConvInputModel()
        self.fc1 = nn.Linear(5 * 5 * 24 + 11, 256)  # question concatenated to all
        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        # print([ a for a in self.parameters() ] )

    def forward(self, img, qst):
        x = self.conv(img)  ## x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1)

        x_ = torch.cat((x, qst), 1)  # Concat question

        x_ = self.fc1(x_)
        x_ = F.relu(x_)

        return self.fcout(x_)

