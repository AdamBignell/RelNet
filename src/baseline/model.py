import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random


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
        return F.sigmoid(x)


class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name = name

    def naive_guess(self, batch_size, input_feats):
        """Stubbed for subclass to implement"""
        probs = torch.empty(0, 0)
        for i in range(batch_size):
            guess = torch.empty(1, 2)
            probconflict = random.uniform(0, 1)
            probsafe = 1-probconflict
            guess[0, 0] = probsafe
            guess[0, 1] = probconflict
            probs = torch.cat((probs, guess), 0)
        return probs

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

        # with torch.enable_grad(): # Enable gradient descent
        #     loss.backward()
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
            posClassProbs = torch.exp(output)[:, 1].detach().numpy()
        else:
            pred = output.view(output.shape[0]).round()
            correct = pred.eq(label.data.float()).cpu().sum()
            posClassProbs = output[:, 0].detach().numpy()

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

        # # Coordinates for objects i and j
        # self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        # self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        # if args.cuda:
        #     self.coord_oi = self.coord_oi.cuda()
        #     self.coord_oj = self.coord_oj.cuda()
        # self.coord_oi = Variable(self.coord_oi)
        # self.coord_oj = Variable(self.coord_oj)

        # # prepare coord tensor
        # def cvt_coord(i):
        #     return [(i / 5 - 2) / 2., (i % 5 - 2) / 2.]

        # self.coord_tensor = torch.FloatTensor(args.batch_size, NUM_FEATURES, 2)

        # if args.cuda:
        #     self.coord_tensor = self.coord_tensor.cuda()
        # self.coord_tensor = Variable(self.coord_tensor)

        # np_coord_tensor = np.zeros((args.batch_size, NUM_FEATURES, 2))

        # for i in range(NUM_FEATURES):
        #     np_coord_tensor[:, i, :] = np.array(cvt_coord(i))
        # self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))


        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, input_feats, args):
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
        # g_input = torch.empty(POSSIBLE_PAIRINGS * batch_size, 2 *
        #                       OBJ_LENGTH, dtype=torch.float, device=device)
        #
        # g_output = torch.zeros(batch_size, HIDDEN_LAYER_UNITS,
        #                        dtype=torch.float, device=device)

        # g_inputs = []
        # g_inputs.extend([first_second, first_third, first_post, second_third, second_post, third_post])
        #
        # # Just make the indices general to batch sizes
        # batch_indices = []
        # for i in range(batch_size + 1):
        #     batch_indices.append(i * batch_size)
        #
        # g_input[batch_indices[0]:batch_indices[1], :] = first_second
        # g_input[batch_indices[1]:batch_indices[2], :] = first_third
        # g_input[batch_indices[2]:batch_indices[3], :] = first_post
        # g_input[batch_indices[3]:batch_indices[4], :] = second_third
        # g_input[batch_indices[4]:batch_indices[5], :] = second_post
        # g_input[batch_indices[5]:batch_indices[6], :] = third_post
        #
        # """g"""
        # for i in range(POSSIBLE_PAIRINGS):
        #     x_ = self.g_fc1(g_input[batch_indices[i]:batch_indices[i + 1], :].float())
        #     x_ = F.relu(x_)
        #     x_ = self.g_fc2(x_)
        #     x_ = F.relu(x_)
        #     x_ = self.g_fc3(x_)
        #     x_ = F.relu(x_)
        #     x_ = self.g_fc4(x_)
        #     x_ = F.relu(x_)
        #     g_output = g_output + x_  # Sum output pairings element-wise DP style



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

        # now reshape
        g_input = g_input.contiguous().view(batch_size*POSSIBLE_PAIRINGS, 2*OBJ_LENGTH)

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

        g_output = x_.view(batch_size, POSSIBLE_PAIRINGS, HIDDEN_LAYER_UNITS)
        # g_output = torch.empty(mb, POSSIBLE_PAIRINGS, HIDDEN_LAYER_UNITS, dtype=torch.float)

        
        # Sum output pairings elementwise
        # f_input has size: mb x 256 (since we sum along the possible pairings)
        f_input = g_output.sum(1).squeeze()


        # Just for readability
        # f_input = g_output
        # print(f_input.shape)

        # reshape again and sum over all the so-called "object pairs"
        # x_g = x_.view(batch_size, NUM_FEATURES * NUM_FEATURES, 256)
        # x_g = x_g.sum(1).squeeze()

        # """f"""
        # x_f = self.f_fc1(x_g)
        # x_f = F.relu(x_f)


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


        # Todo: Implement autoencoder or other dim-reduction technique

        # For now, just take the first 64
        first_embedding = first_embedding[:, :OBJ_LENGTH]
        second_embedding = second_embedding[:, :OBJ_LENGTH]
        third_embedding = third_embedding[:, :OBJ_LENGTH]
        post_embedding = post_embedding[:, :OBJ_LENGTH]

        return first_embedding, second_embedding, third_embedding, post_embedding