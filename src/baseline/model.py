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
        self.fc3 = nn.Linear(256, 10)

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
        output = self(input_feats)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy

    def test_(self, input_feats, label):
        output = self(input_feats)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy

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

NUM_FEATURES = 25


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

        ##(number of filters per object+coordinate of object)*2+question vector
        # self.g_fc1 = nn.Linear((NUM_FEATURES + 2) * 2, 256)

        HIDDEN_LAYER_UNITS = 256
        self.g_fc1 = nn.Linear(6, HIDDEN_LAYER_UNITS)
        self.g_fc2 = nn.Linear(HIDDEN_LAYER_UNITS, HIDDEN_LAYER_UNITS)
        self.g_fc3 = nn.Linear(HIDDEN_LAYER_UNITS, HIDDEN_LAYER_UNITS)
        self.g_fc4 = nn.Linear(HIDDEN_LAYER_UNITS, HIDDEN_LAYER_UNITS)

        self.f_fc1 = nn.Linear(HIDDEN_LAYER_UNITS, HIDDEN_LAYER_UNITS)


        # Coordinates for objects i and j
        self.coord_oi = torch.FloatTensor(args.batch_size, 2) # Batch-size:     64
        self.coord_oj = torch.FloatTensor(args.batch_size, 2) # Batch-size:     64

        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        """ from torch.Autograd 
        The Variable API has been deprecated: Variables are no longer necessary to use autograd with tensors. Autograd automatically supports Tensors with requires_grad set to True. 
        Below please find a quick guide on what has changed:
            Variable(tensor) and Variable(tensor, requires_grad) still work as expected, but they return Tensors instead of Variables.
        """
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]

        self.coord_tensor = torch.FloatTensor(args.batch_size, NUM_FEATURES, 2) # Batch-size:     64

        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)

        np_coord_tensor = np.zeros((args.batch_size, NUM_FEATURES, 2))

        for i in range(NUM_FEATURES):
            np_coord_tensor[:, i, :] = np.array(cvt_coord(i))
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, input_feats):
        x = input_feats  #

        # THESE NUMBERS ARE MADE UP!

        USER_FEATS_START = 0
        USER_FEATS_END = 200
        SOURCE_FEATS_END = USER_FEATS_END + 200
        TARGET_FEATS_END = SOURCE_FEATS_END + 200
        # POST_FEATS_END = TARGET_FEATS_END + 364

        REDUCED_DIMS = 64

        xUser = x[:, :USER_FEATS_END]
        xSource = x[:, USER_FEATS_END : SOURCE_FEATS_END]
        xTarget = x[:, SOURCE_FEATS_END : TARGET_FEATS_END]
        xPost = x[:, TARGET_FEATS_END:]

        X = xUser
        k = REDUCED_DIMS

        # X_mean = torch.mean(X, 0)
        # X = X - X_mean.expand_as(X)
        # U, S, V = torch.svd(torch.t(X))
        # C = torch.mm(X, U[:, :k])



        x = torch.cat([xUser[:64], xSource[:64], xTarget[:64], xPost[:64]])





        # We need to convert x into a mb * (64+64+64+64) vector


        """g"""
        # Minibatch size
        mb = x.size()[0]

        # Dimension of the image (does not apply to our code)
        d = x.size()[1]
        # x_flat = (64 x 25 x 24)

        # Goes from (64, 24, 5, 5) to (64, 5^2, 24) shapewise
        # x_flat = x.view(mb, n_channels, d * d).permute(0, 2, 1)

        # For us, let's just keep it as is
        x_flat = x.view(mb,d,1)


        # add coordinates
        # (64, 25, 24) -> (64, 25, 26)
        x_flat = torch.cat([x_flat, self.coord_tensor], 2)

        # x_flat is now: (64, 1227, 1+2)

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x26+11)
        x_i = x_i.repeat(1, NUM_FEATURES, 1, 1)  # (64x25x25x26+11)

        x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x26+11)
        # x_j = torch.cat([x_j, qst], 3)
        x_j = x_j.repeat(1, 1, NUM_FEATURES, 1)  # (64x25x25x26+11)

        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3)  # (64x25x25x2*26+11)

        # reshape for passing through network

        # FC = fully connected layer
        # relu = relu activation function

        # Once we finally get our data into a good format, we can pass it into the first hidden layer.
        x_ = x_full.view(mb * NUM_FEATURES * NUM_FEATURES, 6)
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)

        # reshape again and sum over all the so-called "object pairs"
        x_g = x_.view(mb, NUM_FEATURES * NUM_FEATURES, 256)
        x_g = x_g.sum(1).squeeze()

        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)

        return self.fcout(x_f)


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

