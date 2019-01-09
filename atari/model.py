# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from collections import Counter
from torch.autograd import Variable

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()

        self.linear_dim = 1920
        relu_func = lambda : nn.LeakyReLU(0.1)
        # relu_func = nn.ReLU

        self.trunk = nn.Sequential(
            nn.Conv2d(params["channel"] * params["frame_stack"], 32, 5, padding = 2),
            relu_func(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 5, padding = 2),
            relu_func(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding = 1),
            relu_func(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding = 1),
            relu_func(),
            nn.MaxPool2d(2, 2),
        )

        self.conv2fc = nn.Sequential(
            nn.Linear(self.linear_dim, 512),
            nn.PReLU()
        )

        self.policy_branch = nn.Linear(512, params["num_action"])
        self.value_branch = nn.Linear(512, 1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, batch):
        # Get the last hist_len frames.
        s = Variable(batch["s"])
        # print("input size = " + str(s.size()))
        rep = self.trunk(s)
        # print("trunk size = " + str(rep.size()))
        rep = self.conv2fc(rep.view(-1, self.linear_dim))
        logpi = self.logsoftmax(self.policy_branch(rep))
        value = self.value_branch(rep)
        return dict(pi=logpi.exp(), logpi=logpi, V=value)
