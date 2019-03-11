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

        self.h = 10
        self.trunk = nn.Linear(params["dim"], self.h)
        self.relu = nn.ReLU()

        self.policy_branch = nn.Linear(self.h, params["num_action"])
        self.value_branch = nn.Linear(self.h, 1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, batch):
        # Get the last hist_len frames.
        s = Variable(batch["s"])
        # print("input size = " + str(s.size()))
        rep = self.trunk(s)
        # print("trunk size = " + str(rep.size()))
        logpi = self.logsoftmax(self.policy_branch(rep))
        value = nn.Tanh()(self.value_branch(rep))
        return dict(pi=logpi.exp(), logpi=logpi, V=value)
