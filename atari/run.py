import sys
import os
import time
from datetime import datetime
sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.join(os.path.dirname(__file__), '../ELF/src_py'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../ELF/build/elf'))

import elf
from rl_old import ActorCritic
from rl.stats import Stats

import _elf as elf_C
import _atari as test
from elf.options import auto_import_options, PyOptionSpec

from model import Model
from copy import deepcopy

import torch
from torch.distributions.categorical import Categorical
import time

class RunGC(object):
    @classmethod
    def get_option_spec(cls):
        spec = PyOptionSpec()
        test.setSpecOptions(spec.getOptionSpec())
        elf_C.setSpecELFOptions(spec.getOptionSpec())
        spec.addIntOption(
            'gpu',
            'GPU id to use',
            0)
        spec.addIntOption(
            'freq_update',
            'How much update before updating the acting model',
            50)
        spec.addStrOption(
            'distri_mode',
            'server or client',
            "")
        spec.addIntOption(
            'num_recv',
            '',
            2)
        spec.addStrListOption(
            'parsed_args',
            'dummy option',
            [])
        spec.merge(PyOptionSpec.fromClasses((ActorCritic,)))
        return spec

    @auto_import_options
    def __init__(self, option_map):
        self.option_map = option_map

    def initialize(self):
        print(self.options)
        options = test.getArgsOptions(self.option_map.getOptionSpec())
        elf_options = elf_C.getArgsELFOptions(self.option_map.getOptionSpec())

        net_opt = elf_C.NetOptions()
        net_opt.port = 5678
        net_options = elf_C.getNetOptions(elf_options, net_opt)

        #import pdb
        #pdb.set_trace()
        if self.options.distri_mode == "client":
            self.remote_client = elf_C.RemoteClients(net_options, ["actor"])
            self.GC = elf_C.BatchReceiver(elf_options, self.remote_client)
            self.GC.setMode(elf_C.RECV_SMEM)

        elif self.options.distri_mode == "server":
            self.remote_server = elf_C.RemoteServers(net_options, ["actor", "train"])
            self.GC = elf_C.BatchSender(elf_options, self.remote_server)
            self.GC.setRemoteLabels(set(["actor"]))

        else:
            self.GC = elf_C.GameContext(elf_options)

        self.game_obj = test.MyContext(options, "actor", "train")
        self.game_obj.setGameFactory(test.getGameFactory(test.AtariOptions()))
        self.game_obj.setGameContext(self.GC)
        self.params = self.game_obj.getParams()
        spec = self.game_obj.getBatchSpec()
        # if you want, you can add spec["train"]["timeout_usec"] = 10 to allow for incomplete batches.

        self.wrapper = elf.GCWrapper(
            self.GC, self.game_obj, int(self.options.batchsize), spec, default_gpu=self.options.gpu, num_recv=self.options.num_recv, histdim=1)

        self.wrapper.reg_callback_if_exists("actor", self.on_batch)
        self.wrapper.reg_callback_if_exists("train", self.on_train)

        if self.options.distri_mode == "client":
            self.model = torch.load(open("model-81.pt", "rb"))
        else:
            self.model = Model(self.params)

        if self.options.gpu >= 0:
            self.model.cuda(self.options.gpu)
        self.train_model = deepcopy(self.model)
        if self.options.gpu >= 0: self.train_model.cuda(self.options.gpu)

        self.method = ActorCritic(self.option_map)
        self.stats = Stats()
        self.optimizer = torch.optim.Adam(self.train_model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-3)

        self.num_update = 0
        self.num_saved = 0
        self.num_iteration = 0
        self.num_eval = 0

    def on_batch(self, batch):
        #print("Receive batch: ", batch.smem.info(),
        #      ", curr_batchsize: ", str(batch.batchsize), sep='')
        #print(batch["s"])
        # print("Actor: " + str(datetime.timestamp(datetime.now())))
        #print("on actor")
        bs = batch["s"].size(0)
        # n = self.params["num_action"]
        self.model.eval()
        res = self.model(batch)

        # Also sampling pi to get actions.
        res["a"] = Categorical(probs=res["pi"].data).sample()

        #if self.num_update > 10:
        #    import pdb
        #    pdb.set_trace()
        self.num_iteration += 1
        self.num_eval += 1
        if self.num_eval >= 5000:
            print(self.game_obj.getSummary())
            self.num_eval = 0

        return dict(a=res["a"], V=res["V"].data, pi=res["pi"].data)

    def on_batch_random(self, batch):
        bs = batch["s"].size(0)
        n = self.params["num_action"]

        m = torch.distributions.categorical.Categorical(torch.FloatTensor(n).fill_(1.0/n))
        a = m.sample(sample_shape=torch.Size([bs]))
        V = torch.FloatTensor(bs).fill_(0)
        pi = torch.FloatTensor(bs, n).fill_(1.0 / n)

        return dict(a=a, V=V, pi=pi)

    def on_train(self, batch):
        # print("Train: " + str(datetime.timestamp(datetime.now())))
        batch.replace_keys("_", "")
        #print("on train")
        #import pdb
        #pdb.set_trace()
        #if self.num_update > 10:
        #    import pdb
        #    pdb.set_trace()

        self.train_model.train()
        self.optimizer.zero_grad()
        self.method.update(self.train_model, batch, self.stats)
        self.optimizer.step()

        self.num_update += 1
        if self.num_update >= self.options.freq_update:
            self.model = deepcopy(self.train_model)
            if self.options.gpu >= 0: self.model.cuda(self.options.gpu)
            self.num_update = 0

        self.num_iteration += 1
        if self.num_iteration >= 5000:
            print("======= Summary %d =======" % self.num_saved)
            print(self.stats.summary())
            self.stats.reset()
            print("")
            torch.save(self.model, open("model-%d.pt" % self.num_saved, "wb"))
            self.num_saved += 1
            print(self.game_obj.getSummary())

            self.num_iteration = 0

if __name__ == '__main__':
    option_spec = PyOptionSpec()
    option_spec.merge(PyOptionSpec.fromClasses((RunGC,)))
    option_map = option_spec.parse()

    rungc = RunGC(option_map)
    rungc.initialize()

    num_batch = 10000000
    start = time.perf_counter()

    rungc.wrapper.start()

    for i in range(num_batch):
        rungc.wrapper.run()

    elapsed = time.perf_counter() - start
    print("Time (s) per batch: %.6f sec" % (elapsed / num_batch))

    print("Stopping................")
    rungc.wrapper.stop()
