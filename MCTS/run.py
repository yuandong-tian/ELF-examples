import sys
import os
import time
from datetime import datetime
sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.join(os.path.dirname(__file__), '../ELF/src_py'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../ELF/build/elf'))

import elf

import _elf as elf_C
import _mcts_demo as test
from elf.options import auto_import_options, PyOptionSpec

from model import Model
from copy import deepcopy

import torch
import time

class RunGC(object):
    @classmethod
    def get_option_spec(cls):
        spec = PyOptionSpec()
        elf_C.setSpecELFOptions(spec.getOptionSpec())
        test.setSpecTSOptions(spec.getOptionSpec())
        spec.addIntOption(
            'gpu',
            'GPU id to use',
            0)
        spec.addStrOption(
            'load',
            'Load old model',
            "")
        spec.addStrListOption(
            'parsed_args',
            'dummy option',
            [])
        return spec

    @auto_import_options
    def __init__(self, option_map):
        self.option_map = option_map

    def initialize(self):
        elf_options = elf_C.getArgsELFOptions(self.option_map.getOptionSpec())
        ts_options = test.getArgsTSOptions(self.option_map.getOptionSpec())
        self.GC = elf_C.GameContext(elf_options)

        self.game_obj = test.MyContext(ts_options, "actor")
        self.game_obj.setGameContext(self.GC)
        self.params = self.game_obj.getParams()
        spec = self.game_obj.getBatchSpec()
        # Important for MCTS. Otherwise it will hang.
        spec["actor"]["timeout_usec"] = 10

        self.wrapper = elf.GCWrapper(
            self.GC, self.game_obj, int(self.options.batchsize), 
            spec, default_gpu=self.options.gpu, num_recv=1)

        self.wrapper.reg_callback_if_exists("actor", self.on_batch)

        if self.options.load != "":
            self.model = torch.load(self.options.load)
        else:
            self.model = Model(self.params)
            torch.save(self.model, "model.bin")

        for s in range(-5, 6):
            ss = torch.FloatTensor(1, 1)
            ss[0, 0] = s
            res = self.model(dict(s=ss))
            print("s: %f, V: %f, pi: %s" % (s, res["V"].data.item(), str(res["pi"].data)))

        if self.options.gpu >= 0:
            self.model.cuda(self.options.gpu)

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
        # n = self.params["num_action"]
        # print(batch.batchsize)
        res = self.model(batch)
        res["V"].data /= 10.0
        #print("s: %f, V: %f, pi: %s" % (batch["s"].data.item(), res["V"].data.item(), str(res["pi"].data)))
        return dict(V=res["V"].data, pi=res["pi"].data)

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
