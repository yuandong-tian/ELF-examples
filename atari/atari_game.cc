/**
* Copyright (c) 2017-present, Facebook, Inc.
* All rights reserved.

* This source code is licensed under the BSD-style license found in the
* LICENSE file in the root directory of this source tree.
*/

//File: atari_game.cc

#include "atari_game.h"
#include <mutex>
#include <cmath>
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

namespace {
// work around bug in ALE.
// see Arcade-Learning-Environment/issues/86
mutex ALE_GLOBAL_LOCK;
}

namespace atari {

static long compute_seed(int v) {
  auto now = system_clock::now();
  auto now_ms = time_point_cast<milliseconds>(now);
  auto value = now_ms.time_since_epoch();
  long duration = value.count();
  long seed = (time(NULL) * 1000 + duration) % 100000000 + v;
  return seed;
}

static void _downsample(const std::vector<unsigned char> &buf, float *output) {
  for (int k = 0; k < 3; ++k) {
    const unsigned char *raw_img = &buf[k];
    for (int j = 0; j < kHeight / kRatio; ++j) {
      for (int i = 0; i < kWidth / kRatio; ++i) {
        float v = 0.0;
        int ii = kRatio * i;
        int jj = kRatio * j;
        /*
           for (int lx = 0; lx < ratio; ++lx) {
           for (int ly = 0; ly < ratio; ++ly) {
           v += *(raw_img + ((jj + ly) * width + (ii + lx)) * 3);
           }
           }
           v /= (ratio * ratio);
           */
        v = *(raw_img + (jj * kWidth + ii) * 3);
        *output ++ = v / 255.0;
      }
    }
  }
}

Game::Game(int game_idx, bool eval_mode, const Options& opt)
  : _game_idx(game_idx), _eval_mode(eval_mode),
    _rng(compute_seed(game_idx)), _distr_start_loc(0, 0), _distr_frame_skip(2, 4) {
  lock_guard<mutex> lg(ALE_GLOBAL_LOCK);
  _ale.reset(new ALEInterface);
  long seed = compute_seed(opt.seed);
  // std::cout << "Seed: " << seed << std::endl;
  _ale->setInt("random_seed", seed);
  _ale->setBool("showinfo", false);
  // _ale->setInt("frame_skip", opt.frame_skip);
  _ale->setBool("color_averaging", false);
  _ale->setFloat("repeat_action_probability", opt.repeat_action_probability);
  _ale->loadROM(opt.rom_file);

  auto& s = _ale->getScreen();
  _width = s.width() / kRatio, _height = s.height() / kRatio;
  _action_set = _ale->getMinimalActionSet();
  _distr_action.reset(new std::uniform_int_distribution<>(0, _action_set.size() - 1));
}

void Game::reset() {
  _reset_stuck_state();

  _ale->reset_game();
  _last_reward = 0;
  int start_loc = _distr_start_loc(_rng);
  // Random start.
  for (int i = 0; i < start_loc; ++i) {
    int act = (*_distr_action)(_rng);
    step(act);
  }
}

std::vector<float> Game::getObs() const {
  // Downsample the image.
  std::vector<unsigned char> buf;
  buf.resize(kBufSize, 0);
  _ale->getScreenRGB(buf);
  
  // channel * height * width
  std::vector<float> obs(kInputStride);
  _downsample(buf, &obs[0]);

  return obs;
}

float Game::getLastReward() const {
  return _last_reward;
}

int Game::getTick() const {
  return _ale->getEpisodeFrameNumber();
}

int Game::getLives() const {
  return _ale->lives();
}

bool Game::step(int act) {
  // std::cout << "[" << _game_idx << "][" << gs.seq.game_counter << "][" << gs.seq.seq << "] act: "
  //          << act << "[a=" << gs.reply.action << "][V=" << gs.reply.value << "]" << std::endl;
  if (act < 0 || act >= (int)_action_set.size() || _ale->game_over()) {
    // Illegal action.
    return false;
  }

  // TODO: We should not distinguish between training and test.
  if (_eval_mode) {
    act = _prevent_stuck(act);
  }
  int frame_skip = _distr_frame_skip(_rng);
  _last_reward = 0;
  for (int j = 0; j < frame_skip; ++j) {
    _last_reward += _ale->act(_action_set.at(act));
  }
  // hack 
  // if (act == 1)
  //  _last_reward += 0.1; 
  return true;
}

int Game::_prevent_stuck(int act) {
  if (act == _last_act) {
    _last_act_count ++;
    if (_last_act_count >= kMaxRep) {
      // The player might get stuck. Save it.
      act = (*_distr_action)(_rng);
    }
  } else {
    // Reset counter.
    _last_act = act;
    _last_act_count = 0;
  }
  return act;
}

void Game::_reset_stuck_state() {
  _last_act_count = 0;
  _last_act = -1;
}

}  // namespace atari
