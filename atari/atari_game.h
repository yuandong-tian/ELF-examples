/**
* Copyright (c) 2017-present, Facebook, Inc.
* All rights reserved.

* This source code is licensed under the BSD-style license found in the
* LICENSE file in the root directory of this source tree.
*/

//File: atari_game.h

#pragma once

#include "ale_interface.hpp"

#include <vector>
#include <string>
#include <random>

namespace atari {

static constexpr int kWidth = 160;
static constexpr int kHeight = 210;
static constexpr int kRatio = 2;
static constexpr int kBufSize = kWidth*kHeight*3;
static constexpr int kInputStride = kBufSize / kRatio / kRatio;
static constexpr int kWidthRatio = kWidth / kRatio;
static constexpr int kHeightRatio = kHeight / kRatio;

struct Options {
  std::string rom_file = "breakout.bin";
  int frame_skip = 1;
  float repeat_action_probability = 0.;
  int seed = 0;
};

class Game {
 public:
  Game(int game_idx, bool eval_mode, const Options&);

  int numActions() const { return _action_set.size(); }
  const std::vector<Action>& actionSet() const { return _action_set; }
  int width() const { return _width; }
  int height() const { return _height; }
  int channel() const { return 3; }
  int obsDim() const { return _width * _height * 3; }

  void reset();
  bool step(int act);

  // T * height * width
  std::vector<float> getObs() const;
  float getLastReward() const;
  int getTick() const;
  int getLives() const;

 private:
  int _game_idx = -1;
  bool _eval_mode = false;
  float _reward_clip;

  std::mt19937 _rng;
  std::unique_ptr<ALEInterface> _ale;

  int _width, _height;
  std::vector<Action> _action_set;

  float _last_reward = 0;

  static const int kMaxRep = 30;
  int _last_act_count = 0;
  int _last_act = -1;

  std::unique_ptr<std::uniform_int_distribution<>> _distr_action;
  std::uniform_int_distribution<int> _distr_start_loc;
  std::uniform_int_distribution<int> _distr_frame_skip;

  int _prevent_stuck(int act);
  void _reset_stuck_state();
};

}  // namespace atari
