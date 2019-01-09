#pragma once

#include <string>
#include <thread>
#include <vector>

#include "atari_game.h"
#include "elf/interface/snippets.h"

using elf::snippet::GameInterface;
using elf::snippet::GameFactory;
using elf::snippet::Reply;

class GameWrapper : public GameInterface {
 public:
  GameWrapper(int idx, bool eval_mode, const atari::Options &options)
    : game_(idx, eval_mode, options) {
      if (idx == 0) {
        ale::Logger::setMode(ale::Logger::mode::Error);
      }
  }

  std::vector<float> feature() const override {
    return game_.getObs();
  }

  int dim() const override { return game_.obsDim(); }
  std::vector<int> dims() const override { return {game_.channel(), game_.height(), game_.width()}; }
  int numActions() const override { return game_.numActions(); }

  void reset() override { game_.reset(); }

  std::unordered_map<std::string, int> getParams() const override {
     return std::unordered_map<std::string, int>{
       { "num_action", game_.numActions() },
         { "width", game_.width() },
         { "height", game_.height() },
         { "channel", game_.channel() }
     };
  }

  // return false if the game has come to an end, and change the reply.
  bool step(Reply *reply) override {
    bool game_continue = game_.step(reply->a);
    reply->r = game_.getLastReward();
    return game_continue;
  }

  static GameFactory getGameFactory(const atari::Options &options) {
    return GameFactory([options](int idx, bool eval_mode) {
      return new GameWrapper(idx, eval_mode, options);
    });
  }

 private:
  atari::Game game_;
};

