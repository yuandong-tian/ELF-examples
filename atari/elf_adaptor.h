#pragma once

#include <string>
#include <thread>
#include <vector>

#include "atari_game.h"
#include "elf/interface/snippets.h"

using elf::snippet::Reply;

class GameInterface : public elf::snippet::Interface {
 public:
   GameInterface(const atari::Options &options)
     : options_(options), example_(0, false, options) {
      ale::Logger::setMode(ale::Logger::mode::Error);
   }

   int dim() const override { return example_.obsDim(); }
   std::vector<int> dims() const override { return {example_.channel(), example_.height(), example_.width()}; }
   int numActions() const override { return example_.numActions(); }

   std::unordered_map<std::string, int> getParams() const override {
     return std::unordered_map<std::string, int>{
       { "num_action", example_.numActions() },
         { "width", example_.width() },
         { "height", example_.height() },
         { "channel", example_.channel() }
     };
   }

   elf::snippet::Game *createGame(int idx, bool eval_mode) const override;

 private:
   const atari::Options options_;
   atari::Game example_;
};

class GameWrapper : public elf::snippet::Game {
 public:
  GameWrapper(int idx, bool eval_mode, const atari::Options &options)
    : game_(idx, eval_mode, options) {
  }

  std::vector<float> feature() const override {
    return game_.getObs();
  }

  void reset() override { game_.reset(); }

  // return false if the game has come to an end, and change the reply.
  bool step(Reply *reply) override {
    bool game_continue = game_.step(reply->a);
    reply->r = game_.getLastReward();
    return game_continue;
  }

 private:
  atari::Game game_;
};

inline elf::snippet::Game *GameInterface::createGame(int idx, bool eval_mode) const {
  return new GameWrapper(idx, eval_mode, options_);
}
