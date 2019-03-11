#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <map>

#include "game.h"
#include "elf/interface/game_interface.h"

using SpecItem = std::unordered_map<std::string, std::vector<std::string>>;
using Spec = std::unordered_map<std::string, SpecItem>;
using elf::Extractor;
using game::StateAction;

struct FeatureReg {
  static Extractor reg(int batchsize) {
     Extractor e;
     e.addField<float>("s")
       .addExtents(batchsize, {batchsize, StateAction::kStateDim})
       .addFunction<StateAction>(&StateAction::getFeature);

     e.addField<float>("V")
       .addExtents(batchsize, {batchsize})
       .addFunction<StateAction>(&StateAction::setValue);

     e.addField<float>("pi")
       .addExtents(batchsize, {batchsize, StateAction::kNumAction})
       .addFunction<StateAction>(&StateAction::setPi);

     return e;
  }
};

class MyContext {
 public:
   MyContext(const elf::ai::tree_search::TSOptions &options, std::string eval_name)
     : options_(options), eval_name_(eval_name) {
     }

   void setGameContext(elf::GCInterface* ctx) {
     int num_games = ctx->options().num_game_thread;

     using std::placeholders::_1;
     for (int i = 0; i < num_games; ++i) {
       auto* g = ctx->getGame(i);
       if (g != nullptr) {
         games_.emplace_back(new game::Game(i, ctx->getClient(), options_, eval_name_));
         games_[i]->reset();
         g->setCallbacks(std::bind(gameMainLoop, games_[i].get(), _1));
       }
     }

     regFunc(ctx);
   }

   std::unordered_map<std::string, int> getParams() const {
     std::unordered_map<std::string, int> params; 
     params["num_action"] = StateAction::kNumAction;
     params["dim"] = StateAction::kStateDim; 
     return params;
   }

   Spec getBatchSpec() const { return spec_; }

 private:
  elf::ai::tree_search::TSOptions options_;
  std::string eval_name_;
  Spec spec_;

  std::vector<std::unique_ptr<game::Game>> games_;

  static void gameMainLoop(game::Game *g, elf::game::Base *) {
    if (! g->step()) {
      g->reset();
    }
  }

  static SpecItem getSpec(const Extractor &e) {
    return SpecItem{
      { "input", e.getState2MemNames() },
      { "reply", e.getMem2StateNames() },
    };
  }

  void regFunc(elf::GCInterface *ctx) {
    Extractor& e = ctx->getExtractor();
    int batchsize = ctx->options().batchsize;

    Extractor e_actor = FeatureReg::reg(batchsize);
    spec_[eval_name_] = getSpec(e_actor);
    e.merge(std::move(e_actor));
  }
};
