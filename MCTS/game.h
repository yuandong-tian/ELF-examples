#pragma once
#include <vector>
#include <string>
#include <functional>

#include "elf/ai/tree_search/tree_search.h" 

namespace game {

// Here we just write a simple game.
// s = 0. 
// a (+1 or -1). 
// s = -6 get negative reward and game over. 
// s = +6 get positive reward and game over.
//
struct Action {
  int a;
};

struct State {
 int s;
 void reset() { s = 0; }
 bool terminated() const { return s == -6 || s == 6; }
 bool forward(Action a) { 
   if (terminated()) return false;
   s += a;
   return true;
 }
 float terminatedValue() const {
   if (! terminated()) return 0;
   else if (s == -6) return -1;
   else return 1;
 }
};

struct StateAction {
  const State &s;
  float V;
  std::vector<float> pi;

  StateAction(const State &s) : s(s) {}
  
  void getFeature(float *f) { *f = s.s; }
  void setValue(const float *pV) { V = *pV; }
  void setPi(const float *ppi) { std::copy(ppi, ppi + pi.size(), pi.begin()); }
};

class MCTSActorParams {
  uint64_t seed;
  std::string target;

  std::string info() const { 
    return "Target: " + target;
  }
};

class MCTSActor {
 public:
  using Action = Action;
  using State = State;
  using Info = void;
  using NodeResponse = elf::ai::tree_search::NodeResponseT<Action, void>;
  using EdgeInfo = elf::ai::tree_search::EdgeInfo;

  MCTSActor(elf::GameClientInterface* client, std::string target) 
      : client_(client) {
     params_.target = target;
  }

  std::string info() const {
    return params_.info();
  }

  void set_ostream(std::ostream* oo) {
    oo_ = oo;
  }

  // batch evaluate. Disable it first for simplicity.
  bool evaluate(
      const std::vector<const State*>& states,
      std::function<void (size_t, NodeResponse &&)> callback) {
    (void)states;
    (void)callback;
    return false;
  }

  void evaluate(const State& s, NodeResponse* resp) {
    if (oo_ != nullptr)
      *oo_ << "Evaluating state at " << std::hex << &s << std::dec << std::endl;

    // if terminated(), get results, res = done
    // else res = EVAL_NEED_NN
    if (!s.terminated()) {
      StateAction sa(s);
      if (client_->sendWait(params_.target, &sa) == SUCCESS) {
        resp->q_flip = false;
        resp->value = sa.value;
        resp->pi = sa.pi;
      }
    } else {
      resp->value = s.terminalValue();
      resp->pi.clear();
    }

    if (oo_ != nullptr)
      *oo_ << "Finish evaluating state at " << std::hex << &s << std::dec
           << std::endl;
  }

  bool forward(State& s, Action a) {
    return s.forward(a);
  }

  void setID(int) {
  }

  float reward(const State& /*s*/, float value) const {
    return value;
  }

 protected:
  elf::GameClientInterface *cient_ = nullptr;
  MCTSActorParams params_;
  std::ostream* oo_ = nullptr;
};

namespace elf {
namespace ai {
namespace tree_search {

template <>
struct ActorTrait<MCTSActor> {
 public:
  static std::string to_string(const MCTSActor& a) {
    return a.info();
  }
};

} // namespace tree_search
} // namespace ai
} // namespace elf

class Game {
 public:
  using MCTSAI = elf::ai::tree_search::MCTSAI_T<MCTSActor>;

  Game(int idx, elf::GameClientInterface* client, std::string target) : idx_(idx) {
    mcts_.reset(new MCTSAI([=](int) { return new MCTSActor(client, target); }));
  }

  bool step() {
    if (s_.terminated()) return false;
    
    // Call MCTS and find the best move. 
    mcts_->setState(s_);
    mcts_->run();
    const auto &result = mcts_->getLastResult();
    Action a = result.best_edge_info.a;
    s_.forward(a);
    return ! s_.terminated();
  }

  void reset() {
    s_.reset();
  }

 private:
  State s_;
  std::unique_ptr<MCTSAI> mcts_;
  int idx_;
};

} // namespace game.
