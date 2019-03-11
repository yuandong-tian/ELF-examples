#pragma once
#include <vector>
#include <string>
#include <functional>

#include "elf/ai/tree_search/mcts.h" 
#include "elf/interface/game_interface.h"
#include "elf/interface/game_base.h"
#include "elf/interface/game_client_interface.h"

namespace game {

// Here we just write a simple game.
// s = 0. 
// a (+1 or -1). 
// s = -6 get negative reward and game over. 
// s = +6 get positive reward and game over.
//
using A = int;

struct S {
 int s;
 void reset() { s = 0; }
 bool terminated() const { return s == -6 || s == 6; }
 bool forward(A a) { 
   if (terminated()) return false;
   s += a;
   return true;
 }
 float terminalValue() const {
   if (! terminated()) return 0;
   else if (s == -6) return -1;
   else return 1;
 }

 friend bool operator==(const S &s1, const S &s2) {
   return s1.s == s2.s;
 }
};

struct StateAction {
  static constexpr int kNumAction = 2;
  static constexpr int kStateDim = 1;

  const S &s;
  float V;
  std::vector<float> pi;

  StateAction(const S &s) : s(s), V(0), pi(kNumAction, 0) {}
  
  void getFeature(float *f) const { *f = s.s; }
  void setValue(const float *pV) { V = *pV; }
  void setPi(const float *ppi) { std::copy(ppi, ppi + pi.size(), pi.begin()); }

  std::string info() const {
    std::stringstream ss;
    ss << "State: " << s.s << ", V: " << V << ", pi: " << pi.size() << "[";
    for (size_t i = 0; i < pi.size(); ++i) {
      if (i > 0) ss << ",";
      ss << pi[i];
    }
    ss << "]";
    return ss.str();
  }
};

struct MCTSActorParams {
  uint64_t seed;
  std::string target;

  std::string info() const { 
    return "Target: " + target;
  }
};

class MCTSActor {
 public:
  using Action = A;
  using State = S;
  using Info = void;
  using NodeResponse = elf::ai::tree_search::NodeResponseT<Action, void>;
  using EdgeInfo = elf::ai::tree_search::EdgeInfo;

  MCTSActor(elf::GameClientInterface* client, std::string target) 
      : client_(client), rng_(time(NULL)) {
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
      *oo_ << "Evaluating state at " << std::hex << &s << std::dec << ", s = " << s.s << std::endl;

    // if terminated(), get results, res = done
    // else res = EVAL_NEED_NN
    if (!s.terminated()) {
      StateAction sa(s);
      if (client_->sendWait(params_.target, sa) == comm::SUCCESS) {
        // std::cout << "Get sa: " << sa.info() << std::endl;
        resp->q_flip = false;
        resp->value = sa.V;
        resp->pi.clear();
        for (size_t i = 0; i < sa.pi.size(); ++i) {
          float prob = sa.pi[i]; 
          A a = (i == 0) ? -1 : 1;
          resp->pi.insert(std::make_pair(a, elf::ai::tree_search::EdgeInfo(prob)));
        }
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

  std::mt19937 *rng() {
    return &rng_;
  }

  float reward(const State& /*s*/, float value) const {
    return value;
  }

 protected:
  elf::GameClientInterface *client_ = nullptr;
  MCTSActorParams params_;
  std::ostream* oo_ = nullptr;
  std::mt19937 rng_;
};

/*
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
*/

class Game {
 public:
  using MCTSAI = elf::ai::tree_search::MCTSAI_T<MCTSActor>;

  Game(int idx, elf::GameClientInterface* client, 
       const elf::ai::tree_search::TSOptions& options, 
       std::string target) : idx_(idx) {
    mcts_.reset(new MCTSAI(options, [=](int) { return new MCTSActor(client, target); }));
  }

  bool step() {
    if (s_.terminated()) return false;
    
    // Call MCTS and find the best move. 
    A a;
    std::cout << "Run MCTS with s: " << s_.s << std::endl;
    mcts_->act(s_, &a);

    const auto &result = mcts_->getLastResult();
    std::cout << "Best action: " << a << ", info: " << result.info() << std::endl;

    s_.forward(a);
    return ! s_.terminated();
  }

  void reset() {
    s_.reset();
  }

 private:
  S s_;
  std::unique_ptr<MCTSAI> mcts_;
  int idx_;
};

} // namespace game.
