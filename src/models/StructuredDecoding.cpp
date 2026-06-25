#include "tesseract/models/StructuredDecoding.hpp"

#include <algorithm>
#include <cctype>
#include <functional>
#include <limits>

#include "tesseract/utils/Logging.hpp"

namespace tesseract::models {

// ===========================================================================
// Regex → AST  (recursive-descent: alt → concat → repeat → atom)
// ===========================================================================
namespace {

struct Node {
  enum Kind { Bytes, Concat, Alt, Repeat } kind;
  std::array<bool, 256> set{};   // Bytes: accepted byte set
  std::vector<Node> kids;        // Concat/Alt children; Repeat: kids[0]
  int rmin = 0;                  // Repeat
  int rmax = -1;                 // Repeat (-1 ⇒ unbounded)
};

class Parser {
 public:
  explicit Parser(const std::string& s) : s_(s) {}

  Node parse() {
    Node n = parse_alt();
    TESSERACT_CHECK(pos_ == s_.size(),
                    "RegexAutomaton: unexpected trailing input at offset {}",
                    pos_);
    return n;
  }

 private:
  const std::string& s_;
  std::size_t pos_ = 0;

  bool eof() const { return pos_ >= s_.size(); }
  char peek() const { return s_[pos_]; }
  char get() { return s_[pos_++]; }

  static Node make_bytes(std::array<bool, 256> set) {
    Node n;
    n.kind = Node::Bytes;
    n.set = set;
    return n;
  }
  static Node single(unsigned char c) {
    std::array<bool, 256> set{};
    set[c] = true;
    return make_bytes(set);
  }

  Node parse_alt() {
    std::vector<Node> branches;
    branches.push_back(parse_concat());
    while (!eof() && peek() == '|') {
      get();
      branches.push_back(parse_concat());
    }
    if (branches.size() == 1) return std::move(branches[0]);
    Node n;
    n.kind = Node::Alt;
    n.kids = std::move(branches);
    return n;
  }

  Node parse_concat() {
    std::vector<Node> parts;
    while (!eof() && peek() != '|' && peek() != ')') {
      parts.push_back(parse_repeat());
    }
    Node n;
    n.kind = Node::Concat;        // empty kids ⇒ matches the empty string
    n.kids = std::move(parts);
    return n;
  }

  Node parse_repeat() {
    Node atom = parse_atom();
    if (eof()) return atom;
    const char c = peek();
    if (c == '*' || c == '+' || c == '?') {
      get();
      Node n;
      n.kind = Node::Repeat;
      n.rmin = (c == '+') ? 1 : 0;
      n.rmax = (c == '?') ? 1 : -1;
      n.kids.push_back(std::move(atom));
      return n;
    }
    if (c == '{') return parse_counted(std::move(atom));
    return atom;
  }

  Node parse_counted(Node atom) {
    get();  // '{'
    int m = 0;
    bool any = false;
    while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
      m = m * 10 + (get() - '0');
      any = true;
    }
    TESSERACT_CHECK(any, "RegexAutomaton: brace quantifier needs a count");
    int n = m;
    bool unbounded = false;
    if (!eof() && peek() == ',') {
      get();
      if (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
        n = 0;
        while (!eof() && std::isdigit(static_cast<unsigned char>(peek())))
          n = n * 10 + (get() - '0');
      } else {
        unbounded = true;
      }
    }
    const bool closed_brace = !eof() && peek() == '}';
    TESSERACT_CHECK(closed_brace,
                    "RegexAutomaton: unterminated brace quantifier");
    get();  // '}'
    Node rep;
    rep.kind = Node::Repeat;
    rep.rmin = m;
    rep.rmax = unbounded ? -1 : n;
    rep.kids.push_back(std::move(atom));
    return rep;
  }

  Node parse_atom() {
    const char c = peek();
    if (c == '(') {
      get();
      Node inner = parse_alt();
      TESSERACT_CHECK(!eof() && peek() == ')', "RegexAutomaton: missing ')'");
      get();
      return inner;
    }
    if (c == '[') return parse_class();
    if (c == '.') {
      get();
      std::array<bool, 256> set{};
      set.fill(true);
      return make_bytes(set);
    }
    if (c == '\\') {
      get();
      return parse_escape();
    }
    const bool is_meta = (c == '*' || c == '+' || c == '?' || c == '{' ||
                          c == '|' || c == ')');
    TESSERACT_CHECK(!is_meta, "RegexAutomaton: stray metacharacter '{}'",
                    std::string(1, c));
    return single(static_cast<unsigned char>(get()));
  }

  static void flip(std::array<bool, 256>& set) {
    for (auto& b : set) b = !b;
  }

  static bool escape_class(char e, std::array<bool, 256>& set) {
    auto add = [&](int a, int b) { for (int i = a; i <= b; ++i) set[i] = true; };
    switch (e) {
      case 'd': add('0', '9'); return true;
      case 'D': add('0', '9'); flip(set); return true;
      case 'w': add('a', 'z'); add('A', 'Z'); add('0', '9'); set['_'] = true;
                return true;
      case 'W': add('a', 'z'); add('A', 'Z'); add('0', '9'); set['_'] = true;
                flip(set); return true;
      case 's': set[' '] = set['\t'] = set['\n'] = set['\r'] = set['\f'] =
                set['\v'] = true; return true;
      case 'S': set[' '] = set['\t'] = set['\n'] = set['\r'] = set['\f'] =
                set['\v'] = true; flip(set); return true;
      default:  return false;
    }
  }

  static unsigned char escape_literal(char e) {
    switch (e) {
      case 'n': return '\n';
      case 't': return '\t';
      case 'r': return '\r';
      case 'f': return '\f';
      case 'v': return '\v';
      case '0': return '\0';
      default:  return static_cast<unsigned char>(e);  // \. \\ \( \[ etc.
    }
  }

  Node parse_escape() {
    TESSERACT_CHECK(!eof(), "RegexAutomaton: trailing backslash");
    const char e = get();
    std::array<bool, 256> set{};
    if (escape_class(e, set)) return make_bytes(set);
    return single(escape_literal(e));
  }

  Node parse_class() {
    get();  // '['
    bool negate = false;
    if (!eof() && peek() == '^') { negate = true; get(); }
    std::array<bool, 256> set{};
    bool first = true;
    while (!eof() && (peek() != ']' || first)) {
      first = false;
      int lo;
      if (peek() == '\\') {
        get();
        TESSERACT_CHECK(!eof(), "RegexAutomaton: trailing backslash in class");
        const char e = get();
        std::array<bool, 256> cls{};
        if (escape_class(e, cls)) {
          for (int i = 0; i < 256; ++i) set[i] = set[i] || cls[i];
          continue;  // class escapes never form ranges
        }
        lo = escape_literal(e);
      } else {
        lo = static_cast<unsigned char>(get());
      }
      if (!eof() && peek() == '-' && pos_ + 1 < s_.size() &&
          s_[pos_ + 1] != ']') {
        get();  // '-'
        int hi;
        if (peek() == '\\') { get(); hi = escape_literal(get()); }
        else { hi = static_cast<unsigned char>(get()); }
        TESSERACT_CHECK(lo <= hi, "RegexAutomaton: reversed class range");
        for (int i = lo; i <= hi; ++i) set[i] = true;
      } else {
        set[lo] = true;
      }
    }
    TESSERACT_CHECK(!eof() && peek() == ']', "RegexAutomaton: unterminated '['");
    get();  // ']'
    if (negate) flip(set);
    return make_bytes(set);
  }
};

}  // namespace

// ===========================================================================
// AST → Thompson NFA → lazy subset DFA
// ===========================================================================

RegexAutomaton RegexAutomaton::compile(const std::string& pattern) {
  const Node ast = Parser(pattern).parse();
  RegexAutomaton a;
  auto& nfa = a.nfa_;

  auto new_state = [&]() -> int {
    nfa.emplace_back();
    return static_cast<int>(nfa.size()) - 1;
  };

  struct Frag { int start; int accept; };
  std::function<Frag(const Node&)> build = [&](const Node& n) -> Frag {
    switch (n.kind) {
      case Node::Bytes: {
        const int s = new_state();
        const int t = new_state();
        nfa[s].has_byte = true;
        nfa[s].set = n.set;
        nfa[s].out = t;
        return {s, t};
      }
      case Node::Concat: {
        if (n.kids.empty()) {           // empty string
          const int s = new_state();
          return {s, s};
        }
        Frag first = build(n.kids[0]);
        int acc = first.accept;
        for (std::size_t i = 1; i < n.kids.size(); ++i) {
          Frag f = build(n.kids[i]);
          nfa[acc].eps.push_back(f.start);
          acc = f.accept;
        }
        return {first.start, acc};
      }
      case Node::Alt: {
        const int s = new_state();
        const int t = new_state();
        for (const Node& k : n.kids) {
          Frag f = build(k);
          nfa[s].eps.push_back(f.start);
          nfa[f.accept].eps.push_back(t);
        }
        return {s, t};
      }
      case Node::Repeat: {
        const int s = new_state();
        int cur = s;
        for (int i = 0; i < n.rmin; ++i) {     // required copies
          Frag f = build(n.kids[0]);
          nfa[cur].eps.push_back(f.start);
          cur = f.accept;
        }
        if (n.rmax < 0) {                       // {rmin,} → trailing star
          Frag f = build(n.kids[0]);
          const int loop = new_state();
          const int out = new_state();
          nfa[cur].eps.push_back(loop);
          nfa[loop].eps.push_back(f.start);
          nfa[loop].eps.push_back(out);
          nfa[f.accept].eps.push_back(loop);
          return {s, out};
        }
        const int end = new_state();            // {rmin,rmax} → optional tail
        for (int i = n.rmin; i < n.rmax; ++i) {
          nfa[cur].eps.push_back(end);          // may stop here
          Frag f = build(n.kids[0]);
          nfa[cur].eps.push_back(f.start);
          cur = f.accept;
        }
        nfa[cur].eps.push_back(end);
        return {s, end};
      }
    }
    const int s = new_state();                  // unreachable
    return {s, s};
  };

  const Frag whole = build(ast);
  a.nfa_start_ = whole.start;
  a.nfa_accept_ = whole.accept;
  // Register the start set as DFA id 0 eagerly so start() is trivial.
  a.intern(a.closure({a.nfa_start_}));
  return a;
}

std::vector<int> RegexAutomaton::closure(std::vector<int> set) const {
  std::vector<char> seen(nfa_.size(), 0);
  std::vector<int> stack;
  for (int s : set) {
    if (s >= 0 && !seen[s]) { seen[s] = 1; stack.push_back(s); }
  }
  while (!stack.empty()) {
    const int q = stack.back();
    stack.pop_back();
    for (int e : nfa_[q].eps) {
      if (!seen[e]) { seen[e] = 1; stack.push_back(e); }
    }
  }
  std::vector<int> out;
  for (int i = 0; i < static_cast<int>(nfa_.size()); ++i)
    if (seen[i]) out.push_back(i);  // ascending ⇒ canonical key
  return out;
}

int64_t RegexAutomaton::intern(std::vector<int> closed) const {
  auto it = set_to_id_.find(closed);
  if (it != set_to_id_.end()) return it->second;
  const int64_t id = static_cast<int64_t>(dfa_sets_.size());
  const bool acc =
      std::find(closed.begin(), closed.end(), nfa_accept_) != closed.end();
  set_to_id_.emplace(closed, id);
  dfa_sets_.push_back(std::move(closed));
  trans_.emplace_back();
  trans_.back().fill(-2);
  accepts_.push_back(acc ? 1 : 0);
  return id;
}

int64_t RegexAutomaton::start() const {
  return dfa_sets_.empty() ? kDead : 0;
}

int64_t RegexAutomaton::step(int64_t state, std::uint8_t byte) const {
  if (state < 0) return kDead;
  const int64_t cached = trans_[state][byte];
  if (cached != -2) return cached;
  std::vector<int> next;
  for (int q : dfa_sets_[state]) {
    const NfaState& st = nfa_[q];
    if (st.has_byte && st.set[byte]) next.push_back(st.out);
  }
  const int64_t result = next.empty() ? kDead : intern(closure(std::move(next)));
  trans_[state][byte] = result;  // re-index: intern may have grown trans_
  return result;
}

bool RegexAutomaton::accepting(int64_t state) const {
  return state >= 0 && state < static_cast<int64_t>(accepts_.size()) &&
         accepts_[state] != 0;
}

bool RegexAutomaton::matches(std::string_view s) const {
  int64_t st = start();
  for (unsigned char c : s) {
    st = step(st, c);
    if (st == kDead) return false;
  }
  return accepting(st);
}

// ===========================================================================
// GrammarConstraint
// ===========================================================================

GrammarConstraint::GrammarConstraint(const ByteAutomaton& automaton,
                                     std::vector<std::string> token_bytes,
                                     int32_t eos_id)
    : a_(automaton),
      token_bytes_(std::move(token_bytes)),
      eos_id_(eos_id),
      state_(automaton.start()) {}

GrammarConstraint GrammarConstraint::from_tokenizer(
    const ByteAutomaton& automaton, const io::Tokenizer& tk) {
  const int64_t V = static_cast<int64_t>(tk.vocab_size());
  std::vector<std::string> token_bytes(static_cast<std::size_t>(V));
  for (int64_t i = 0; i < V; ++i) {
    const int32_t id = static_cast<int32_t>(i);
    token_bytes[static_cast<std::size_t>(i)] =
        tk.decode(std::span<const int32_t>(&id, 1), /*skip_special=*/false);
  }
  return GrammarConstraint(automaton, std::move(token_bytes),
                           tk.eos_token_id());
}

void GrammarConstraint::reset() { state_ = a_.start(); }

int64_t GrammarConstraint::step_bytes(int64_t state,
                                      const std::string& bytes) const {
  for (unsigned char c : bytes) {
    state = a_.step(state, c);
    if (state == ByteAutomaton::kDead) return state;
  }
  return state;
}

int64_t GrammarConstraint::apply(std::span<float> logits) const {
  const float ninf = -std::numeric_limits<float>::infinity();
  const int64_t V = static_cast<int64_t>(logits.size());
  const int64_t TB = static_cast<int64_t>(token_bytes_.size());
  int64_t allowed = 0;
  for (int64_t id = 0; id < V; ++id) {
    bool ok;
    if (id == eos_id_) {
      ok = a_.accepting(state_);
    } else if (id < TB) {
      const std::string& b = token_bytes_[static_cast<std::size_t>(id)];
      ok = !b.empty() && step_bytes(state_, b) != ByteAutomaton::kDead;
    } else {
      ok = false;
    }
    if (ok) ++allowed;
    else logits[static_cast<std::size_t>(id)] = ninf;
  }
  TESSERACT_CHECK(allowed > 0,
                  "GrammarConstraint: no legal next token from state {} "
                  "(grammar dead-end / max length without acceptance)", state_);
  return allowed;
}

void GrammarConstraint::accept(int32_t token_id) {
  if (token_id == eos_id_) return;
  TESSERACT_CHECK(token_id >= 0 &&
                      token_id < static_cast<int32_t>(token_bytes_.size()),
                  "GrammarConstraint::accept: token id {} out of range",
                  token_id);
  state_ = step_bytes(state_,
                      token_bytes_[static_cast<std::size_t>(token_id)]);
}

bool GrammarConstraint::at_accepting() const { return a_.accepting(state_); }

}  // namespace tesseract::models
