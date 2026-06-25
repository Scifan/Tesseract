// Tesseract Studio — graph executor (M5 / B-047).
//
// Turns a validated BlockGraph into live engine work: it instantiates the C++
// nn/models objects the blocks describe and runs the sink (TrainLoop or
// Generate), emitting a stream of typed events (log / loss / metric / token /
// text / done / error). The executor is synchronous and engine-facing; the app
// layer runs it on a worker thread and forwards events to the UI. No GUI, no
// Python — pure framework calls, so it is unit-testable headless.

#ifndef TESSERACT_STUDIO_EXECUTOR_HPP
#define TESSERACT_STUDIO_EXECUTOR_HPP

#include <atomic>
#include <functional>
#include <string>

#include "tesseract/studio/BlockGraph.hpp"
#include "tesseract/studio/Json.hpp"

namespace tesseract::studio {

struct RunEvent {
  std::string type;  // "log" | "loss" | "metric" | "token" | "text" | "done" | "error"
  Json data;
};

using EventSink = std::function<void(const RunEvent&)>;

struct RunOptions {
  std::string device = "cpu";          // "cpu" | "cuda" | "cuda:N"
  const std::atomic<bool>* cancel = nullptr;  // cooperative cancellation
};

class Executor {
 public:
  // Runs the graph to completion (or until cancelled), emitting events through
  // `sink`. Returns a summary Json. Never throws — failures become an "error"
  // event and an `{"ok": false}` summary.
  static Json run(const BlockGraph& g, const EventSink& sink,
                  const RunOptions& opt);
};

}  // namespace tesseract::studio

#endif  // TESSERACT_STUDIO_EXECUTOR_HPP
