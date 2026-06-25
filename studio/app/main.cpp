// Tesseract Studio — application entry point (M5 / B-047).
//
// A single self-contained native executable: it embeds the C++ engine, the
// block-graph core, and the web UI, exposing them through a tiny localhost
// HTTP control plane. Runs headless (the browser is just the display), so it
// builds and serves on a GPU box with no X11/OpenGL. No Python anywhere.

#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "tesseract/studio/Analysis.hpp"
#include "tesseract/studio/BlockCatalog.hpp"
#include "tesseract/studio/BlockGraph.hpp"
#include "tesseract/studio/Codegen.hpp"
#include "tesseract/studio/Executor.hpp"
#include "tesseract/studio/HttpServer.hpp"
#include "tesseract/studio/Json.hpp"
#include "tesseract/studio/WebAssets.hpp"

using namespace tesseract::studio;

namespace {

// Holds the state of the single active run: a growing event log the UI polls,
// a cancel flag, and the worker thread.
struct RunSession {
  std::mutex mu;
  std::vector<Json> events;     // each: {"i": idx, "type": ..., "data": ...}
  std::atomic<bool> running{false};
  std::atomic<bool> cancel{false};
  std::thread worker;

  void reset() {
    std::lock_guard<std::mutex> lk(mu);
    events.clear();
    cancel = false;
  }

  void push(const RunEvent& ev) {
    std::lock_guard<std::mutex> lk(mu);
    Json e = Json::object();
    e.set("i", Json(static_cast<int64_t>(events.size())));
    e.set("type", Json(ev.type));
    e.set("data", ev.data);
    events.push_back(std::move(e));
  }

  Json since(int64_t from) {
    std::lock_guard<std::mutex> lk(mu);
    Json out = Json::object();
    Json arr = Json::array();
    for (size_t i = (from < 0 ? 0 : static_cast<size_t>(from));
         i < events.size(); ++i)
      arr.push_back(events[i]);
    out.set("events", std::move(arr));
    out.set("next", Json(static_cast<int64_t>(events.size())));
    out.set("running", Json(running.load()));
    return out;
  }

  ~RunSession() {
    cancel = true;
    if (worker.joinable()) worker.join();
  }
};

RunSession g_session;
std::string g_default_device = "cpu";

}  // namespace

int main(int argc, char** argv) {
  int port = 8770;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
    else if (a == "--device" && i + 1 < argc) g_default_device = argv[++i];
    else if (a == "--help") {
      std::printf("tesseract_studio [--port N] [--device cpu|cuda|cuda:N]\n");
      return 0;
    }
  }

  HttpServer server(port);

  // --- static assets ----------------------------------------------------- //
  server.set_default([](const HttpRequest& req) -> HttpResponse {
    bool found = false;
    std::string_view body = web_asset(req.path, found);
    if (!found) return HttpResponse::text(404, "not found");
    HttpResponse res;
    res.status = 200;
    res.content_type = std::string(web_mime(req.path));
    res.body.assign(body.data(), body.size());
    return res;
  });

  // --- catalog ------------------------------------------------------------ //
  server.route("GET", "/api/catalog", [](const HttpRequest&) {
    return HttpResponse::json(BlockCatalog::instance().to_json().dump());
  });

  server.route("GET", "/api/health", [](const HttpRequest&) {
    Json o = Json::object();
    o.set("ok", Json(true));
    o.set("device", Json(g_default_device));
    return HttpResponse::json(o.dump());
  });

  // --- validate ----------------------------------------------------------- //
  server.route("POST", "/api/validate", [](const HttpRequest& req) {
    try {
      BlockGraph g = BlockGraph::from_tsb(req.body);
      return HttpResponse::json(analyze(g).to_json().dump());
    } catch (const std::exception& ex) {
      Json o = Json::object();
      o.set("ok", Json(false));
      Json diags = Json::array();
      Json d = Json::object();
      d.set("severity", Json(std::string("error")));
      d.set("node", Json(-1));
      d.set("message", Json(std::string("parse: ") + ex.what()));
      diags.push_back(std::move(d));
      o.set("diagnostics", std::move(diags));
      return HttpResponse::json(o.dump());
    }
  });

  // --- codegen ------------------------------------------------------------ //
  server.route("POST", "/api/codegen", [](const HttpRequest& req) {
    try {
      BlockGraph g = BlockGraph::from_tsb(req.body);
      std::string lang = "cpp";
      auto it = req.query.find("lang");
      if (it != req.query.end()) lang = it->second;
      Json o = Json::object();
      o.set("lang", Json(lang));
      o.set("code", Json(lang == "python" ? generate_python(g)
                                           : generate_cpp(g)));
      return HttpResponse::json(o.dump());
    } catch (const std::exception& ex) {
      return HttpResponse::text(400, ex.what());
    }
  });

  // --- IR (tesseract-dialect MLIR view) ----------------------------------- //
  server.route("POST", "/api/ir", [](const HttpRequest& req) {
    try {
      BlockGraph g = BlockGraph::from_tsb(req.body);
      Json o = Json::object();
      o.set("ir", Json(generate_ir(g)));
      return HttpResponse::json(o.dump());
    } catch (const std::exception& ex) {
      return HttpResponse::text(400, ex.what());
    }
  });

  // --- open (import .tsb OR a generated source file) ---------------------- //
  server.route("POST", "/api/open", [](const HttpRequest& req) {
    // Try raw .tsb first, then a generated file's @tsb header.
    try {
      BlockGraph g = BlockGraph::from_tsb(req.body);
      return HttpResponse::json(g.to_json().dump());
    } catch (...) {
    }
    if (auto g = extract_tsb(req.body))
      return HttpResponse::json(g->to_json().dump());
    return HttpResponse::text(400, "not a .tsb file or a Studio-generated source");
  });

  // --- run ---------------------------------------------------------------- //
  server.route("POST", "/api/run", [](const HttpRequest& req) {
    if (g_session.running.load())
      return HttpResponse::text(409, "a run is already in progress");
    BlockGraph g;
    try {
      g = BlockGraph::from_tsb(req.body);
    } catch (const std::exception& ex) {
      return HttpResponse::text(400, std::string("parse: ") + ex.what());
    }
    std::string device = g.device.empty() ? g_default_device : g.device;
    auto it = req.query.find("device");
    if (it != req.query.end()) device = it->second;

    if (g_session.worker.joinable()) g_session.worker.join();
    g_session.reset();
    g_session.running = true;
    g_session.worker = std::thread([g, device] {
      RunOptions opt;
      opt.device = device;
      opt.cancel = &g_session.cancel;
      Executor::run(g, [](const RunEvent& ev) { g_session.push(ev); }, opt);
      g_session.running = false;
    });

    Json o = Json::object();
    o.set("ok", Json(true));
    o.set("device", Json(device));
    return HttpResponse::json(o.dump());
  });

  server.route("GET", "/api/events", [](const HttpRequest& req) {
    int64_t since = 0;
    auto it = req.query.find("since");
    if (it != req.query.end()) since = std::stoll(it->second);
    return HttpResponse::json(g_session.since(since).dump());
  });

  server.route("POST", "/api/stop", [](const HttpRequest&) {
    g_session.cancel = true;
    Json o = Json::object();
    o.set("ok", Json(true));
    return HttpResponse::json(o.dump());
  });

  int bound = port;
  std::printf("Tesseract Studio — open  http://localhost:%d  in your browser\n",
              port);
  std::printf("(default device: %s)\n", g_default_device.c_str());
  std::fflush(stdout);
  int rc = server.serve(&bound);
  if (rc != 0)
    std::fprintf(stderr, "studio: failed to bind port %d\n", port);
  return rc;
}
