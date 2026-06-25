// Tesseract Studio — minimal HTTP/1.1 server (M5 / B-047).
//
// A tiny dependency-free server over POSIX sockets: enough to serve the
// embedded SPA and a small JSON control plane (catalog / validate / run /
// events / codegen / save / load). Single acceptor thread + a thread pool of
// short-lived request handlers; long-poll friendly. Linux-only (the dev box),
// which is all Studio targets for now.

#ifndef TESSERACT_STUDIO_HTTPSERVER_HPP
#define TESSERACT_STUDIO_HTTPSERVER_HPP

#include <functional>
#include <map>
#include <string>

namespace tesseract::studio {

struct HttpRequest {
  std::string method;
  std::string path;                          // without query string
  std::map<std::string, std::string> query;  // parsed ?a=b&c=d
  std::string body;
};

struct HttpResponse {
  int status = 200;
  std::string content_type = "application/json";
  std::string body;

  static HttpResponse json(const std::string& s) {
    return {200, "application/json", s};
  }
  static HttpResponse text(int status, const std::string& s) {
    return {status, "text/plain", s};
  }
};

using HttpHandler = std::function<HttpResponse(const HttpRequest&)>;

class HttpServer {
 public:
  explicit HttpServer(int port) : port_(port) {}

  // Register a handler for an exact method+path (e.g. "POST", "/api/run").
  void route(const std::string& method, const std::string& path,
             HttpHandler h);
  // Fallback handler (used for static assets / 404).
  void set_default(HttpHandler h) { default_ = std::move(h); }

  // Binds and serves forever (blocks). Returns on fatal socket error.
  // `actual_port` (if non-null) receives the bound port (useful when port 0).
  int serve(int* actual_port = nullptr);

 private:
  int port_;
  std::map<std::string, HttpHandler> routes_;  // key = METHOD + " " + PATH
  HttpHandler default_;
};

}  // namespace tesseract::studio

#endif  // TESSERACT_STUDIO_HTTPSERVER_HPP
