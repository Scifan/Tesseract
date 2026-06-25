#include "tesseract/studio/HttpServer.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <sstream>
#include <thread>

namespace tesseract::studio {

namespace {

std::string url_decode(const std::string& s) {
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%' && i + 2 < s.size()) {
      int v = std::stoi(s.substr(i + 1, 2), nullptr, 16);
      out.push_back(static_cast<char>(v));
      i += 2;
    } else if (s[i] == '+') {
      out.push_back(' ');
    } else {
      out.push_back(s[i]);
    }
  }
  return out;
}

bool read_request(int fd, HttpRequest& req) {
  std::string buf;
  char tmp[4096];
  // Read until we have headers (\r\n\r\n).
  size_t header_end = std::string::npos;
  while (header_end == std::string::npos) {
    ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
    if (n <= 0) return false;
    buf.append(tmp, static_cast<size_t>(n));
    header_end = buf.find("\r\n\r\n");
    if (buf.size() > (8u << 20)) return false;  // 8 MiB header guard
  }
  std::string head = buf.substr(0, header_end);
  std::string rest = buf.substr(header_end + 4);

  std::istringstream hs(head);
  std::string line;
  std::getline(hs, line);
  if (!line.empty() && line.back() == '\r') line.pop_back();
  {
    std::istringstream rl(line);
    std::string target;
    rl >> req.method >> target;
    auto qpos = target.find('?');
    if (qpos == std::string::npos) {
      req.path = target;
    } else {
      req.path = target.substr(0, qpos);
      std::string q = target.substr(qpos + 1);
      std::istringstream qs(q);
      std::string kv;
      while (std::getline(qs, kv, '&')) {
        auto eq = kv.find('=');
        if (eq == std::string::npos)
          req.query[url_decode(kv)] = "";
        else
          req.query[url_decode(kv.substr(0, eq))] = url_decode(kv.substr(eq + 1));
      }
    }
  }
  size_t content_length = 0;
  while (std::getline(hs, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) break;
    auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    std::string key = line.substr(0, colon);
    std::string val = line.substr(colon + 1);
    for (auto& c : key) c = static_cast<char>(::tolower(c));
    size_t a = val.find_first_not_of(" \t");
    if (a != std::string::npos) val = val.substr(a);
    if (key == "content-length") content_length = std::stoul(val);
  }
  // Read the body (if any).
  req.body = rest;
  while (req.body.size() < content_length) {
    ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
    if (n <= 0) break;
    req.body.append(tmp, static_cast<size_t>(n));
  }
  return true;
}

void write_all(int fd, const std::string& s) {
  size_t off = 0;
  while (off < s.size()) {
    ssize_t n = ::send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
    if (n <= 0) break;
    off += static_cast<size_t>(n);
  }
}

void send_response(int fd, const HttpResponse& res) {
  std::ostringstream os;
  os << "HTTP/1.1 " << res.status << " "
     << (res.status == 200 ? "OK" : (res.status == 404 ? "Not Found" : "Error"))
     << "\r\n"
     << "Content-Type: " << res.content_type << "\r\n"
     << "Content-Length: " << res.body.size() << "\r\n"
     << "Access-Control-Allow-Origin: *\r\n"
     << "Connection: close\r\n\r\n";
  std::string head = os.str();
  write_all(fd, head);
  write_all(fd, res.body);
}

}  // namespace

void HttpServer::route(const std::string& method, const std::string& path,
                       HttpHandler h) {
  routes_[method + " " + path] = std::move(h);
}

int HttpServer::serve(int* actual_port) {
  int srv = ::socket(AF_INET, SOCK_STREAM, 0);
  if (srv < 0) return -1;
  int yes = 1;
  ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(static_cast<uint16_t>(port_));
  if (::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(srv);
    return -1;
  }
  if (::listen(srv, 64) < 0) {
    ::close(srv);
    return -1;
  }
  if (actual_port) {
    sockaddr_in bound{};
    socklen_t len = sizeof(bound);
    if (::getsockname(srv, reinterpret_cast<sockaddr*>(&bound), &len) == 0)
      *actual_port = ntohs(bound.sin_port);
  }

  while (true) {
    int cfd = ::accept(srv, nullptr, nullptr);
    if (cfd < 0) continue;
    std::thread([this, cfd] {
      HttpRequest req;
      if (read_request(cfd, req)) {
        HttpResponse res;
        auto it = routes_.find(req.method + " " + req.path);
        if (it != routes_.end()) {
          res = it->second(req);
        } else if (default_) {
          res = default_(req);
        } else {
          res = HttpResponse::text(404, "not found");
        }
        send_response(cfd, res);
      }
      ::close(cfd);
    }).detach();
  }
  ::close(srv);
  return 0;
}

}  // namespace tesseract::studio
