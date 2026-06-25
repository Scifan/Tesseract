// Tesseract Studio — embedded web asset accessor (M5 / B-047).
//
// The UI (index.html / app.js / style.css) is compiled into the binary as
// byte arrays (generated at build time by studio/CMakeLists.txt) so the
// resulting executable is fully self-contained — no asset directory to ship.

#ifndef TESSERACT_STUDIO_WEBASSETS_HPP
#define TESSERACT_STUDIO_WEBASSETS_HPP

#include <string_view>

namespace tesseract::studio {

// Returns the embedded asset for `path` (e.g. "/", "/app.js", "/style.css").
// `found` is set to false when the path is unknown (caller returns 404).
std::string_view web_asset(std::string_view path, bool& found);

// MIME type for a path's extension.
std::string_view web_mime(std::string_view path);

}  // namespace tesseract::studio

#endif  // TESSERACT_STUDIO_WEBASSETS_HPP
