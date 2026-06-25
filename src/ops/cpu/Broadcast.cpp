#include "tesseract/ops/Broadcast.hpp"

#include <algorithm>

#include "tesseract/utils/Logging.hpp"

namespace tesseract::ops {

Shape broadcast_shape(const Shape& a, const Shape& b) {
  const std::size_t ra = a.rank();
  const std::size_t rb = b.rank();
  const std::size_t rout = std::max(ra, rb);
  Shape out;
  out.resize(rout);
  for (std::size_t i = 0; i < rout; ++i) {
    const int64_t da = (i < rout - ra) ? 1 : a[i - (rout - ra)];
    const int64_t db = (i < rout - rb) ? 1 : b[i - (rout - rb)];
    if (da == db) {
      out[i] = da;
    } else if (da == 1) {
      out[i] = db;
    } else if (db == 1) {
      out[i] = da;
    } else {
      TESSERACT_THROW("broadcast_shape: dim {} incompatible ({} vs {})", i, da, db);
    }
  }
  return out;
}

void align_for_broadcast(const Shape& in_shape, const Shape& in_strides,
                         const Shape& out_shape, Shape& out_strides) {
  const std::size_t ri = in_shape.rank();
  const std::size_t ro = out_shape.rank();
  TESSERACT_CHECK(ri <= ro, "align_for_broadcast: input rank {} > output rank {}", ri, ro);
  out_strides.resize(ro);
  const std::size_t offset = ro - ri;
  for (std::size_t i = 0; i < ro; ++i) {
    if (i < offset) {
      out_strides[i] = 0;
    } else {
      const int64_t in_dim = in_shape[i - offset];
      const int64_t in_stride = in_strides[i - offset];
      if (in_dim == 1 && out_shape[i] != 1) {
        out_strides[i] = 0;
      } else {
        TESSERACT_CHECK(in_dim == out_shape[i],
                        "align_for_broadcast: cannot broadcast dim {} ({} vs {})",
                        i, in_dim, out_shape[i]);
        out_strides[i] = in_stride;
      }
    }
  }
}

bool is_broadcastable_to(const Shape& from, const Shape& to) noexcept {
  if (from.rank() > to.rank()) return false;
  const std::size_t offset = to.rank() - from.rank();
  for (std::size_t i = 0; i < from.rank(); ++i) {
    const int64_t f = from[i];
    const int64_t t = to[i + offset];
    if (f != t && f != 1) return false;
  }
  return true;
}

}  // namespace tesseract::ops
