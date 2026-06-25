#include "tesseract/quant/Scheme.hpp"

#include <memory>

#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Module.hpp"
#include "tesseract/nn/QuantizedLinear.hpp"
#include "tesseract/nn/QuantizedLinearInt4G.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::quant {

std::shared_ptr<nn::Module> quantize_linear(const nn::Linear& src,
                                            const Scheme& scheme) {
  // Both `from_linear` factories internally validate dtype (FP32 /
  // FP16 / BF16), `in_features`, and — for the INT4 path — the group
  // size. We rely on those throws instead of re-validating here so
  // the error message stays in the same place as the packer
  // constraints it describes.
  switch (scheme.method) {
    case Method::Int8Symmetric: {
      auto q = nn::QuantizedLinear::from_linear(src);
      return std::static_pointer_cast<nn::Module>(q);
    }
    case Method::Int4GroupSymmetric: {
      TESSERACT_CHECK(scheme.group_size > 0,
                      "quant::quantize_linear: Int4GroupSymmetric requires "
                      "group_size > 0, got {}", scheme.group_size);
      auto q = nn::QuantizedLinearInt4G::from_linear(src, scheme.group_size);
      return std::static_pointer_cast<nn::Module>(q);
    }
  }
  TESSERACT_THROW("quant::quantize_linear: unknown Method enum value {}",
                  static_cast<int>(scheme.method));
}

}  // namespace tesseract::quant
