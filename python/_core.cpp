// Tesseract Python frontend (M4 Track B1 / B-041).
//
// A pybind11 extension `tesseract._core` exposing the training + inference
// stack: Tensor (with NumPy buffer-protocol interop + autograd), the core ops,
// the nn modules, optimizers, and the Llama / Mamba generation models plus
// tokenizer + sampler. The thin pure-Python `tesseract/` package re-exports
// these under ergonomic submodule names.
//
// Scope: CPU-first (every binding takes the same `Device` arg, so a CUDA build
// rides the same surface). The binding deliberately stays a thin shim — all
// logic lives in the C++ libraries; nothing is reimplemented here.

#include <cstring>
#include <memory>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/DType.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/io/BpeTokenizer.hpp"
#include "tesseract/io/Tokenizer.hpp"
#include "tesseract/models/Llama.hpp"
#include "tesseract/models/MambaModel.hpp"
#include "tesseract/models/Sampler.hpp"
#include "tesseract/nn/Activation.hpp"
#include "tesseract/nn/Embedding.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Module.hpp"
#include "tesseract/nn/Sequential.hpp"
#include "tesseract/optim/Adam.hpp"
#include "tesseract/optim/SGD.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Loss.hpp"
#include "tesseract/ops/MatMul.hpp"
#include "tesseract/ops/Softmax.hpp"
#include "tesseract/ops/View.hpp"

namespace py = pybind11;
using namespace tesseract;

namespace {

DType dtype_of_numpy(const py::array& arr) {
  if (arr.dtype().is(py::dtype::of<float>()))   return DType::Float32;
  if (arr.dtype().is(py::dtype::of<double>()))  return DType::Float64;
  if (arr.dtype().is(py::dtype::of<int64_t>())) return DType::Int64;
  if (arr.dtype().is(py::dtype::of<int32_t>())) return DType::Int32;
  throw std::runtime_error(
      "tesseract: unsupported numpy dtype (use float32/float64/int64/int32)");
}

Tensor tensor_from_numpy(const py::array& arr_in) {
  // Force a C-contiguous copy so the byte layout matches our row-major Tensor.
  py::array arr = py::array::ensure(arr_in, py::array::c_style);
  if (!arr) throw std::runtime_error("tesseract: could not coerce array to C-contiguous");
  py::buffer_info info = arr.request();
  std::vector<int64_t> shape(info.shape.begin(), info.shape.end());
  const DType dt = dtype_of_numpy(arr);
  Tensor t = Tensor::empty(Shape(shape), dt, cpu_device());
  if (t.nbytes() > 0)
    std::memcpy(t.raw_data(), info.ptr, t.nbytes());
  return t;
}

py::array tensor_to_numpy(const Tensor& t_in) {
  Tensor t = t_in.device().is_cpu() ? t_in : t_in.to(cpu_device());
  t = t.is_contiguous() ? t : t.contiguous();
  std::vector<py::ssize_t> shape;
  for (int64_t i = 0; i < t.rank(); ++i)
    shape.push_back(static_cast<py::ssize_t>(t.shape()[i]));
  py::dtype dt;
  switch (t.dtype()) {
    case DType::Float32: dt = py::dtype::of<float>(); break;
    case DType::Float64: dt = py::dtype::of<double>(); break;
    case DType::Int64:   dt = py::dtype::of<int64_t>(); break;
    case DType::Int32:   dt = py::dtype::of<int32_t>(); break;
    default:
      throw std::runtime_error("tesseract: numpy() only supports "
                               "float32/float64/int64/int32 tensors");
  }
  py::array arr(dt, shape);
  if (t.nbytes() > 0)
    std::memcpy(arr.mutable_data(), t.raw_data(), t.nbytes());
  return arr;
}

std::vector<int64_t> tensor_shape(const Tensor& t) {
  std::vector<int64_t> s;
  for (int64_t i = 0; i < t.rank(); ++i) s.push_back(t.shape()[i]);
  return s;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Tesseract C++ deep-learning framework — Python frontend";

  py::enum_<DType>(m, "DType")
      .value("float32", DType::Float32)
      .value("float64", DType::Float64)
      .value("int64", DType::Int64)
      .value("int32", DType::Int32);

  py::class_<Device>(m, "Device")
      .def_static("cpu", []() { return cpu_device(); })
      .def("is_cpu", &Device::is_cpu)
      .def("is_cuda", &Device::is_cuda)
      .def("__repr__", [](const Device& d) { return d.to_string(); });

  // --- Tensor -------------------------------------------------------------- //
  py::class_<Tensor>(m, "Tensor")
      .def(py::init([](const py::array& a) { return tensor_from_numpy(a); }),
           py::arg("array"))
      .def_static("from_numpy", &tensor_from_numpy, py::arg("array"))
      .def_static("zeros",
                  [](std::vector<int64_t> shape, DType dt) {
                    return Tensor::zeros(Shape(shape), dt, cpu_device());
                  },
                  py::arg("shape"), py::arg("dtype") = DType::Float32)
      .def_static("ones",
                  [](std::vector<int64_t> shape, DType dt) {
                    return Tensor::ones(Shape(shape), dt, cpu_device());
                  },
                  py::arg("shape"), py::arg("dtype") = DType::Float32)
      .def_static("full",
                  [](std::vector<int64_t> shape, double v, DType dt) {
                    return Tensor::full(Shape(shape), v, dt, cpu_device());
                  },
                  py::arg("shape"), py::arg("value"),
                  py::arg("dtype") = DType::Float32)
      .def("numpy", &tensor_to_numpy)
      .def_property_readonly("shape", &tensor_shape)
      .def_property_readonly("dtype", &Tensor::dtype)
      .def_property_readonly("ndim", &Tensor::rank)
      .def_property_readonly("numel", &Tensor::numel)
      .def_property_readonly("device", &Tensor::device)
      .def_property("requires_grad", &Tensor::requires_grad,
                    &Tensor::set_requires_grad)
      .def("grad", [](const Tensor& t) -> py::object {
        const Tensor& g = t.grad();
        if (!g.defined()) return py::none();
        return py::cast(g);
      })
      .def("backward",
           [](const Tensor& t, py::object grad) {
             if (grad.is_none()) Engine::backward(t);
             else Engine::backward(t, grad.cast<Tensor>());
           },
           py::arg("grad") = py::none())
      .def("item", [](const Tensor& t) {
        Tensor h = t.device().is_cpu() ? t : t.to(cpu_device());
        h = h.is_contiguous() ? h : h.contiguous();
        if (h.dtype() == DType::Float64) return static_cast<double>(h.data_ptr<double>()[0]);
        if (h.dtype() == DType::Int64)   return static_cast<double>(h.data_ptr<int64_t>()[0]);
        if (h.dtype() == DType::Int32)   return static_cast<double>(h.data_ptr<int32_t>()[0]);
        return static_cast<double>(h.data_ptr<float>()[0]);
      })
      .def("reshape", [](const Tensor& t, std::vector<int64_t> shape) {
        return ops::reshape(t, Shape(shape));
      })
      .def("__add__", [](const Tensor& a, const Tensor& b) { return ops::add(a, b); })
      .def("__sub__", [](const Tensor& a, const Tensor& b) { return ops::sub(a, b); })
      .def("__mul__", [](const Tensor& a, const Tensor& b) { return ops::mul(a, b); })
      .def("matmul",  [](const Tensor& a, const Tensor& b) { return ops::matmul(a, b); })
      .def("__matmul__", [](const Tensor& a, const Tensor& b) { return ops::matmul(a, b); })
      .def("__repr__", [](const Tensor& t) { return t.to_string(); });

  m.def("backward",
        [](const Tensor& t, py::object grad) {
          if (grad.is_none()) Engine::backward(t);
          else Engine::backward(t, grad.cast<Tensor>());
        },
        py::arg("root"), py::arg("grad") = py::none());

  // --- ops ---------------------------------------------------------------- //
  py::module ops = m.def_submodule("ops", "Functional operations");
  ops.def("add", &ops::add);
  ops.def("sub", &ops::sub);
  ops.def("mul", &ops::mul);
  ops.def("div", &ops::div);
  ops.def("matmul", &ops::matmul);
  ops.def("relu", &ops::relu);
  ops.def("sigmoid", &ops::sigmoid);
  ops.def("softmax", &ops::softmax, py::arg("x"), py::arg("dim"));
  ops.def("cross_entropy", &ops::cross_entropy_with_logits,
          py::arg("logits"), py::arg("targets"));

  // --- nn ----------------------------------------------------------------- //
  py::module nn = m.def_submodule("nn", "Neural-network modules");
  py::class_<nn::Module, std::shared_ptr<nn::Module>>(nn, "Module")
      .def("forward", &nn::Module::forward)
      .def("__call__", &nn::Module::forward)
      .def("parameters", &nn::Module::parameters)
      .def("zero_grad", &nn::Module::zero_grad)
      .def("train", &nn::Module::train, py::arg("on") = true)
      .def("eval", &nn::Module::eval);

  py::class_<nn::Linear, nn::Module, std::shared_ptr<nn::Linear>>(nn, "Linear")
      .def(py::init<int64_t, int64_t, bool, DType>(), py::arg("in_features"),
           py::arg("out_features"), py::arg("bias") = true,
           py::arg("dtype") = DType::Float32)
      .def("weight", &nn::Linear::weight);

  py::class_<nn::Embedding, nn::Module, std::shared_ptr<nn::Embedding>>(
      nn, "Embedding")
      .def(py::init<int64_t, int64_t, DType>(), py::arg("num_embeddings"),
           py::arg("embedding_dim"), py::arg("dtype") = DType::Float32);

  py::class_<nn::ReLU, nn::Module, std::shared_ptr<nn::ReLU>>(nn, "ReLU")
      .def(py::init<>());
  py::class_<nn::Sigmoid, nn::Module, std::shared_ptr<nn::Sigmoid>>(nn, "Sigmoid")
      .def(py::init<>());
  py::class_<nn::Tanh, nn::Module, std::shared_ptr<nn::Tanh>>(nn, "Tanh")
      .def(py::init<>());

  py::class_<nn::Sequential, nn::Module, std::shared_ptr<nn::Sequential>>(
      nn, "Sequential")
      .def(py::init([](std::vector<std::shared_ptr<nn::Module>> mods) {
        auto seq = std::make_shared<nn::Sequential>();
        for (auto& mm : mods) seq->add(mm);
        return seq;
      }), py::arg("modules"))
      .def("add", &nn::Sequential::add);

  // --- optim -------------------------------------------------------------- //
  py::module optim = m.def_submodule("optim", "Optimizers");
  py::class_<optim::SGD>(optim, "SGD")
      .def(py::init<std::vector<Tensor>, double, double>(), py::arg("params"),
           py::arg("lr"), py::arg("momentum") = 0.0)
      .def("step", &optim::SGD::step)
      .def("zero_grad", &optim::SGD::zero_grad);
  py::class_<optim::Adam>(optim, "Adam")
      .def(py::init<std::vector<Tensor>, double, double, double, double>(),
           py::arg("params"), py::arg("lr") = 1e-3, py::arg("beta1") = 0.9,
           py::arg("beta2") = 0.999, py::arg("eps") = 1e-8)
      .def("step", &optim::Adam::step)
      .def("zero_grad", &optim::Adam::zero_grad);

  // --- models ------------------------------------------------------------- //
  py::module models = m.def_submodule("models", "Generation models");

  py::class_<models::SamplingParams>(models, "SamplingParams")
      .def(py::init<>())
      .def_readwrite("temperature", &models::SamplingParams::temperature)
      .def_readwrite("top_k", &models::SamplingParams::top_k)
      .def_readwrite("top_p", &models::SamplingParams::top_p)
      .def_readwrite("repetition_penalty",
                     &models::SamplingParams::repetition_penalty);

  py::class_<models::LlamaConfig>(models, "LlamaConfig")
      .def(py::init<>())
      .def_static("llama_3_2_1b", &models::LlamaConfig::llama_3_2_1b)
      .def_static("from_json", &models::LlamaConfig::from_json)
      .def_readwrite("vocab_size", &models::LlamaConfig::vocab_size)
      .def_readwrite("hidden_size", &models::LlamaConfig::hidden_size)
      .def_readwrite("num_hidden_layers", &models::LlamaConfig::num_hidden_layers)
      .def_readwrite("num_attention_heads",
                     &models::LlamaConfig::num_attention_heads)
      .def_readwrite("num_key_value_heads",
                     &models::LlamaConfig::num_key_value_heads)
      .def_readwrite("intermediate_size", &models::LlamaConfig::intermediate_size)
      .def_readwrite("num_experts", &models::LlamaConfig::num_experts)
      .def_readwrite("num_experts_per_tok",
                     &models::LlamaConfig::num_experts_per_tok)
      .def_readwrite("max_position_embeddings",
                     &models::LlamaConfig::max_position_embeddings)
      .def_readwrite("rope_theta", &models::LlamaConfig::rope_theta)
      .def_readwrite("rms_norm_eps", &models::LlamaConfig::rms_norm_eps)
      .def_readwrite("eos_token_id", &models::LlamaConfig::eos_token_id);

  py::class_<models::LlamaModel::GenerateConfig>(models, "LlamaGenerateConfig")
      .def(py::init<>())
      .def_readwrite("max_new_tokens",
                     &models::LlamaModel::GenerateConfig::max_new_tokens)
      .def_readwrite("eos_token_id",
                     &models::LlamaModel::GenerateConfig::eos_token_id)
      .def_readwrite("do_sample", &models::LlamaModel::GenerateConfig::do_sample)
      .def_readwrite("sampling", &models::LlamaModel::GenerateConfig::sampling)
      .def_readwrite("seed", &models::LlamaModel::GenerateConfig::seed)
      .def_readwrite("kv_int8", &models::LlamaModel::GenerateConfig::kv_int8);

  py::class_<models::LlamaModel, nn::Module, std::shared_ptr<models::LlamaModel>>(
      models, "LlamaModel")
      .def(py::init<const models::LlamaConfig&>(), py::arg("config"))
      .def_static("from_pretrained", &models::LlamaModel::from_pretrained,
                  py::arg("path"), py::arg("config"))
      .def("forward", &models::LlamaModel::forward)
      .def("generate", &models::LlamaModel::generate, py::arg("prompt_ids"),
           py::arg("config"))
      .def("config", &models::LlamaModel::config);

  py::class_<models::MambaConfig>(models, "MambaConfig")
      .def(py::init<>())
      .def_readwrite("vocab_size", &models::MambaConfig::vocab_size)
      .def_readwrite("hidden_size", &models::MambaConfig::hidden_size)
      .def_readwrite("num_hidden_layers", &models::MambaConfig::num_hidden_layers)
      .def_readwrite("d_state", &models::MambaConfig::d_state)
      .def_readwrite("d_conv", &models::MambaConfig::d_conv)
      .def_readwrite("expand", &models::MambaConfig::expand)
      .def_readwrite("eos_token_id", &models::MambaConfig::eos_token_id);

  py::class_<models::MambaModel::GenerateConfig>(models, "MambaGenerateConfig")
      .def(py::init<>())
      .def_readwrite("max_new_tokens",
                     &models::MambaModel::GenerateConfig::max_new_tokens)
      .def_readwrite("eos_token_id",
                     &models::MambaModel::GenerateConfig::eos_token_id);

  py::class_<models::MambaModel, nn::Module, std::shared_ptr<models::MambaModel>>(
      models, "MambaModel")
      .def(py::init<const models::MambaConfig&>(), py::arg("config"))
      .def("forward", &models::MambaModel::forward)
      .def("generate", &models::MambaModel::generate, py::arg("prompt_ids"),
           py::arg("config"));

  // --- io (tokenizers) ---------------------------------------------------- //
  py::module io = m.def_submodule("io", "Tokenizers and loaders");
  py::class_<io::Tokenizer, std::shared_ptr<io::Tokenizer>>(io, "Tokenizer")
      .def("encode", &io::Tokenizer::encode, py::arg("text"),
           py::arg("add_special_tokens") = true)
      .def("decode",
           [](const io::Tokenizer& tk, std::vector<int32_t> ids, bool skip) {
             return tk.decode(std::span<const int32_t>(ids.data(), ids.size()),
                              skip);
           },
           py::arg("ids"), py::arg("skip_special_tokens") = true)
      .def("vocab_size", &io::Tokenizer::vocab_size)
      .def("bos_token_id", &io::Tokenizer::bos_token_id)
      .def("eos_token_id", &io::Tokenizer::eos_token_id);

  py::class_<io::WhitespaceTokenizer::Config>(io, "WhitespaceTokenizerConfig")
      .def(py::init<>())
      .def_readwrite("vocab", &io::WhitespaceTokenizer::Config::vocab)
      .def_readwrite("bos", &io::WhitespaceTokenizer::Config::bos)
      .def_readwrite("eos", &io::WhitespaceTokenizer::Config::eos)
      .def_readwrite("pad", &io::WhitespaceTokenizer::Config::pad)
      .def_readwrite("unk", &io::WhitespaceTokenizer::Config::unk);

  py::class_<io::WhitespaceTokenizer, io::Tokenizer,
             std::shared_ptr<io::WhitespaceTokenizer>>(io, "WhitespaceTokenizer")
      .def(py::init<io::WhitespaceTokenizer::Config>(), py::arg("config"));

  py::class_<io::BpeTokenizer, io::Tokenizer,
             std::shared_ptr<io::BpeTokenizer>>(io, "BpeTokenizer")
      .def_static("from_file",
                  [](const std::string& path) {
                    return std::make_shared<io::BpeTokenizer>(
                        io::BpeTokenizer::from_file(path));
                  })
      .def_static("from_json", [](const std::string& json) {
        return std::make_shared<io::BpeTokenizer>(
            io::BpeTokenizer::from_json(json));
      });
}
