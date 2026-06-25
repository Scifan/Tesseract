#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"

using tesseract::DType;
using tesseract::Shape;
using tesseract::Tensor;

TEST_CASE("Default-constructed Tensor is undefined", "[tensor][ctor]") {
  Tensor t;
  REQUIRE_FALSE(t.defined());
  REQUIRE_THROWS_AS(t.rank(), tesseract::Error);
}

TEST_CASE("empty / zeros / ones produce correct shapes and dtypes", "[tensor][factory]") {
  Tensor e = Tensor::empty({2, 3}, DType::Float32);
  REQUIRE(e.defined());
  REQUIRE(e.shape() == Shape{2, 3});
  REQUIRE(e.dtype() == DType::Float32);
  REQUIRE(e.rank() == 2);
  REQUIRE(e.numel() == 6);
  REQUIRE(e.nbytes() == 24);
  REQUIRE(e.is_contiguous());
  REQUIRE(e.strides() == Shape{3, 1});

  Tensor z = Tensor::zeros({4}, DType::Int64);
  const int64_t* zp = z.data_ptr<int64_t>();
  for (int64_t i = 0; i < 4; ++i) REQUIRE(zp[i] == 0);

  Tensor o = Tensor::ones({3}, DType::Float64);
  const double* op = o.data_ptr<double>();
  for (int i = 0; i < 3; ++i) REQUIRE(op[i] == 1.0);
}

TEST_CASE("full and arange", "[tensor][factory]") {
  Tensor f = Tensor::full({2, 2}, 7.5, DType::Float32);
  const float* fp = f.data_ptr<float>();
  for (int i = 0; i < 4; ++i) REQUIRE(fp[i] == 7.5f);

  Tensor a = Tensor::arange(5);
  REQUIRE(a.shape() == Shape{5});
  REQUIRE(a.dtype() == DType::Int64);
  const int64_t* ap = a.data_ptr<int64_t>();
  for (int i = 0; i < 5; ++i) REQUIRE(ap[i] == i);

  Tensor b = Tensor::arange(2, 10, 2, DType::Int32);
  REQUIRE(b.shape() == Shape{4});
  const int32_t* bp = b.data_ptr<int32_t>();
  const int32_t expected[] = {2, 4, 6, 8};
  for (int i = 0; i < 4; ++i) REQUIRE(bp[i] == expected[i]);
}

TEST_CASE("from_blob wraps externally owned memory", "[tensor][factory]") {
  std::vector<float> buf = {1, 2, 3, 4, 5, 6};
  Tensor t = Tensor::from_blob(buf.data(), Shape{2, 3}, DType::Float32);
  REQUIRE(t.is_contiguous());
  REQUIRE(t.data_ptr<float>() == buf.data());
  REQUIRE(t.numel() == 6);
  buf[0] = 99.0f;
  REQUIRE(t.data_ptr<float>()[0] == 99.0f);
}

TEST_CASE("view and reshape of contiguous tensors", "[tensor][view]") {
  Tensor t = Tensor::arange(12, DType::Int32);
  Tensor v = t.view({3, 4});
  REQUIRE(v.shape() == Shape{3, 4});
  REQUIRE(v.is_contiguous());
  REQUIRE(v.data_ptr<int32_t>() == t.data_ptr<int32_t>());  // shares storage

  Tensor r = t.reshape({2, 2, 3});
  REQUIRE(r.shape() == Shape{2, 2, 3});
  REQUIRE(r.is_contiguous());
}

TEST_CASE("view rejects shape with mismatched numel", "[tensor][view]") {
  Tensor t = Tensor::arange(12, DType::Int32);
  REQUIRE_THROWS_AS(t.view({5, 3}), tesseract::Error);
}

TEST_CASE("permute produces non-contiguous view and contiguous() copies", "[tensor][permute]") {
  std::vector<float> data(24);
  for (int i = 0; i < 24; ++i) data[i] = static_cast<float>(i);

  Tensor src = Tensor::from_blob(data.data(), Shape{2, 3, 4}, DType::Float32);
  REQUIRE(src.is_contiguous());

  // (2,3,4) -> permute(2,0,1) -> (4,2,3)
  Tensor p = src.permute({2, 0, 1});
  REQUIRE(p.shape() == Shape{4, 2, 3});
  REQUIRE_FALSE(p.is_contiguous());
  // strides should be the permuted original strides {12,4,1} -> {1,12,4}.
  REQUIRE(p.strides() == Shape{1, 12, 4});

  Tensor c = p.contiguous();
  REQUIRE(c.is_contiguous());
  REQUIRE(c.shape() == Shape{4, 2, 3});
  REQUIRE(c.strides() == Shape{6, 3, 1});

  // Sanity check that element order matches the strided walk.
  const float* cp = c.data_ptr<float>();
  const float* sp = src.data_ptr<float>();
  for (int k = 0; k < 4; ++k)
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j) {
        const float expected = sp[i * 12 + j * 4 + k * 1];
        const float actual = cp[k * 6 + i * 3 + j];
        REQUIRE(expected == actual);
      }
}

TEST_CASE("transpose swaps two dims", "[tensor][transpose]") {
  Tensor t = Tensor::arange(6, DType::Int32).view({2, 3});
  Tensor tr = t.transpose(0, 1);
  REQUIRE(tr.shape() == Shape{3, 2});
  REQUIRE(tr.strides() == Shape{1, 3});
  REQUIRE_FALSE(tr.is_contiguous());

  Tensor cc = tr.contiguous();
  const int32_t* cp = cc.data_ptr<int32_t>();
  // Original data laid out as [[0,1,2],[3,4,5]] -> transposed -> [[0,3],[1,4],[2,5]]
  const int32_t expected[] = {0, 3, 1, 4, 2, 5};
  for (int i = 0; i < 6; ++i) REQUIRE(cp[i] == expected[i]);
}

TEST_CASE("clone produces an independent copy", "[tensor][clone]") {
  Tensor t = Tensor::arange(4, DType::Int32);
  Tensor c = t.clone();
  REQUIRE(c.is_contiguous());
  REQUIRE(c.data_ptr<int32_t>() != t.data_ptr<int32_t>());
  c.data_ptr<int32_t>()[0] = 42;
  REQUIRE(t.data_ptr<int32_t>()[0] == 0);
}

TEST_CASE("data_ptr<T> enforces dtype match", "[tensor][safety]") {
  Tensor t = Tensor::zeros({2}, DType::Float32);
  REQUIRE_THROWS_AS(t.data_ptr<int32_t>(), tesseract::Error);
}

TEST_CASE("to_string produces a non-empty diagnostic", "[tensor][repr]") {
  Tensor t = Tensor::arange(3, DType::Int32);
  const std::string s = t.to_string();
  REQUIRE(s.find("Tensor(") == 0);
  REQUIRE(s.find("shape=[3]") != std::string::npos);
  REQUIRE(s.find("dtype=i32") != std::string::npos);
}
