// Wave 16 (B-033) — HF config.json → LlamaConfig parsing.
//
// `LlamaConfig::from_json` is what lets `llama_generate` read a real
// checkpoint's architecture instead of having it typed on the command
// line. These tests pin the scalar extraction across the fields a real
// HF Llama `config.json` carries, plus the robustness cases that matter:
// nested objects (rope_scaling) must be ignored, torch_dtype must map to
// the right DType, and missing fields must fall back to struct defaults.

#include <string>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/DType.hpp"
#include "tesseract/models/Llama.hpp"

using tesseract::DType;
using tesseract::models::LlamaConfig;

TEST_CASE("LlamaConfig::from_json parses a Llama-3.2-1B-style config", "[models][config]") {
  // Trimmed but realistic: field order shuffled, a nested rope_scaling
  // object present (and ignored), torch_dtype bfloat16, GQA, tied head.
  const std::string json = R"JSON({
    "architectures": ["LlamaForCausalLM"],
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "max_position_embeddings": 131072,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 16,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
      "factor": 32.0,
      "high_freq_factor": 4.0,
      "low_freq_factor": 1.0,
      "original_max_position_embeddings": 8192,
      "rope_type": "llama3"
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": true,
    "torch_dtype": "bfloat16",
    "vocab_size": 128256
  })JSON";

  const LlamaConfig c = LlamaConfig::from_json(json);
  REQUIRE(c.vocab_size == 128256);
  REQUIRE(c.hidden_size == 2048);
  REQUIRE(c.num_hidden_layers == 16);
  REQUIRE(c.num_attention_heads == 32);
  REQUIRE(c.num_key_value_heads == 8);
  REQUIRE(c.kv_heads() == 8);
  REQUIRE(c.intermediate_size == 8192);
  REQUIRE(c.max_position_embeddings == 131072);
  REQUIRE(c.rope_theta == 500000.0);
  REQUIRE(c.rms_norm_eps == 1e-05);
  REQUIRE(c.tie_word_embeddings == true);
  REQUIRE(c.bos_token_id == 128000);
  REQUIRE(c.eos_token_id == 128001);
  REQUIRE(c.dtype == DType::BFloat16);
}

TEST_CASE("LlamaConfig::from_json maps torch_dtype variants", "[models][config]") {
  auto with_dtype = [](const char* d) {
    return LlamaConfig::from_json(std::string("{\"vocab_size\":32,\"hidden_size\":16,"
        "\"num_hidden_layers\":1,\"num_attention_heads\":4,\"intermediate_size\":32,"
        "\"torch_dtype\":\"") + d + "\"}");
  };
  REQUIRE(with_dtype("float32").dtype == DType::Float32);
  REQUIRE(with_dtype("float16").dtype == DType::Float16);
  REQUIRE(with_dtype("bfloat16").dtype == DType::BFloat16);
  REQUIRE(with_dtype("float64").dtype == DType::Float64);
}

TEST_CASE("LlamaConfig::from_json fills defaults for missing fields", "[models][config]") {
  // Only the required architecture fields; everything else defaults.
  const std::string json = R"JSON({
    "vocab_size": 100, "hidden_size": 64, "num_hidden_layers": 3,
    "num_attention_heads": 8, "intermediate_size": 128
  })JSON";
  const LlamaConfig def;  // struct defaults
  const LlamaConfig c = LlamaConfig::from_json(json);
  REQUIRE(c.vocab_size == 100);
  REQUIRE(c.hidden_size == 64);
  // num_key_value_heads unspecified ⇒ default 0 ⇒ kv_heads() == query heads.
  REQUIRE(c.num_key_value_heads == 0);
  REQUIRE(c.kv_heads() == 8);
  REQUIRE(c.rope_theta == def.rope_theta);
  REQUIRE(c.tie_word_embeddings == def.tie_word_embeddings);
  REQUIRE(c.bos_token_id == -1);
  REQUIRE(c.eos_token_id == -1);
  REQUIRE(c.dtype == def.dtype);
}

TEST_CASE("LlamaConfig::from_json rejects malformed input", "[models][config]") {
  REQUIRE_THROWS(LlamaConfig::from_json("not json at all"));
  // A JSON object missing every required arch field is invalid.
  REQUIRE_THROWS(LlamaConfig::from_json("{\"model_type\":\"llama\"}"));
}
