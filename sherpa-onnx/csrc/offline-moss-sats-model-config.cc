// sherpa-onnx/csrc/offline-moss-sats-model-config.cc
//
// Copyright (c)  2026

#include "sherpa-onnx/csrc/offline-moss-sats-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineMossSatsModelConfig::Register(ParseOptions *po) {
  po->Register("moss-sats-encoder", &encoder,
               "Path to the MOSS-SATS audio encoder onnx (Whisper encoder + "
               "time merge + adaptor), e.g., encoder.onnx");
  po->Register("moss-sats-embedding", &embedding,
               "Path to the MOSS-SATS token-embedding onnx, e.g., "
               "embedding.onnx");
  po->Register("moss-sats-decoder", &decoder,
               "Path to the MOSS-SATS LLM decoder onnx with KV cache, e.g., "
               "decoder.onnx");
  po->Register("moss-sats-tokenizer-dir", &tokenizer,
               "Directory containing the Qwen BPE tokenizer assets "
               "(tokenizer.json or vocab.json + merges.txt)");
  po->Register("moss-sats-max-new-tokens", &max_new_tokens,
               "Maximum number of tokens to generate per audio window");
  po->Register("moss-sats-hotwords", &hotwords,
               "Optional comma-separated hotwords injected into the prompt");
  po->Register("moss-sats-temperature", &temperature,
               "Sampling temperature; 0 = greedy");
  po->Register("moss-sats-top-p", &top_p, "Top-p nucleus sampling");
  po->Register("moss-sats-seed", &seed, "Sampling seed");
  po->Register("moss-sats-max-total-len", &max_total_len,
               "KV-cache length fallback if the decoder graph's cache dim is "
               "dynamic");
}

bool OfflineMossSatsModelConfig::Validate() const {
  if (encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --moss-sats-encoder");
    return false;
  }
  if (!FileExists(encoder)) {
    SHERPA_ONNX_LOGE("MOSS-SATS encoder '%s' does not exist", encoder.c_str());
    return false;
  }
  // `embedding` is optional: the sherpa-interface decoder embeds input_ids
  // in-graph. Kept for potential future split-graph deployments.
  if (!embedding.empty() && !FileExists(embedding)) {
    SHERPA_ONNX_LOGE("MOSS-SATS embedding '%s' does not exist",
                     embedding.c_str());
    return false;
  }
  if (decoder.empty() || !FileExists(decoder)) {
    SHERPA_ONNX_LOGE("MOSS-SATS decoder '%s' does not exist", decoder.c_str());
    return false;
  }
  if (tokenizer.empty()) {
    SHERPA_ONNX_LOGE("Please provide --moss-sats-tokenizer-dir");
    return false;
  }
  if (max_new_tokens <= 0) {
    SHERPA_ONNX_LOGE("--moss-sats-max-new-tokens must be positive, got %d",
                     max_new_tokens);
    return false;
  }
  return true;
}

std::string OfflineMossSatsModelConfig::ToString() const {
  std::ostringstream os;
  os << "OfflineMossSatsModelConfig(";
  os << "encoder=\"" << encoder << "\", ";
  os << "embedding=\"" << embedding << "\", ";
  os << "decoder=\"" << decoder << "\", ";
  os << "tokenizer=\"" << tokenizer << "\", ";
  os << "max_new_tokens=" << max_new_tokens << ", ";
  os << "max_total_len=" << max_total_len << ")";
  return os.str();
}

}  // namespace sherpa_onnx
