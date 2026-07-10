// sherpa-onnx/csrc/offline-moss-sats-model-config.h
//
// Config for MOSS-Transcribe-Diarize: Speaker-Attributed, Time-Stamped
// Transcription (SATS). See https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize
//
// Copyright (c)  2026

#ifndef SHERPA_ONNX_CSRC_OFFLINE_MOSS_SATS_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_MOSS_SATS_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineMossSatsModelConfig {
  // Whisper-Medium encoder + 4x time merge + VQAdaptor, folded into one graph.
  // Input: mel (B, num_mel_bins, T). Output: audio embeddings (B, T/8, hidden).
  std::string encoder;

  // Qwen3 token-embedding lookup: input_ids (B, S) -> embeds (B, S, hidden).
  std::string embedding;

  // Qwen3-0.6B decoder with KV cache. Inputs: inputs_embeds, attention_mask,
  // past_k_i/past_v_i. Outputs: logits, present_k_i/present_v_i.
  std::string decoder;

  // Directory holding the HF tokenizer assets (tokenizer.json or
  // vocab.json + merges.txt) for the Qwen3 BPE.
  std::string tokenizer;

  // Maximum number of new tokens to generate per window.
  int32_t max_new_tokens = 2048;

  // Fixed KV-cache length fallback when the decoder's past_key dim1 is
  // dynamic (normally read from the graph; our export bakes 8192).
  int32_t max_total_len = 8192;

  // Optional hotwords (comma separated), injected into the prompt.
  std::string hotwords;

  // Sampling (0 temperature = greedy).
  float temperature = 0.0f;
  float top_p = 1.0f;
  int32_t seed = 0;

  OfflineMossSatsModelConfig() = default;
  OfflineMossSatsModelConfig(const std::string &encoder,
                             const std::string &embedding,
                             const std::string &decoder,
                             const std::string &tokenizer,
                             int32_t max_new_tokens, int32_t max_total_len)
      : encoder(encoder),
        embedding(embedding),
        decoder(decoder),
        tokenizer(tokenizer),
        max_new_tokens(max_new_tokens),
        max_total_len(max_total_len) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_MOSS_SATS_MODEL_CONFIG_H_
