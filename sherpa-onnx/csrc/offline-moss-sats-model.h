// sherpa-onnx/csrc/offline-moss-sats-model.h
//
// Copyright (c)  2026  zengyw

#ifndef SHERPA_ONNX_CSRC_OFFLINE_MOSS_SATS_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_MOSS_SATS_MODEL_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-moss-sats-model-config.h"

namespace sherpa_onnx {

class OfflineMossSatsModel {
 public:
  explicit OfflineMossSatsModel(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineMossSatsModel(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineMossSatsModel();

  /** Run the encoder (mel -> audio embeddings).
   *
   * @param mel Tensor (B, num_mel_bins, T), float32.
   * @return audio embeddings (B, T', hidden_size)
   */
  Ort::Value ForwardEncoder(Ort::Value mel);

  /** Run the LLM model (KV cache mode).
   *
   * @param input_ids  A tensor of shape (N, T) containing token IDs, int64.
   * @param audio_features  A tensor of shape (N, A, hidden_size) containing
   * audio embeddings, float32.
   * @param cache_position  A tensor of shape (T,) containing cache positions,
   * int64.
   * @param cache_kv  Fixed-size KV cache, vector of (key, value) pairs.
   * @return Return tuple (logits, kv_outputs...). Logits shape (N, T,
   * vocab_size), float32. kv_outputs is a vector of (key_delta, value_delta)
   * pairs for each layer.
   */
  std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
  ForwardLLM(Ort::Value input_ids, Ort::Value audio_features,
             const Ort::Value &cache_position,
             const std::vector<std::pair<Ort::Value, Ort::Value>> &cache_kv);

  /** Create fixed-size KV cache buffer.
   *
   * @param batch  Batch size (usually 1).
   * @return Return vector of (key, value) pairs with fixed cache dimensions [B,
   * max_total_len, kv_h, hd].
   */
  std::vector<std::pair<Ort::Value, Ort::Value>> CreateEmptyKVCache(
      int64_t batch);

  /** Apply KV delta in-place to KV cache buffer.
   *
   * @param cache_kv  Fixed-size KV cache to update, vector of (key, value)
   * pairs.
   * @param kv_delta  KV deltas from current step, vector of (key_delta,
   * value_delta) pairs.
   * @param cache_position  Cache position tensor indicating where to write
   * deltas.
   */
  void ApplyKvDeltaInplace(
      std::vector<std::pair<Ort::Value, Ort::Value>> *cache_kv,
      const std::vector<std::pair<Ort::Value, Ort::Value>> &kv_delta,
      const Ort::Value &cache_position);

  /** Return the maximum total sequence length (from metadata or config)
   */
  int32_t GetMaxTotalLen() const;

  /** Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_MOSS_SATS_MODEL_H_
