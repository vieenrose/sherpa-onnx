// sherpa-onnx/csrc/offline-tts-mbistft-stream-model.h
//
// Streaming PrimeTTS v2-Stream (MB-iSTFT-VITS, token-level band-attn encoder +
// causal vocoder). Two ONNX graphs:
//   enc : (x,tone,lang,x_lengths,noise_scale,length_scale) -> z[1,192,T]   (once)
//   dec : z[1,192,Tc] -> wav[1,1,Tc*256]                                   (per chunk)
// Streaming = enc once, then decode z in 24-frame chunks via overlap-save
// (decode z[:, :, a-LEFT : b+RIGHT], keep the middle (b-a)*256 samples), firing
// the audio callback per chunk. Bit-exact vs whole-utterance (validated: cos
// 1.000000, first-chunk ~15-30ms CPU). See streaming/onnx_stream.py (reference).
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_MBISTFT_STREAM_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_MBISTFT_STREAM_MODEL_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

class OfflineTtsMbistftStreamModel {
 public:
  ~OfflineTtsMbistftStreamModel();
  explicit OfflineTtsMbistftStreamModel(const OfflineTtsModelConfig &config);

  // Callback: (samples, n, progress) -> keep going? Mirrors GeneratedAudioCallback.
  using Callback = std::function<bool(const float *, int32_t, float)>;

  // token ids are blank-interleaved by the frontend (add_blank convention).
  // Runs enc once, streams dec chunk-by-chunk, returns the full waveform.
  std::vector<float> Generate(const std::vector<int64_t> &x,
                              const std::vector<int64_t> &tone,
                              const std::vector<int64_t> &lang,
                              float noise_scale, float length_scale,
                              const Callback &callback) const;

  int32_t SampleRate() const { return 16000; }

  // streaming params (validated on the trained model)
  static constexpr int32_t kChunk = 24;
  static constexpr int32_t kLeft = 64;
  static constexpr int32_t kRight = 4;
  static constexpr int32_t kHop = 256;  // samples per frame
  static constexpr int32_t kChan = 192;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_MBISTFT_STREAM_MODEL_H_
