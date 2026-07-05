// sherpa-onnx/csrc/offline-tts-mbistft-stream-model.cc
//
// Copyright (c)  2026

#include "sherpa-onnx/csrc/offline-tts-mbistft-stream-model.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/ort-env.h"
#include "sherpa-onnx/csrc/session.h"

namespace sherpa_onnx {

class OfflineTtsMbistftStreamModel::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(CreateOrtEnv()),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    // config_.mbistft.enc / .dec — add OfflineTtsMbistftStreamModelConfig
    // (see STREAMING_INTEGRATION.md). enc: text->z ; dec: z-chunk->wav.
    enc_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.mbistft.enc), sess_opts_);
    dec_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.mbistft.dec), sess_opts_);
    GetInputNames(enc_.get(), &enc_in_, &enc_in_ptr_);
    GetOutputNames(enc_.get(), &enc_out_, &enc_out_ptr_);
    GetInputNames(dec_.get(), &dec_in_, &dec_in_ptr_);
    GetOutputNames(dec_.get(), &dec_out_, &dec_out_ptr_);
  }

  std::vector<float> Generate(const std::vector<int64_t> &x,
                              const std::vector<int64_t> &tone,
                              const std::vector<int64_t> &lang,
                              float noise_scale, float length_scale,
                              const Callback &callback) const {
    auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    const int64_t T = static_cast<int64_t>(x.size());
    std::array<int64_t, 2> tok_shape{1, T};
    std::array<int64_t, 1> s1{1};

    auto mk_tok = [&](const std::vector<int64_t> &v) {
      return Ort::Value::CreateTensor<int64_t>(
          mem, const_cast<int64_t *>(v.data()), v.size(), tok_shape.data(), 2);
    };
    int64_t len = T;
    float ns = noise_scale, ls = length_scale;

    std::vector<Ort::Value> enc_inputs;
    enc_inputs.reserve(6);
    enc_inputs.push_back(mk_tok(x));
    enc_inputs.push_back(mk_tok(tone));
    enc_inputs.push_back(mk_tok(lang));
    enc_inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, &len, 1, s1.data(), 1));
    enc_inputs.push_back(Ort::Value::CreateTensor<float>(mem, &ns, 1, s1.data(), 1));
    enc_inputs.push_back(Ort::Value::CreateTensor<float>(mem, &ls, 1, s1.data(), 1));

    auto z_out = enc_->Run({}, enc_in_ptr_.data(), enc_inputs.data(),
                           enc_inputs.size(), enc_out_ptr_.data(),
                           enc_out_ptr_.size());
    // z: [1, kChan, F]
    auto z_shape = z_out[0].GetTensorTypeAndShapeInfo().GetShape();
    const int32_t F = static_cast<int32_t>(z_shape[2]);
    const float *z = z_out[0].GetTensorData<float>();  // contiguous [C, F] (ne: F fastest)

    std::vector<float> out;
    out.reserve(static_cast<size_t>(F) * kHop);
    std::vector<float> slice;  // reused chunk buffer [C, e-s0]

    for (int32_t a = 0; a < F; a += kChunk) {
      int32_t b = std::min(a + kChunk, F);
      int32_t s0 = std::max(0, a - kLeft);
      int32_t e = std::min(F, b + kRight);
      int32_t w = e - s0;
      // gather z[:, :, s0:e] -> [1, C, w] (z is row-major [C, F])
      slice.resize(static_cast<size_t>(kChan) * w);
      for (int32_t c = 0; c < kChan; ++c) {
        std::copy(z + static_cast<size_t>(c) * F + s0,
                  z + static_cast<size_t>(c) * F + e,
                  slice.data() + static_cast<size_t>(c) * w);
      }
      std::array<int64_t, 3> zc_shape{1, kChan, w};
      Ort::Value zc = Ort::Value::CreateTensor<float>(
          mem, slice.data(), slice.size(), zc_shape.data(), 3);
      auto wav = dec_->Run({}, dec_in_ptr_.data(), &zc, 1, dec_out_ptr_.data(),
                           dec_out_ptr_.size());
      const float *wp = wav[0].GetTensorData<float>();
      int32_t off = (a - s0) * kHop;
      int32_t keep = (b - a) * kHop;
      out.insert(out.end(), wp + off, wp + off + keep);
      if (callback) {
        if (!callback(wp + off, keep, static_cast<float>(b) / F)) break;
      }
    }
    return out;
  }

 private:
  OfflineTtsModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::unique_ptr<Ort::Session> enc_, dec_;
  std::vector<std::string> enc_in_, enc_out_, dec_in_, dec_out_;
  std::vector<const char *> enc_in_ptr_, enc_out_ptr_, dec_in_ptr_, dec_out_ptr_;
};

OfflineTtsMbistftStreamModel::OfflineTtsMbistftStreamModel(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineTtsMbistftStreamModel::~OfflineTtsMbistftStreamModel() = default;

std::vector<float> OfflineTtsMbistftStreamModel::Generate(
    const std::vector<int64_t> &x, const std::vector<int64_t> &tone,
    const std::vector<int64_t> &lang, float noise_scale, float length_scale,
    const Callback &callback) const {
  return impl_->Generate(x, tone, lang, noise_scale, length_scale, callback);
}

}  // namespace sherpa_onnx
