// sherpa-onnx/csrc/offline-recognizer-moss-sats-impl.cc
//
// Copyright (c)  2026 zengyw

#include "sherpa-onnx/csrc/offline-recognizer-moss-sats-impl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-moss-sats-parser.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

// Mel-frame chunk length (in frames) assumed by the MOSS-SATS conv frontend
// when mapping log-mel features to audio tokens. Must match the chunk size
// baked into the exported ONNX graph; used by FeatToAudioTokensLen() to size
// the encoder mask.
constexpr int32_t kMossSatsChunkSize = 100;
// Number of mel bins per frame for MOSS-SATS (Whisper-style log-mel). Must
// match the feature extractor (`WhisperTag` dim), `NormalizeWhisperFeatures`
// row width, and the last dimension of the conv-frontend ONNX input.
constexpr int32_t kMossSatsMelDim = 80;  // Whisper-Medium mel bins

// MOSS-SATS chat template: fixed system prompt; the transcription instruction
// (plus optional hotwords) follows the audio inside the user turn.
constexpr char kMossSatsSystemPromptPrefix[] =
"<|im_start|>system\nYou are a helpful assistant.";
constexpr char kMossSatsSystemPromptSuffix[] =
"<|im_end|>\n<|im_start|>user\n<|audio_start|>";
// zh: request SATS transcription ([start][Sxx]text[end] per segment).
constexpr char kMossSatsInstruction[] =
"\xe8\xaf\xb7\xe5\xb0\x86\xe9\x9f\xb3\xe9\xa2\x91\xe8\xbd\xac\xe5"
"\x86\x99\xe4\xb8\xba\xe6\x96\x87\xe6\x9c\xac\xef\xbc\x8c\xe6\xaf"
"\x8f\xe4\xb8\x80\xe6\xae\xb5\xe9\x9c\x80\xe4\xbb\xa5\xe8\xb5\xb7"
"\xe5\xa7\x8b\xe6\x97\xb6\xe9\x97\xb4\xe6\x88\xb3\xe5\x92\x8c\xe8"
"\xaf\xb4\xe8\xaf\x9d\xe4\xba\xba\xe7\xbc\x96\xe5\x8f\xb7\xef\xbc"
"\x88[S01]\xe3\x80\x81[S02]\xe3\x80\x81[S03]\xe2\x80\xa6\xef\xbc"
"\x89\xe5\xbc\x80\xe5\xa4\xb4\xef\xbc\x8c\xe6\xad\xa3\xe6\x96\x87"
"\xe4\xb8\xba\xe5\xaf\xb9\xe5\xba\x94\xe7\x9a\x84\xe8\xaf\xad\xe9"
"\x9f\xb3\xe5\x86\x85\xe5\xae\xb9\xef\xbc\x8c\xe5\xb9\xb6\xe5\x9c"
"\xa8\xe6\xae\xb5\xe6\x9c\xab\xe6\xa0\x87\xe6\xb3\xa8\xe7\xbb\x93"
"\xe6\x9d\x9f\xe6\x97\xb6\xe9\x97\xb4\xe6\x88\xb3\xef\xbc\x8c\xe4"
"\xbb\xa5\xe6\xb8\x85\xe6\x99\xb0\xe6\xa0\x87\xe6\x98\x8e\xe8\xaf"
"\xa5\xe6\xae\xb5\xe8\xaf\xad\xe9\x9f\xb3\xe8\x8c\x83\xe5\x9b\xb4"
"\xe3\x80\x82";

// Format hotwords for the Qwen3 chat template: ASCII comma-separated list
// (e.g. "foo,bar,baz");
static std::string Qwen3FormatHotwordsForPrompt(const std::string &csv) {
  const std::vector<std::string> parts = SplitStringAndTrim(csv, ',');
  return Join(parts, " ");
}

static void Qwen3LogMaxTotalLenSuggestions(int32_t max_seq_len,
                                           int32_t model_max_len) {
  SHERPA_ONNX_LOGE(
      "The max_total_len (%d) caps prompt + audio KV (model limit %d). "
      "Suggestions:",
      max_seq_len, model_max_len);
  SHERPA_ONNX_LOGE(
      "  1) Reduce hotwords: fewer or shorter hotwords shorten the prompt.");
  SHERPA_ONNX_LOGE(
      "  2) Shorten audio: shorter clips yield fewer audio_token_len.");
  SHERPA_ONNX_LOGE(
      "  3) Re-export the MOSS-SATS decoder ONNX with a larger max_total_len, "
      "raise --qwen3-asr-max-total-len (up to the model limit), and/or "
      "increase --qwen3-asr-max-new-tokens if generation is truncated.");
}

int32_t FeatToAudioTokensLen(int32_t feat_len, int32_t /*chunk_size*/) {
  // MOSS-SATS: Whisper encoder halves the mel frames (conv stride 2), then a
  // 4x time-merge: tokens = (T_mel / 2) / 4.  (30 s = 3000 mel -> 375 tokens.)
  if (feat_len <= 0) {
    return 0;
  }
  return (feat_len / 2) / 4;
}

inline bool IsFloatOrHalfBitsTensorType(ONNXTensorElementDataType elem_type) {
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
}

inline float ReadFloatOrHalfBitsValue(const float *data_f32,
                                      const uint16_t *data_f16_bits,
                                      ONNXTensorElementDataType elem_type,
                                      int64_t index) {
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return data_f32[index];
  }

  return HalfBitsToFloat(data_f16_bits[index]);
}

Ort::Value TrimAudioFeatures(Ort::Value audio_features,
                             OrtAllocator *allocator) {
  auto info = audio_features.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  if (shape.size() != 3 || shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0) {
    return audio_features;
  }

  auto elem_type =
      static_cast<ONNXTensorElementDataType>(info.GetElementType());
  if (!IsFloatOrHalfBitsTensorType(elem_type)) {
    return audio_features;
  }

  const int32_t A = static_cast<int32_t>(shape[1]);
  const int32_t H = static_cast<int32_t>(shape[2]);

  const float *data_f32 = nullptr;
  const uint16_t *data_f16_bits = nullptr;
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    data_f32 = audio_features.GetTensorData<float>();
  } else {
    data_f16_bits = audio_features.GetTensorData<uint16_t>();
  }

  int32_t A_valid = 0;
  const float eps = 1e-6f;

  for (int32_t a = A - 1; a >= 0; --a) {
    float max_energy = 0.0f;
    for (int32_t h = 0; h < H; ++h) {
      float v = ReadFloatOrHalfBitsValue(data_f32, data_f16_bits, elem_type,
                                         static_cast<int64_t>(a) * H + h);
      float abs_val = std::abs(v);
      if (abs_val > max_energy) {
        max_energy = abs_val;
      }
    }

    if (max_energy > eps) {
      A_valid = a + 1;
      break;
    }
  }

  if (A_valid <= 0) {
    return audio_features;
  }

  if (A_valid == A) {
    return audio_features;
  }

  std::array<int64_t, 3> new_shape{1, static_cast<int64_t>(A_valid), H};
  Ort::Value trimmed = Ort::Value::CreateTensor(allocator, new_shape.data(),
                                                new_shape.size(), elem_type);

  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const float *src = audio_features.GetTensorData<float>();
    float *dst = trimmed.GetTensorMutableData<float>();
    std::memcpy(
        dst, src,
        static_cast<size_t>(A_valid) * static_cast<size_t>(H) * sizeof(float));
  } else {
    const uint16_t *src = audio_features.GetTensorData<uint16_t>();
    uint16_t *dst = trimmed.GetTensorMutableData<uint16_t>();
    std::memcpy(dst, src,
                static_cast<size_t>(A_valid) * static_cast<size_t>(H) *
                    sizeof(uint16_t));
  }

  return trimmed;
}

Ort::Value TruncateAudioFeatures(Ort::Value audio_features, int32_t keep_frames,
                                 OrtAllocator *allocator) {
  if (keep_frames <= 0) {
    return audio_features;
  }

  auto info = audio_features.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  if (shape.size() != 3 || shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0) {
    return audio_features;
  }

  int32_t A = static_cast<int32_t>(shape[1]);
  int32_t H = static_cast<int32_t>(shape[2]);
  if (keep_frames >= A) {
    return audio_features;
  }

  auto elem_type =
      static_cast<ONNXTensorElementDataType>(info.GetElementType());
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
    return audio_features;
  }

  std::array<int64_t, 3> new_shape{1, static_cast<int64_t>(keep_frames), H};
  Ort::Value truncated = Ort::Value::CreateTensor(
      allocator, new_shape.data(), new_shape.size(),
      static_cast<ONNXTensorElementDataType>(elem_type));

  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const float *src = audio_features.GetTensorData<float>();
    float *dst = truncated.GetTensorMutableData<float>();
    std::memcpy(dst, src,
                static_cast<size_t>(keep_frames) * static_cast<size_t>(H) *
                    sizeof(float));
  } else {
    const uint16_t *src = audio_features.GetTensorData<uint16_t>();
    uint16_t *dst = truncated.GetTensorMutableData<uint16_t>();
    std::memcpy(dst, src,
                static_cast<size_t>(keep_frames) * static_cast<size_t>(H) *
                    sizeof(uint16_t));
  }

  return truncated;
}

Ort::Value BuildCachePosition(OrtAllocator *allocator, int32_t seq_len) {
  std::array<int64_t, 1> pos_shape{seq_len};
  Ort::Value cache_position = Ort::Value::CreateTensor<int64_t>(
      allocator, pos_shape.data(), pos_shape.size());

  int64_t *p = cache_position.GetTensorMutableData<int64_t>();
  std::iota(p, p + seq_len, int64_t{0});

  return cache_position;
}

inline float TensorAbsMax(const Ort::Value &t, int64_t limit) {
  auto info = t.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();

  int64_t n = 1;
  for (auto d : shape) {
    if (d <= 0) {
      return 0.0f;
    }
    if (n > (std::numeric_limits<int64_t>::max() / d)) {
      return 0.0f;
    }
    n *= d;
  }

  if (limit > 0 && n > limit) {
    n = limit;
  }

  auto elem_type =
      static_cast<ONNXTensorElementDataType>(info.GetElementType());
  if (!IsFloatOrHalfBitsTensorType(elem_type)) {
    return 0.0f;
  }

  const float *data_f32 = nullptr;
  const uint16_t *data_f16_bits = nullptr;
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    data_f32 = t.GetTensorData<float>();
  } else {
    data_f16_bits = t.GetTensorData<uint16_t>();
  }

  float abs_max = 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    float v = std::abs(
        ReadFloatOrHalfBitsValue(data_f32, data_f16_bits, elem_type, i));
    if (std::isfinite(v) && v > abs_max) {
      abs_max = v;
    }
  }

  return abs_max;
}

inline void RemoveUtf8ReplacementChars(std::string *s) {
  if (!s || s->empty()) {
    return;
  }

  const std::string kReplacement = "\xEF\xBF\xBD";
  size_t pos = 0;
  while ((pos = s->find(kReplacement, pos)) != std::string::npos) {
    s->erase(pos, kReplacement.size());
  }
}

}  // namespace

OfflineRecognizerMossSatsImpl::OfflineRecognizerMossSatsImpl(
    const OfflineRecognizerConfig &config)
    : OfflineRecognizerImpl(config),
      config_(config),
      model_(std::make_unique<OfflineMossSatsModel>(config.model_config)),
      tokenizer_(std::make_unique<QwenAsrTokenizer>(
          config.model_config.moss_sats.tokenizer)),
      rng_(config.model_config.moss_sats.seed) {
  InitPromptTemplateIds();
}

template <typename Manager>
OfflineRecognizerMossSatsImpl::OfflineRecognizerMossSatsImpl(
    Manager *mgr, const OfflineRecognizerConfig &config)
    : OfflineRecognizerImpl(mgr, config),
      config_(config),
      model_(std::make_unique<OfflineMossSatsModel>(mgr, config.model_config)),
      tokenizer_(std::make_unique<QwenAsrTokenizer>(
          mgr, config.model_config.moss_sats.tokenizer)),
      rng_(config.model_config.moss_sats.seed) {
  InitPromptTemplateIds();
}

std::unique_ptr<OfflineStream> OfflineRecognizerMossSatsImpl::CreateStream()
    const {
  return std::make_unique<OfflineStream>(WhisperTag{kMossSatsMelDim});
}

void OfflineRecognizerMossSatsImpl::InitPromptTemplateIds() {
  const std::string audio_pad = "<|audio_pad|>";
  const std::string user_suffix = std::string("<|audio_end|>\n") +
                                  kMossSatsInstruction + "<|im_end|>\n";
  const std::string assistant_text = "<|im_start|>assistant\n";

  audio_pad_ids_ = tokenizer_->Encode(audio_pad);
  prompt_ids_after_ = tokenizer_->Encode(user_suffix + assistant_text);

  if (audio_pad_ids_.empty()) {
    SHERPA_ONNX_LOGE("Failed to tokenize <|audio_pad|> for moss-sats prompt");
    SHERPA_ONNX_EXIT(-1);
  }

  asr_text_token_id_ = -1;  // MOSS-SATS has no <asr_text> marker
}

std::vector<int64_t> OfflineRecognizerMossSatsImpl::BuildSourceIds(
    const std::string &hotwords, const std::string &language,
    int32_t audio_token_len, int32_t *before_len,
    int32_t *fake_audio_token_len) const {
  const std::string before_utf8 = std::string(kMossSatsSystemPromptPrefix) +
                                  hotwords + kMossSatsSystemPromptSuffix;
  std::vector<int64_t> prompt_ids_before = tokenizer_->Encode(before_utf8);

  if (before_len) {
    *before_len = static_cast<int32_t>(prompt_ids_before.size());
  }
  if (fake_audio_token_len) {
    *fake_audio_token_len = audio_token_len;
  }

  std::vector<int64_t> prompt_ids_after_with_language;
  const std::vector<int64_t> *ids_after = &prompt_ids_after_;
  if (!language.empty()) {
    auto language_ids = tokenizer_->Encode("language " + language);
    prompt_ids_after_with_language.reserve(prompt_ids_after_.size() +
                                           language_ids.size() + 1);
    prompt_ids_after_with_language.insert(prompt_ids_after_with_language.end(),
                                          prompt_ids_after_.begin(),
                                          prompt_ids_after_.end());
    prompt_ids_after_with_language.insert(prompt_ids_after_with_language.end(),
                                          language_ids.begin(),
                                          language_ids.end());
    ids_after = &prompt_ids_after_with_language;
  }

  std::vector<int64_t> source_ids;
  size_t estimated_size =
      prompt_ids_before.size() +
      static_cast<size_t>(audio_token_len) * audio_pad_ids_.size() +
      ids_after->size();
  source_ids.reserve(estimated_size);
  source_ids.insert(source_ids.end(), prompt_ids_before.begin(),
                    prompt_ids_before.end());

  for (int32_t i = 0; i < audio_token_len; ++i) {
    source_ids.insert(source_ids.end(), audio_pad_ids_.begin(),
                      audio_pad_ids_.end());
  }

  source_ids.insert(source_ids.end(), ids_after->begin(), ids_after->end());

  return source_ids;
}

int64_t OfflineRecognizerMossSatsImpl::SampleTokenFromLogitsFp16OrFp32(
    const void *logits, bool is_fp16, int32_t vocab_size) const {
  if (!logits || vocab_size <= 0) {
    return 0;
  }

  int32_t best = 0;
  float best_val = -std::numeric_limits<float>::infinity();
  bool found_valid = false;

  if (is_fp16) {
    const uint16_t *p = reinterpret_cast<const uint16_t *>(logits);
    for (int32_t i = 0; i < vocab_size; ++i) {
      float v = HalfBitsToFloat(p[i]);
      if (std::isfinite(v) && v > best_val) {
        best_val = v;
        best = i;
        found_valid = true;
      }
    }
  } else {
    const float *p = reinterpret_cast<const float *>(logits);
    for (int32_t i = 0; i < vocab_size; ++i) {
      float v = p[i];
      if (std::isfinite(v) && v > best_val) {
        best_val = v;
        best = i;
        found_valid = true;
      }
    }
  }

  return found_valid ? best : 0;
}

int64_t OfflineRecognizerMossSatsImpl::SampleTokenFromLogits(
    const Ort::Value &logits, int32_t time_index, float temperature,
    float top_p) const {
  auto info = logits.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  if (shape.size() < 3 || shape[1] <= 0 || shape[2] <= 0 || time_index < 0) {
    return 0;
  }

  const int32_t time_dim = static_cast<int32_t>(shape[1]);
  if (time_index >= time_dim) {
    return 0;
  }

  const int32_t vocab_size = static_cast<int32_t>(shape[2]);
  auto elem_type =
      static_cast<ONNXTensorElementDataType>(info.GetElementType());

  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
    return 0;
  }

  const bool is_fp16 = (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
                        elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);

  const void *base =
      is_fp16 ? static_cast<const void *>(logits.GetTensorData<uint16_t>())
              : static_cast<const void *>(logits.GetTensorData<float>());

  const size_t offset = static_cast<size_t>(time_index) * vocab_size;
  const void *row = is_fp16
                        ? static_cast<const void *>(
                              reinterpret_cast<const uint16_t *>(base) + offset)
                        : static_cast<const void *>(
                              reinterpret_cast<const float *>(base) + offset);

  return SampleTokenWithTemperatureAndTopP(row, is_fp16, vocab_size,
                                           temperature, top_p);
}

int64_t OfflineRecognizerMossSatsImpl::SampleTokenWithTemperatureAndTopP(
    const void *logits, bool is_fp16, int32_t vocab_size, float temperature,
    float top_p, int64_t avoid_id) const {
  if (!logits || vocab_size <= 0) {
    return 0;
  }

  if (temperature <= 1e-6f) {
    int32_t best = 0;
    float best_val = -std::numeric_limits<float>::infinity();
    bool found_valid = false;

    if (is_fp16) {
      const uint16_t *p = reinterpret_cast<const uint16_t *>(logits);
      for (int32_t i = 0; i < vocab_size; ++i) {
        if (avoid_id >= 0 && i == avoid_id) {
          continue;
        }
        float v = HalfBitsToFloat(p[i]);
        if (std::isfinite(v) && v > best_val) {
          best_val = v;
          best = i;
          found_valid = true;
        }
      }
    } else {
      const float *p = reinterpret_cast<const float *>(logits);
      for (int32_t i = 0; i < vocab_size; ++i) {
        if (avoid_id >= 0 && i == avoid_id) {
          continue;
        }
        float v = p[i];
        if (std::isfinite(v) && v > best_val) {
          best_val = v;
          best = i;
          found_valid = true;
        }
      }
    }

    return found_valid ? best : 0;
  }

  std::vector<float> probs(vocab_size, 0.0f);
  float max_logit = -std::numeric_limits<float>::infinity();

  if (is_fp16) {
    const uint16_t *p = reinterpret_cast<const uint16_t *>(logits);
    for (int32_t i = 0; i < vocab_size; ++i) {
      if (avoid_id >= 0 && i == avoid_id) {
        probs[i] = -std::numeric_limits<float>::infinity();
        continue;
      }

      float v = HalfBitsToFloat(p[i]);
      if (!std::isfinite(v)) {
        probs[i] = -std::numeric_limits<float>::infinity();
        continue;
      }

      probs[i] = v / temperature;
      if (probs[i] > max_logit) {
        max_logit = probs[i];
      }
    }
  } else {
    const float *p = reinterpret_cast<const float *>(logits);
    for (int32_t i = 0; i < vocab_size; ++i) {
      if (avoid_id >= 0 && i == avoid_id) {
        probs[i] = -std::numeric_limits<float>::infinity();
        continue;
      }

      float v = p[i];
      if (!std::isfinite(v)) {
        probs[i] = -std::numeric_limits<float>::infinity();
        continue;
      }

      probs[i] = v / temperature;
      if (probs[i] > max_logit) {
        max_logit = probs[i];
      }
    }
  }

  if (!std::isfinite(max_logit)) {
    return SampleTokenFromLogitsFp16OrFp32(logits, is_fp16, vocab_size);
  }

  float sum = 0.0f;
  for (int32_t i = 0; i < vocab_size; ++i) {
    if (!std::isfinite(probs[i])) {
      probs[i] = 0.0f;
      continue;
    }

    probs[i] = std::exp(probs[i] - max_logit);
    sum += probs[i];
  }

  if (sum <= 0.0f) {
    return SampleTokenFromLogitsFp16OrFp32(logits, is_fp16, vocab_size);
  }

  if (top_p < 1.0f - 1e-6f) {
    std::vector<std::pair<int32_t, float>> prob_idx;
    prob_idx.reserve(vocab_size);

    for (int32_t i = 0; i < vocab_size; ++i) {
      if (probs[i] > 0.0f) {
        prob_idx.push_back({i, probs[i]});
      }
    }

    if (prob_idx.empty()) {
      return SampleTokenFromLogitsFp16OrFp32(logits, is_fp16, vocab_size);
    }

    std::sort(
        prob_idx.begin(), prob_idx.end(),
        [](const std::pair<int32_t, float> &a,
           const std::pair<int32_t, float> &b) { return a.second > b.second; });

    float kept_sum = 0.0f;
    int32_t cutoff = static_cast<int32_t>(prob_idx.size());
    for (int32_t i = 0; i < static_cast<int32_t>(prob_idx.size()); ++i) {
      kept_sum += prob_idx[i].second;
      if (kept_sum / sum >= top_p) {
        cutoff = i + 1;
        break;
      }
    }

    if (cutoff <= 0) {
      return prob_idx[0].first;
    }

    kept_sum = 0.0f;
    for (int32_t i = 0; i < cutoff; ++i) {
      kept_sum += prob_idx[i].second;
    }

    if (kept_sum <= 0.0f) {
      return prob_idx[0].first;
    }

    float r = std::uniform_real_distribution<float>(0.0f, kept_sum)(rng_);
    float cumsum = 0.0f;
    for (int32_t i = 0; i < cutoff; ++i) {
      cumsum += prob_idx[i].second;
      if (r <= cumsum) {
        return prob_idx[i].first;
      }
    }

    return prob_idx[cutoff - 1].first;
  }

  float r = std::uniform_real_distribution<float>(0.0f, sum)(rng_);
  float cumsum = 0.0f;
  for (int32_t i = 0; i < vocab_size; ++i) {
    cumsum += probs[i];
    if (r <= cumsum) {
      return i;
    }
  }

  return vocab_size - 1;
}

OfflineRecognitionResult OfflineRecognizerMossSatsImpl::GenerateText(
    Ort::Value audio_features, int32_t audio_token_len,
    OfflineStream *stream) const {
  OfflineRecognitionResult result;
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  const auto &qwen3_config = config_.model_config.moss_sats;

  int32_t max_new_tokens =
      stream->GetOptionInt("max_new_tokens", qwen3_config.max_new_tokens);
  if (max_new_tokens <= 0) {
    max_new_tokens = qwen3_config.max_new_tokens;
  }

  const float temperature =
      stream->GetOptionFloat("temperature", qwen3_config.temperature);
  const float top_p = stream->GetOptionFloat("top_p", qwen3_config.top_p);

  Ort::Value trimmed_audio_features =
      TrimAudioFeatures(std::move(audio_features), model_->Allocator());

  auto trimmed_shape =
      trimmed_audio_features.GetTensorTypeAndShapeInfo().GetShape();
  if (trimmed_shape.size() == 3 && trimmed_shape[1] > 0) {
    audio_token_len = std::min<int32_t>(audio_token_len,
                                        static_cast<int32_t>(trimmed_shape[1]));
  }

  if (config_.model_config.debug) {
    float abs_max = TensorAbsMax(trimmed_audio_features, 1LL << 20);
    SHERPA_ONNX_LOGE(
        "qwen3-asr: audio_features shape=[%d,%d,%d] abs_max=%f "
        "audio_token_len=%d",
        static_cast<int32_t>(trimmed_shape.size() > 0 ? trimmed_shape[0] : -1),
        static_cast<int32_t>(trimmed_shape.size() > 1 ? trimmed_shape[1] : -1),
        static_cast<int32_t>(trimmed_shape.size() > 2 ? trimmed_shape[2] : -1),
        abs_max, audio_token_len);
  }

  if (audio_token_len <= 0) {
    result.text = "";
    return result;
  }

  // Optional per-stream hotwords via SetOption("hotwords", comma-separated
  // CSV).
  const std::string hotwords = Qwen3FormatHotwordsForPrompt(
      stream->HasOption("hotwords") ? stream->GetOption("hotwords")
                                    : qwen3_config.hotwords);

  std::string language;
  if (stream->HasOption("language")) {
    language = stream->GetOption("language");
  }

  int32_t before_len = 0;
  int32_t fake_audio_token_len = 0;
  std::vector<int64_t> source_ids = BuildSourceIds(
      hotwords, language, audio_token_len, &before_len, &fake_audio_token_len);

  int32_t context_len = static_cast<int32_t>(source_ids.size());
  if (context_len == 0) {
    result.text = "";
    return result;
  }

  std::vector<std::pair<Ort::Value, Ort::Value>> cache_kv =
      model_->CreateEmptyKVCache(1);
  const int32_t model_max_len = model_->GetMaxTotalLen();
  int32_t max_seq_len = model_max_len;
  const int32_t max_total_len_opt =
      stream->GetOptionInt("max_total_len", qwen3_config.max_total_len);
  if (max_total_len_opt > 0) {
    max_seq_len = std::min(model_max_len, max_total_len_opt);
  }

  if (!hotwords.empty()) {
    const std::string scaffold_no_hw =
        std::string(kMossSatsSystemPromptPrefix) + kMossSatsSystemPromptSuffix;
    const std::vector<int64_t> base_ids = tokenizer_->Encode(scaffold_no_hw);
    const int32_t base_before = static_cast<int32_t>(base_ids.size());
    const int32_t hotword_tokens = std::max(0, before_len - base_before);

    const int32_t tail_len = static_cast<int32_t>(prompt_ids_after_.size());
    const int32_t one_audio_len = static_cast<int32_t>(audio_pad_ids_.size());
    const int32_t room = max_seq_len - before_len - tail_len;
    const bool tight = hotword_tokens >= 48 ||
                       (one_audio_len > 0 && room < one_audio_len * 32);

    if (config_.model_config.debug || tight) {
      SHERPA_ONNX_LOGE(
          "qwen3-asr: hotwords add %d tokenizer tokens in the prompt head "
          "(before_audio=%d, scaffold_without_hotwords=%d).",
          hotword_tokens, before_len, base_before);
    }
    if (tight) {
      Qwen3LogMaxTotalLenSuggestions(max_seq_len, model_max_len);
    }
  }

  if (context_len > max_seq_len) {
    const int32_t one_audio_len = static_cast<int32_t>(audio_pad_ids_.size());
    if (one_audio_len <= 0) {
      result.text = "";
      return result;
    }

    int32_t after_len =
        context_len - before_len - fake_audio_token_len * one_audio_len;
    if (after_len < 0) {
      after_len = 0;
    }

    int32_t keep_audio = (max_seq_len - before_len - after_len) / one_audio_len;
    if (keep_audio < 0) {
      SHERPA_ONNX_LOGE(
          "qwen3-asr prompt scaffold exceeds max_total_len: before=%d after=%d "
          "max_total_len=%d",
          before_len, after_len, max_seq_len);
      Qwen3LogMaxTotalLenSuggestions(max_seq_len, model_max_len);
      result.text = "";
      return result;
    }

    if (keep_audio == 0) {
      SHERPA_ONNX_LOGE(
          "qwen3-asr max_total_len=%d leaves no room for audio placeholders "
          "(before=%d after=%d)",
          max_seq_len, before_len, after_len);
      Qwen3LogMaxTotalLenSuggestions(max_seq_len, model_max_len);
      result.text = "";
      return result;
    }

    if (keep_audio < fake_audio_token_len) {
      SHERPA_ONNX_LOGE(
          "qwen3-asr: context_len (%d) exceeds max_total_len (%d). Truncating "
          "audio placeholders: audio_token_len=%d -> keep_audio=%d (before=%d "
          "after=%d).",
          context_len, max_seq_len, fake_audio_token_len, keep_audio,
          before_len, after_len);
      Qwen3LogMaxTotalLenSuggestions(max_seq_len, model_max_len);
      std::vector<int64_t> ids_before(source_ids.begin(),
                                      source_ids.begin() + before_len);
      std::vector<int64_t> ids_after(source_ids.end() - after_len,
                                     source_ids.end());

      source_ids.clear();
      source_ids.reserve(before_len + keep_audio * one_audio_len + after_len);
      source_ids.insert(source_ids.end(), ids_before.begin(), ids_before.end());

      for (int32_t i = 0; i < keep_audio; ++i) {
        source_ids.insert(source_ids.end(), audio_pad_ids_.begin(),
                          audio_pad_ids_.end());
      }

      source_ids.insert(source_ids.end(), ids_after.begin(), ids_after.end());

      fake_audio_token_len = keep_audio;
      audio_token_len = keep_audio;
      context_len = static_cast<int32_t>(source_ids.size());

      trimmed_audio_features = TruncateAudioFeatures(
          std::move(trimmed_audio_features), keep_audio, model_->Allocator());
    }
  }

  std::vector<int64_t> input_ids = source_ids;
  std::array<int64_t, 2> ids_shape{1, context_len};
  Ort::Value input_ids_tensor =
      Ort::Value::CreateTensor(memory_info, input_ids.data(), input_ids.size(),
                               ids_shape.data(), ids_shape.size());

  std::array<int64_t, 2> attn_mask_shape{1, context_len};
  std::vector<int64_t> attn_mask_vec(context_len, 1);
  Ort::Value attention_mask = Ort::Value::CreateTensor<int64_t>(
      memory_info, attn_mask_vec.data(), attn_mask_vec.size(),
      attn_mask_shape.data(), attn_mask_shape.size());

  Ort::Value cache_position =
      BuildCachePosition(model_->Allocator(), context_len);
  Ort::Value audio_features_view = View(&trimmed_audio_features);

  auto tmp = model_->ForwardLLM(
      std::move(input_ids_tensor), std::move(audio_features_view),
      std::move(attention_mask), cache_position, cache_kv);
  Ort::Value logits = std::move(tmp.first);
  auto kv_outputs = std::move(tmp.second);

  model_->ApplyKvDeltaInplace(&cache_kv, kv_outputs, cache_position);

  std::vector<int64_t> generated_ids;
  generated_ids.reserve(static_cast<size_t>(max_new_tokens));

  const int64_t eos_id = tokenizer_->GetEosTokenId();

  auto log_shape = logits.GetTensorTypeAndShapeInfo().GetShape();
  if (log_shape.size() < 3) {
    result.text = "";
    return result;
  }

  const int32_t time_dim = static_cast<int32_t>(log_shape[1]);
  const int32_t last_idx = context_len - 1;
  if (last_idx >= time_dim) {
    if (config_.model_config.debug) {
      SHERPA_ONNX_LOGE(
          "qwen3-asr: logits time_dim (%d) < context_len (%d); "
          "cannot sample first token",
          time_dim, context_len);
    }
    result.text = "";
    return result;
  }

  int64_t next_id = SampleTokenFromLogits(logits, last_idx, temperature, top_p);

  if (next_id == eos_id) {
    if (config_.model_config.debug) {
      float abs_max = TensorAbsMax(logits, 1LL << 20);
      SHERPA_ONNX_LOGE(
          "qwen3-asr: first token is EOS (eos_id=%d). logits_abs_max=%f "
          "context_len=%d max_total_len=%d",
          static_cast<int32_t>(eos_id), abs_max, context_len, max_seq_len);
    }

    const int32_t vocab_size = static_cast<int32_t>(log_shape[2]);
    auto elem_type = static_cast<ONNXTensorElementDataType>(
        logits.GetTensorTypeAndShapeInfo().GetElementType());
    const bool is_fp16 = (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
                          elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);

    const void *base =
        is_fp16 ? static_cast<const void *>(logits.GetTensorData<uint16_t>())
                : static_cast<const void *>(logits.GetTensorData<float>());

    const size_t offset = static_cast<size_t>(last_idx) * vocab_size;
    const void *row =
        is_fp16 ? static_cast<const void *>(
                      reinterpret_cast<const uint16_t *>(base) + offset)
                : static_cast<const void *>(
                      reinterpret_cast<const float *>(base) + offset);

    next_id = SampleTokenWithTemperatureAndTopP(row, is_fp16, vocab_size,
                                                temperature, top_p, eos_id);

    if (next_id == eos_id) {
      result.text = "";
      return result;
    }
  }

  generated_ids.push_back(next_id);
  int32_t cur_len = context_len;

  for (int32_t step = 1; step < max_new_tokens; ++step) {
    if (cur_len >= max_seq_len) {
      break;
    }

    if (step + 1 == max_new_tokens) {
      SHERPA_ONNX_LOGE(
          "Result is truncated. max_new_tokens %d is too small for "
          "this audio input. Please either use a shorter audio or use a "
          "larger max_new_tokens",
          max_new_tokens);
    }

    const int64_t last_token_id = next_id;
    std::vector<int64_t> one_id{last_token_id};
    std::array<int64_t, 2> one_shape{1, 1};
    Ort::Value one_tensor =
        Ort::Value::CreateTensor(memory_info, one_id.data(), one_id.size(),
                                 one_shape.data(), one_shape.size());

    std::array<int64_t, 2> mask_shape{1, 1};
    std::vector<int64_t> mask_vec(1, 1);
    Ort::Value next_attention_mask = Ort::Value::CreateTensor<int64_t>(
        memory_info, mask_vec.data(), mask_vec.size(), mask_shape.data(),
        mask_shape.size());

    std::array<int64_t, 1> cache_pos_shape{1};
    std::vector<int64_t> cache_pos_vec{static_cast<int64_t>(cur_len)};
    Ort::Value next_cache_position = Ort::Value::CreateTensor<int64_t>(
        memory_info, cache_pos_vec.data(), cache_pos_vec.size(),
        cache_pos_shape.data(), cache_pos_shape.size());

    Ort::Value audio_features_view2 = View(&trimmed_audio_features);

    auto tmp2 = model_->ForwardLLM(
        std::move(one_tensor), std::move(audio_features_view2),
        std::move(next_attention_mask), next_cache_position, cache_kv);
    logits = std::move(tmp2.first);
    auto kv_outputs2 = std::move(tmp2.second);

    model_->ApplyKvDeltaInplace(&cache_kv, kv_outputs2, next_cache_position);

    auto log_shape2 = logits.GetTensorTypeAndShapeInfo().GetShape();
    if (log_shape2.size() < 3) {
      break;
    }

    const int32_t time_dim2 = static_cast<int32_t>(log_shape2[1]);
    if (time_dim2 < 1) {
      break;
    }

    next_id = SampleTokenFromLogits(logits, time_dim2 - 1, temperature, top_p);

    if (next_id == eos_id) {
      break;
    }

    generated_ids.push_back(next_id);
    ++cur_len;
  }

  std::vector<int64_t> cleaned_ids = generated_ids;
  if (!generated_ids.empty()) {
    const size_t prefix_window = std::min<size_t>(16, generated_ids.size());
    auto asr_text_it =
        std::find(generated_ids.begin(), generated_ids.begin() + prefix_window,
                  asr_text_token_id_);

    // Only strip a leading scaffold prefix recognized by token ID.
    if (asr_text_it != generated_ids.begin() + prefix_window &&
        asr_text_it != generated_ids.begin()) {
      std::vector<int64_t> prefix_ids(generated_ids.begin(),
                                      std::next(asr_text_it));
      std::string prefix_text = tokenizer_->Decode(prefix_ids);
      if (prefix_text.rfind("language ", 0) == 0 && prefix_text.size() >= 10 &&
          prefix_text.compare(prefix_text.size() - 10, 10, "<asr_text>") == 0) {
        cleaned_ids.assign(std::next(asr_text_it), generated_ids.end());
      }
    }
  }

  result.text = tokenizer_->Decode(cleaned_ids);
  RemoveUtf8ReplacementChars(&result.text);

  // MOSS-SATS: parse the compact [start][Sxx]text[end] stream into
  // speaker-attributed, timestamped segments.
  {
    auto segments = MossSatsTranscriptParser::Parse(result.text);
    result.segment_timestamps.reserve(segments.size());
    result.segment_durations.reserve(segments.size());
    result.segment_texts.reserve(segments.size());
    for (const auto &seg : segments) {
      result.segment_timestamps.push_back(seg.start);
      result.segment_durations.push_back(seg.end - seg.start);
      result.segment_texts.push_back(seg.speaker + ": " + seg.text);
    }
  }

  if (!cleaned_ids.empty()) {
    std::vector<std::string> all_tokens;
    all_tokens.reserve(cleaned_ids.size());
    std::string pending_bytes;

    for (int64_t token_id : cleaned_ids) {
      std::string s =
          tokenizer_->GetTokenStringStreaming(token_id, &pending_bytes);
      all_tokens.push_back(std::move(s));
    }

    if (!pending_bytes.empty() && !all_tokens.empty()) {
      all_tokens.back().append("\xEF\xBF\xBD");
    }

    result.tokens = std::move(all_tokens);
  }

  return result;
}

void OfflineRecognizerMossSatsImpl::DecodeStreams(OfflineStream **ss,
                                                  int32_t n) const {
  for (int32_t i = 0; i != n; ++i) {
    Decode(ss[i]);
  }
}

void OfflineRecognizerMossSatsImpl::Decode(OfflineStream *stream) const {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  std::vector<float> f = stream->GetFrames();
  if (f.empty()) {
    OfflineRecognitionResult r;
    r.text = "";
    stream->SetResult(r);
    return;
  }

  int32_t num_frames =
      static_cast<int32_t>(f.size() / static_cast<size_t>(kMossSatsMelDim));
  if (static_cast<size_t>(num_frames) * static_cast<size_t>(kMossSatsMelDim) !=
      f.size()) {
    OfflineRecognitionResult r;
    r.text = "";
    stream->SetResult(r);
    return;
  }
  if (num_frames < 2) {
    OfflineRecognitionResult r;
    r.text = "";
    stream->SetResult(r);
    return;
  }

  NormalizeWhisperFeatures(f.data(), num_frames, kMossSatsMelDim);

  int32_t F = kMossSatsMelDim;
  int32_t feat_frames = num_frames;

  // MOSS-SATS encoder is a Whisper encoder: it processes fixed 30-s windows
  // (<=3000 mel frames). Chunk the mel, encode each window, and concatenate
  // the audio embeddings along time (mirrors the Python audio_chunk_mapping).
  constexpr int32_t kMelChunk = 3000;
  std::vector<float> mel(static_cast<size_t>(F) * feat_frames);
  for (int32_t t = 0; t < feat_frames; ++t) {
    for (int32_t b = 0; b < F; ++b) {
      mel[static_cast<size_t>(b) * feat_frames + t] =
          f[static_cast<size_t>(t) * F + b];
    }
  }

  std::vector<float> embeds;  // concatenated (T_tok, H)
  int64_t total_tok = 0;
  int64_t hidden = 0;
  for (int32_t off = 0; off < feat_frames; off += kMelChunk) {
    const int32_t n = std::min(kMelChunk, feat_frames - off);
    if (n < 16) {
      break;  // ignore a tiny tail (sub-0.2 s)
    }
    // The exported encoder has a fixed 30-s positional table: always feed
    // exactly kMelChunk frames (zero-pad the tail) and keep only the tokens
    // corresponding to real audio: (n/2)/4.
    std::vector<float> chunk(static_cast<size_t>(F) * kMelChunk, 0.0f);
    for (int32_t b = 0; b < F; ++b) {
      std::memcpy(chunk.data() + static_cast<size_t>(b) * kMelChunk,
                  mel.data() + static_cast<size_t>(b) * feat_frames + off,
                  sizeof(float) * n);
    }
    std::array<int64_t, 3> mel_shape{1, static_cast<int64_t>(F),
                                     static_cast<int64_t>(kMelChunk)};
    Ort::Value mel_tensor = Ort::Value::CreateTensor<float>(
        memory_info, chunk.data(), chunk.size(), mel_shape.data(),
        mel_shape.size());
    Ort::Value part = model_->ForwardEncoder(std::move(mel_tensor));
    auto shp = part.GetTensorTypeAndShapeInfo().GetShape();  // (1, T', H)
    const float *pd = part.GetTensorData<float>();
    hidden = shp[2];
    const int64_t keep = std::min<int64_t>(shp[1], (n / 2) / 4);
    embeds.insert(embeds.end(), pd, pd + keep * shp[2]);
    total_tok += keep;
  }
  if (total_tok == 0 || hidden == 0) {
    OfflineRecognitionResult r0;
    stream->SetResult(r0);
    return;
  }
  std::array<int64_t, 3> emb_shape{1, total_tok, hidden};
  Ort::Value audio_features = Ort::Value::CreateTensor<float>(
      memory_info, embeds.data(), embeds.size(), emb_shape.data(),
      emb_shape.size());

  int32_t expected_audio_token_len = static_cast<int32_t>(total_tok);

  if (config_.model_config.debug) {
    SHERPA_ONNX_LOGE("moss-sats: feat_frames=%d expected_audio_tokens=%d",
                     feat_frames, expected_audio_token_len);
  }

  OfflineRecognitionResult r = GenerateText(std::move(audio_features),
                                            expected_audio_token_len, stream);

  r.text = ApplyHomophoneReplacer(std::move(r.text));

  stream->SetResult(r);
}

#if __ANDROID_API__ >= 9
template OfflineRecognizerMossSatsImpl::OfflineRecognizerMossSatsImpl(
    AAssetManager *mgr, const OfflineRecognizerConfig &config);
#endif

#if __OHOS__
template OfflineRecognizerMossSatsImpl::OfflineRecognizerMossSatsImpl(
    NativeResourceManager *mgr, const OfflineRecognizerConfig &config);
#endif

}  // namespace sherpa_onnx
