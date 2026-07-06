// sherpa-onnx/python/csrc/offline-tts-mbistft-stream-model.cc
//
// Copyright (c)  2026

#include "sherpa-onnx/csrc/offline-tts-mbistft-stream-model.h"

#include <string>
#include <vector>

#include "sherpa-onnx/python/csrc/offline-tts-mbistft-stream-model.h"

namespace sherpa_onnx {

void PybindOfflineTtsMbistftStreamModel(py::module *m) {
  using PyClass = OfflineTtsMbistftStreamModel;
  py::class_<PyClass>(*m, "OfflineTtsMbistftStreamModel")
      .def(py::init<const std::string &, const std::string &, int32_t,
                    const std::string &, int32_t>(),
           py::arg("enc"), py::arg("dec"), py::arg("num_threads") = 2,
           py::arg("provider") = "cpu", py::arg("right_lookahead") = 16)
      .def_property_readonly("sample_rate", &PyClass::SampleRate)
      // Generate(x, tone, lang, noise_scale, length_scale, callback):
      // callback(samples: List[float], progress: float) -> bool (keep going).
      // Returns the full waveform as a list<float>.
      .def(
          "generate",
          [](const PyClass &self, const std::vector<int64_t> &x,
             const std::vector<int64_t> &tone, const std::vector<int64_t> &lang,
             float noise_scale, float length_scale, py::object callback) {
            PyClass::Callback cb = nullptr;
            if (!callback.is_none()) {
              cb = [callback](const float *s, int32_t n, float p) -> bool {
                py::gil_scoped_acquire gil;
                std::vector<float> chunk(s, s + n);
                py::object r = callback(chunk, p);
                return r.is_none() ? true : r.cast<bool>();
              };
            }
            std::vector<float> audio;
            {
              py::gil_scoped_release rel;
              audio = self.Generate(x, tone, lang, noise_scale, length_scale, cb);
            }
            return audio;
          },
          py::arg("x"), py::arg("tone"), py::arg("lang"),
          py::arg("noise_scale") = 0.667f, py::arg("length_scale") = 1.0f,
          py::arg("callback") = py::none());
}

}  // namespace sherpa_onnx
