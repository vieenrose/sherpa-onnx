# Streaming PrimeTTS (MB-iSTFT-VITS) integration

Adds a token-level **streaming TTS** to sherpa-onnx: text encoder runs once, the
causal vocoder runs per 24-frame chunk, and audio is emitted incrementally via
the existing `GeneratedAudioCallback`. Bit-exact vs the whole-utterance model
(validated: cos 1.000000, first-chunk ~15-30 ms CPU).

## Model (two ONNX graphs)
- `v2stream_enc.onnx` : `(x,tone,lang,x_lengths,noise_scale,length_scale) -> z[1,192,T]`
- `v2stream_dec.onnx` : `z[1,192,Tc] -> wav[1,1,Tc*256]`

Export: `MB-iSTFT-VITS/tools/export_onnx_stream_split.py`.
Reference orchestration (authoritative recipe): `jetson-tts/streaming/onnx_stream.py`.

Streaming = overlap-save: for chunk frames `[a,b)` decode `z[:, :, a-LEFT : b+RIGHT]`
and keep the middle `(b-a)*256` samples. Params (validated, in the model header):
`CHUNK=24, LEFT=64, RIGHT=4, HOP=256, CHAN=192`.

## Files added
- `csrc/offline-tts-mbistft-stream-model.{h,cc}` — the model class (enc once +
  chunked dec + per-chunk callback). **DONE.**

## Remaining wiring (build-side)
1. **Config struct** — add `OfflineTtsMbistftStreamModelConfig { std::string enc, dec;
   float noise_scale=0.667, length_scale=1.0; }` (new file
   `csrc/offline-tts-mbistft-stream-model-config.{h,cc}`, mirror
   `offline-tts-vits-model-config.*`), then a field `mbistft` in
   `OfflineTtsModelConfig` (`csrc/offline-tts-model-config.h`) + its Register/Validate.
2. **Impl subclass** — `csrc/offline-tts-mbistft-stream-impl.h` mirroring
   `offline-tts-vits-impl.h`: frontend tokens -> `model_->Generate(x,tone,lang,
   ns,ls,callback)`. NOTE the frontend must emit **tone** and **lang** id streams
   (3 embeddings) + **add_blank** interleave (blank=0 between tokens), matching
   `streaming/onnx_stream.py::_blank`. The lexicon/g2p is bopomofo+arpabet (88 syms,
   6 tones, 2 langs) — same as the deployed PrimeTTS frontend.
3. **Dispatch** — in `csrc/offline-tts-impl.cc::OfflineTtsImpl::Create`, route to the
   new impl when `config.model.mbistft.enc` is non-empty.
4. **CMake** — add the two new `.cc` to `csrc/CMakeLists.txt` (SHERPA_ONNX_SRCS).
5. **(optional) Python/CLI bindings** — mirror vits config in `python/csrc/…`.

## Build
Needs the ORT fork built from source (github.com/vieenrose/onnxruntime, `main`).
Then standard sherpa-onnx cmake build. Validate against `streaming/onnx_stream.py`
output (must match bit-for-bit).
