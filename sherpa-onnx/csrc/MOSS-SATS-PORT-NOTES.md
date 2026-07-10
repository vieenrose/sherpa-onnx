# MOSS-SATS port — continuation map (branch feature/moss-transcribe-diarize)

## Done
- offline-moss-sats-parser.{h,cc}: [start][Sxx]text[end] state machine,
  differential-tested 40/40 byte-exact vs Python reference.
- offline-moss-sats-model-config.{h,cc}.

## Strategy (revised): adapt the existing Qwen3-ASR implementation
Upstream already ships the machinery MOSS needs:
- offline-qwen3-asr-model.{h,cc}   : encoder + LLM forward w/ FIXED-size KV
  cache + per-step KV deltas (ApplyKvDeltaInplace), audio_features as a
  separate graph input (splice done INSIDE the decoder graph).
- offline-recognizer-qwen3-asr-impl.cc (1125 lines): chat-template prompt
  building, hotwords in system role, greedy/sampled decode loop.
- qwen-asr-tokenizer.{h,cc}: C++ Qwen BPE (reusable as-is for MOSS's Qwen3
  tokenizer).

## Remaining steps
1. RE-EXPORT the MOSS decoder to the Qwen3-ASR graph interface (do this in
   the distil-vibevoice repo, scripts/31_export_moss_qwen3style.py):
   inputs  = (input_ids[B,T] i64, audio_features[B,A,H] f32,
              attention_mask[B,T] i64, cache_position[T] i64,
              per-layer fixed KV cache pairs)
   outputs = (logits, per-layer KV deltas)
   Inside the wrapper: embeds = embed(input_ids);
   mask = (input_ids == audio_token_id); embeds[mask] = audio_features;
   run Qwen3 attention with the fixed-cache/delta convention — copy the
   convention from the Qwen3-ASR export script (see the model card /
   k2-fsa docs for qwen3-asr export; the C++ reads I/O names positionally,
   layout must match). Encoder export stays as-is from
   scripts/30_export_moss_onnx.py (mel -> audio embeds; parity PASS).
   MOSS audio token id: see models/moss/added_tokens.json / config
   (audio placeholder token used by masked_scatter in modeling file).
2. Copy offline-qwen3-asr-model.{h,cc} -> offline-moss-sats-model.{h,cc}:
   - drop ForwardConvFrontend; encoder input = mel [B, 80, T] (Whisper-style),
     output audio embeds [B, T/8, 1024]
   - keep ForwardLLM/CreateEmptyKVCache/ApplyKvDeltaInplace unchanged.
3. Copy offline-recognizer-qwen3-asr-impl.* -> offline-recognizer-moss-sats-impl.*:
   - prompt = MOSS chat template (see models/moss/chat_template.jinja; user
     content = audio placeholders + instruction text; hotwords/custom prompt
     supported the same way)
   - after decode: feed text through MossSatsTranscriptParser; put segments
     into the recognizer result (extend OfflineRecognizerResult or attach
     json in `text` + a `segments` field).
4. Register: offline-model-config.{h,cc} (add moss_sats member + Register +
   Validate branch), offline-recognizer-impl.cc dispatch (model-type
   "moss_sats"), CMakeLists.txt (3 new .cc).
5. Long-form: chunk >30s audio into windows; cross-window speaker linking via
   speaker-embedding-extractor (ECAPA) + agglomerative clustering (see
   distil-vibevoice runtime/consolidate.py recluster method, validated 0.93).
6. Build + E2E vs models/moss_onnx + models/moss_ft_zhtw exports; python
   binding; docs.

## Qwen3-ASR decoder ONNX interface (introspected 2026-07-10 from
## csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25, cached /tmp/q3ref)
INPUTS (60): input_ids[B,S] i64; audio_features[B,A,1024] f32;
  attention_mask[B,S] i64; cache_position[S] i64;
  cache_key_i/cache_value_i [B, max_total_len, 8, 128] f32  (28 layers)
OUTPUTS (57): logits[B,S,151936]; key_delta_i/value_delta_i [B,S,8,128] (28)
KEY FACTS:
- Audio splice happens INSIDE the graph: input_ids carries audio placeholder
  tokens; audio_features fed separately; graph scatters them (same
  masked_scatter pattern as MOSS modeling code, audio_token_id=151671).
- KV layout [B, max_total_len, kv_heads, head_dim] (seq at dim1, NOT HF's
  [B,kv,S,hd]); fixed-size cache, per-step deltas out; C++ writes deltas at
  cache_position (ApplyKvDeltaInplace).
- GEOMETRY IDENTICAL to MOSS decoder: 28 layers, kv 8, head_dim 128,
  hidden 1024 (both Qwen3-0.6B). Even audio_features hidden (1024) matches.
  => DONE (2026-07-10): scripts/31_export_moss_qwen3style.py in distil repo — parity PASS (prefill 2.9e-5, cached 3.3e-5), ONNX at models/moss_onnx_sherpa/decoder.onnx (fp32 2.0G, int8-quantize before ship). NOTE: attention_mask input pruned by tracer (59 inputs) — C++ copy must drop that input. Encoder graph: reuse models/moss_onnx/encoder.onnx (mel->audio embeds, parity-passed). Next: C++ offline-moss-sats-model copy-adapt. custom attention over the fixed
  cache (mask by cache_position) — replicate whatever the qwen3-asr exporter
  did; compare logits vs stock HF forward for parity. Then C++ side of
  offline-moss-sats-model can be a near-verbatim copy of offline-qwen3-asr-model
  minus the conv_frontend (Whisper mel encoder instead).
