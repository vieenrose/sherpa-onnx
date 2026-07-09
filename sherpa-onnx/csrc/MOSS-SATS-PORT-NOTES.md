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
