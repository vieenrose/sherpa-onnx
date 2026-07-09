// sherpa-onnx/csrc/offline-moss-sats-parser.h
//
// Streaming parser for MOSS-Transcribe-Diarize compact transcript output:
//     [start][Sxx]text[end]...
// C++ port of moss_transcribe_diarize/transcript_parser.py (Apache-2.0).
//
// Copyright (c)  2026

#ifndef SHERPA_ONNX_CSRC_OFFLINE_MOSS_SATS_PARSER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_MOSS_SATS_PARSER_H_

#include <functional>
#include <string>
#include <vector>

namespace sherpa_onnx {

struct MossSatsSegment {
  float start = 0.0f;
  float end = 0.0f;
  std::string speaker;  // e.g. "S01"
  std::string text;
};

// Single-pass character state machine; no regex, tolerant of malformed spans
// (falls back to treating them as text, mirroring the reference parser).
class MossSatsTranscriptParser {
 public:
  explicit MossSatsTranscriptParser(bool strip_text = true,
                                    bool skip_empty = true)
      : strip_text_(strip_text), skip_empty_(skip_empty) {}

  void Reset();

  // Consume a UTF-8 text chunk; append completed segments to *segments.
  void Feed(const std::string &chunk, std::vector<MossSatsSegment> *segments);

  // Finish the stream; emits a trailing complete segment if any.
  void Close(std::vector<MossSatsSegment> *segments);

  // Convenience: parse a whole transcript in one call.
  static std::vector<MossSatsSegment> Parse(const std::string &transcript);

 private:
  enum class State {
    kSeekStart,
    kReadStart,
    kExpectSpeakerOpen,
    kReadSpeaker,
    kReadText,
    kReadEnd,
    kAfterEnd,
  };

  void FeedByte(char ch, std::vector<MossSatsSegment> *segments);
  void Emit(std::vector<MossSatsSegment> *segments);

  static bool ParseTimestamp(const std::string &token, float *value);
  static bool ParseSpeaker(const std::string &token, std::string *speaker);
  static bool IsTimestampChar(char ch) {
    return (ch >= '0' && ch <= '9') || ch == '.';
  }
  static bool IsSpeakerChar(char ch) {
    return (ch >= '0' && ch <= '9') || ch == 'S';
  }
  static bool IsAsciiSpace(char ch) {
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' ||
           ch == '\v';
  }

  bool strip_text_;
  bool skip_empty_;

  State state_ = State::kSeekStart;
  std::string token_;
  std::string text_;
  std::string pending_after_end_;
  std::string end_token_;
  std::string speaker_;
  float start_ = -1.0f;
  float end_ = -1.0f;
  bool has_start_ = false;
  bool has_end_ = false;
  bool has_speaker_ = false;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_MOSS_SATS_PARSER_H_
