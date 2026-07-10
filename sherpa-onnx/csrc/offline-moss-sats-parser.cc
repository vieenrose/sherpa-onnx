// sherpa-onnx/csrc/offline-moss-sats-parser.cc
//
// Copyright (c)  2026

#include "sherpa-onnx/csrc/offline-moss-sats-parser.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace sherpa_onnx {

void MossSatsTranscriptParser::Reset() {
  state_ = State::kSeekStart;
  token_.clear();
  text_.clear();
  pending_after_end_.clear();
  end_token_.clear();
  speaker_.clear();
  has_start_ = has_end_ = has_speaker_ = false;
}

void MossSatsTranscriptParser::Feed(const std::string &chunk,
                                    std::vector<MossSatsSegment> *segments) {
  for (char ch : chunk) {
    FeedByte(ch, segments);
  }
}

void MossSatsTranscriptParser::Close(std::vector<MossSatsSegment> *segments) {
  if (state_ == State::kAfterEnd) {
    Emit(segments);
  }
  Reset();
}

std::vector<MossSatsSegment> MossSatsTranscriptParser::Parse(
    const std::string &transcript) {
  MossSatsTranscriptParser parser;
  std::vector<MossSatsSegment> segments;
  parser.Feed(transcript, &segments);
  parser.Close(&segments);
  return segments;
}

bool MossSatsTranscriptParser::ParseTimestamp(const std::string &token,
                                              float *value) {
  if (token.empty()) return false;
  int dots = 0, digits = 0;
  for (char ch : token) {
    if (ch >= '0' && ch <= '9') {
      ++digits;
    } else if (ch == '.') {
      if (++dots > 1) return false;
    } else {
      return false;
    }
  }
  if (digits == 0) return false;
  *value = std::strtof(token.c_str(), nullptr);
  return true;
}

bool MossSatsTranscriptParser::ParseSpeaker(const std::string &token,
                                            std::string *speaker) {
  if (token.size() < 2 || token[0] != 'S') return false;
  for (size_t i = 1; i < token.size(); ++i) {
    if (token[i] < '0' || token[i] > '9') return false;
  }
  *speaker = token;
  return true;
}

void MossSatsTranscriptParser::FeedByte(char ch,
                                        std::vector<MossSatsSegment> *segs) {
  switch (state_) {
    case State::kSeekStart: {
      if (ch == '[') {
        token_.clear();
        state_ = State::kReadStart;
      }
      break;
    }
    case State::kReadStart: {
      if (ch == ']') {
        float v;
        if (!ParseTimestamp(token_, &v)) {
          Reset();
          return;
        }
        start_ = v;
        has_start_ = true;
        token_.clear();
        state_ = State::kExpectSpeakerOpen;
        return;
      }
      if (IsTimestampChar(ch)) {
        token_.push_back(ch);
        if (token_.size() <= 32) return;
      }
      Reset();
      if (ch == '[') state_ = State::kReadStart;
      break;
    }
    case State::kExpectSpeakerOpen: {
      if (ch == '[') {
        token_.clear();
        state_ = State::kReadSpeaker;
      } else if (!IsAsciiSpace(ch)) {
        Reset();
      }
      break;
    }
    case State::kReadSpeaker: {
      if (ch == ']') {
        std::string spk;
        if (!ParseSpeaker(token_, &spk)) {
          Reset();
          return;
        }
        speaker_ = spk;
        has_speaker_ = true;
        text_.clear();
        token_.clear();
        state_ = State::kReadText;
        return;
      }
      if (IsSpeakerChar(ch)) {
        token_.push_back(ch);
        if (token_.size() <= 16) return;
      }
      Reset();
      if (ch == '[') state_ = State::kReadStart;
      break;
    }
    case State::kReadText: {
      if (ch == '[') {
        token_.clear();
        state_ = State::kReadEnd;
      } else {
        text_.push_back(ch);
      }
      break;
    }
    case State::kReadEnd: {
      if (ch == ']') {
        float v;
        if (ParseTimestamp(token_, &v) && has_start_ && v >= start_) {
          end_ = v;
          has_end_ = true;
          end_token_ = token_;
          pending_after_end_.clear();
          state_ = State::kAfterEnd;
        } else {
          text_.push_back('[');
          text_ += token_;
          text_.push_back(']');
          state_ = State::kReadText;
        }
        token_.clear();
        return;
      }
      if (IsTimestampChar(ch)) {
        token_.push_back(ch);
        if (token_.size() <= 32) return;
      }
      text_.push_back('[');
      text_ += token_;
      text_.push_back(ch);
      token_.clear();
      state_ = State::kReadText;
      break;
    }
    case State::kAfterEnd: {
      if (ch == '[') {
        Emit(segs);
        token_.clear();
        state_ = State::kReadStart;
        return;
      }
      if (IsAsciiSpace(ch)) {
        pending_after_end_.push_back(ch);
        return;
      }
      // The "[end]" turned out to be inline text; roll it back into the text.
      text_.push_back('[');
      text_ += end_token_;
      text_.push_back(']');
      text_ += pending_after_end_;
      text_.push_back(ch);
      pending_after_end_.clear();
      has_end_ = false;
      end_token_.clear();
      state_ = State::kReadText;
      break;
    }
  }
}

void MossSatsTranscriptParser::Emit(std::vector<MossSatsSegment> *segs) {
  if (!has_start_ || !has_end_ || !has_speaker_) {
    Reset();
    return;
  }
  std::string text = text_;
  if (strip_text_) {
    size_t b = text.find_first_not_of(" \t\n\r\f\v");
    size_t e = text.find_last_not_of(" \t\n\r\f\v");
    text = (b == std::string::npos) ? "" : text.substr(b, e - b + 1);
  }
  if (!text.empty() || !skip_empty_) {
    MossSatsSegment seg;
    seg.start = start_;
    seg.end = end_;
    seg.speaker = speaker_;
    seg.text = std::move(text);
    segs->push_back(std::move(seg));
  }
  token_.clear();
  text_.clear();
  pending_after_end_.clear();
  has_start_ = has_end_ = has_speaker_ = false;
}

}  // namespace sherpa_onnx
