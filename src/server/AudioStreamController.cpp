#include "AudioStreamController.hpp"
#include "AudioProcessor.hpp"
#include "server/TriggerEngine.h"

#include <drogon/drogon.h>
#include <json/json.h>

#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string_view>
#include <thread>

namespace {

struct TranscriptEntry {
  int64_t ts_ms;
  std::string text;
};

class VadSegmenter;

enum class SessionMode {
  kNote,
  kLecture,
  kDefense,
};

struct ServiceConfig {
  std::string whisper_base = "http://127.0.0.1:8081";
  std::string llama_base = "http://127.0.0.1:8000";
  std::string tts_base = "http://127.0.0.1:9880";
  std::string llama_model = "Qwen2.5-14B-Instruct-Q4_K_M.gguf";
  std::string tts_ref_audio_path =
      "/root/madcamp04/FSD/GPT-SoVITS/output/slicer_opt_sb_cor/"
      "1_(Vocals).wav_0000161600_0000318720.wav";
  std::string tts_prompt_text =
      "제 목소리는 자연스럽고 또렷하게 들리도록 일정한 톤으로 말하겠습니다.";
  std::string tts_prompt_lang = "ko";
  std::string tts_text_lang = "ko";
  std::string attendance_audio_path = "/root/madcamp04/FSD/yes.m4a";
  std::string stt_language = "ko";
  std::string stt_prompt;
  int64_t recent_window_ms = 10 * 60 * 1000;
  int stt_max_in_flight = 2;
  float vad_threshold = 0.01f;
  // Accuracy-first VAD segmenter config
  int sample_rate = 16000;
  int vad_frame_ms = 20;
  int vad_start_ms = 200;
  int vad_end_ms = 200;
  int vad_prepad_ms = 200;
  int vad_postpad_ms = 300;
  int vad_min_utt_ms = 600;
  int vad_max_utt_ms = 8000;
  int vad_overlap_ms = 700;
  float vad_threshold_db = -40.0f;
  int vad_noise_calib_ms = 1500;
  float vad_noise_margin_db = 10.0f;
  bool vad_auto_calib = true;
};

ServiceConfig LoadConfig() {
  ServiceConfig cfg;
  auto get_int = [](const char* v, int fallback) {
    if (!v) return fallback;
    try { return std::stoi(v); } catch (...) { return fallback; }
  };
  auto get_f = [](const char* v, float fallback) {
    if (!v) return fallback;
    try { return std::stof(v); } catch (...) { return fallback; }
  };
  if (const char* v = std::getenv("CD_WHISPER_BASE")) cfg.whisper_base = v;
  if (const char* v = std::getenv("CD_LLM_BASE")) cfg.llama_base = v;
  if (const char* v = std::getenv("CD_TTS_BASE")) cfg.tts_base = v;
  if (const char* v = std::getenv("CD_LLM_MODEL")) cfg.llama_model = v;
  if (const char* v = std::getenv("CD_TTS_REF_AUDIO")) cfg.tts_ref_audio_path = v;
  if (const char* v = std::getenv("CD_TTS_PROMPT_TEXT")) cfg.tts_prompt_text = v;
  if (const char* v = std::getenv("CD_TTS_PROMPT_LANG")) cfg.tts_prompt_lang = v;
  if (const char* v = std::getenv("CD_TTS_TEXT_LANG")) cfg.tts_text_lang = v;
  if (const char* v = std::getenv("CD_ATTENDANCE_AUDIO")) cfg.attendance_audio_path = v;
  if (const char* v = std::getenv("CD_STT_LANGUAGE")) cfg.stt_language = v;
  if (const char* v = std::getenv("CD_STT_PROMPT")) cfg.stt_prompt = v;
  cfg.vad_frame_ms = get_int(std::getenv("CD_VAD_FRAME_MS"), cfg.vad_frame_ms);
  cfg.vad_start_ms = get_int(std::getenv("CD_VAD_START_MS"), cfg.vad_start_ms);
  cfg.vad_end_ms = get_int(std::getenv("CD_VAD_END_MS"), cfg.vad_end_ms);
  cfg.vad_prepad_ms = get_int(std::getenv("CD_VAD_PREPAD_MS"), cfg.vad_prepad_ms);
  cfg.vad_postpad_ms = get_int(std::getenv("CD_VAD_POSTPAD_MS"), cfg.vad_postpad_ms);
  cfg.vad_min_utt_ms = get_int(std::getenv("CD_VAD_MIN_UTT_MS"), cfg.vad_min_utt_ms);
  cfg.vad_max_utt_ms = get_int(std::getenv("CD_VAD_MAX_UTT_MS"), cfg.vad_max_utt_ms);
  cfg.vad_overlap_ms = get_int(std::getenv("CD_VAD_OVERLAP_MS"), cfg.vad_overlap_ms);
  cfg.vad_threshold_db = get_f(std::getenv("CD_VAD_THRESHOLD_DB"), cfg.vad_threshold_db);
  cfg.vad_noise_calib_ms = get_int(std::getenv("CD_VAD_NOISE_CALIB_MS"), cfg.vad_noise_calib_ms);
  cfg.vad_noise_margin_db = get_f(std::getenv("CD_VAD_NOISE_MARGIN_DB"), cfg.vad_noise_margin_db);
  return cfg;
}

ServiceConfig& GetConfig() {
  static ServiceConfig cfg = LoadConfig();
  return cfg;
}

struct SessionState {
  AudioProcessor audio;
  cd::server::TriggerEngine trigger;
  std::unique_ptr<VadSegmenter> vad;
  SessionMode mode = SessionMode::kLecture;
  std::deque<TranscriptEntry> recent;
  std::vector<TranscriptEntry> full;
  std::mutex lock;
  std::atomic<int> stt_in_flight{0};
  std::atomic<bool> llm_busy{false};
  std::atomic<bool> classifier_busy{false};
  int64_t last_question_ts = 0;
  int64_t last_question_suggest_ts = 0;  // "질문 있나요?" 감지 쿨다운
  bool attendance_mode = false;
  int64_t attendance_started_ms = 0;
  int64_t last_attendance_tts_ms = 0;

  SessionState() : audio(AudioProcessor::Config{}), vad(std::make_unique<VadSegmenter>(GetConfig())) {}
};

int64_t NowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

bool ParseJson(std::string_view body, Json::Value& out, std::string& err) {
  Json::CharReaderBuilder builder;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  const char* begin = body.data();
  const char* end = body.data() + body.size();
  return reader->parse(begin, end, &out, &err);
}

void PruneRecent(std::deque<TranscriptEntry>& recent, int64_t now_ms, int64_t window_ms) {
  while (!recent.empty()) {
    if (now_ms - recent.front().ts_ms <= window_ms) break;
    recent.pop_front();
  }
}

std::string JoinText(const std::deque<TranscriptEntry>& entries, size_t max_chars) {
  std::string out;
  for (const auto& e : entries) {
    if (!out.empty()) out += "\n";
    out += e.text;
    if (out.size() >= max_chars) {
      if (out.size() > max_chars) {
        out.erase(0, out.size() - max_chars);
      }
      break;
    }
  }
  return out;
}

std::string JoinText(const std::vector<TranscriptEntry>& entries, size_t max_chars) {
  std::string out;
  for (const auto& e : entries) {
    if (!out.empty()) out += "\n";
    out += e.text;
    if (out.size() >= max_chars) break;
  }
  return out;
}

// Ambiguous-question classification disabled: only explicit trigger keywords activate responses.

bool IsAttendanceStart(const std::string& text) {
  return text.find("출석") != std::string::npos;
}

bool IsNameCalled(const std::string& text) {
  static const std::vector<std::string> names = {"이상범", "이상 범", "이 상범", "이 상 범", "이. 상. 범.", "이상봉"};
  for (const auto& name : names) {
    if (text.find(name) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool IsAskingForQuestions(const std::string& text) {
  // "질문 있나요?", "질문 있습니까?", "궁금한 점", "질문 있으면" 등 감지
  static const std::vector<std::string> triggers = {
    "질문 있나요", "질문 있습니까", "질문있나요", "질문있습니까",
  };
  for (const auto& t : triggers) {
    if (text.find(t) != std::string::npos) {
      return true;
    }
  }
  return false;
}

std::string MakeBoundary() {
  static std::atomic<int> counter{0};
  return "----cd-boundary-" + std::to_string(counter.fetch_add(1));
}

void WriteU16LE(char* dst, uint16_t v) {
  dst[0] = static_cast<char>(v & 0xff);
  dst[1] = static_cast<char>((v >> 8) & 0xff);
}

void WriteU32LE(char* dst, uint32_t v) {
  dst[0] = static_cast<char>(v & 0xff);
  dst[1] = static_cast<char>((v >> 8) & 0xff);
  dst[2] = static_cast<char>((v >> 16) & 0xff);
  dst[3] = static_cast<char>((v >> 24) & 0xff);
}

std::string ReadFileBinary(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  std::ostringstream oss;
  oss << in.rdbuf();
  return oss.str();
}

std::string BuildWav(const std::vector<int16_t>& pcm, int sample_rate) {
  uint32_t data_size = static_cast<uint32_t>(pcm.size() * sizeof(int16_t));
  std::string wav;
  wav.resize(44 + data_size);
  char* p = wav.data();
  std::memcpy(p, "RIFF", 4);
  WriteU32LE(p + 4, 36 + data_size);
  std::memcpy(p + 8, "WAVE", 4);
  std::memcpy(p + 12, "fmt ", 4);
  WriteU32LE(p + 16, 16);
  WriteU16LE(p + 20, 1);
  WriteU16LE(p + 22, 1);
  WriteU32LE(p + 24, static_cast<uint32_t>(sample_rate));
  WriteU32LE(p + 28, static_cast<uint32_t>(sample_rate * 2));
  WriteU16LE(p + 32, 2);
  WriteU16LE(p + 34, 16);
  std::memcpy(p + 36, "data", 4);
  WriteU32LE(p + 40, data_size);
  if (!pcm.empty()) {
    std::memcpy(p + 44, pcm.data(), data_size);
  }
  return wav;
}

std::string BuildMultipartBody(const std::string& boundary, const std::string& wav_bytes) {
  std::ostringstream oss;
  oss << "--" << boundary << "\r\n";
  oss << "Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n";
  oss << "Content-Type: audio/wav\r\n\r\n";
  oss << wav_bytes << "\r\n";
  // Optional whisper params
  const auto& cfg = GetConfig();
  if (!cfg.stt_language.empty()) {
    oss << "--" << boundary << "\r\n";
    oss << "Content-Disposition: form-data; name=\"language\"\r\n\r\n";
    oss << cfg.stt_language << "\r\n";
  }
  if (!cfg.stt_prompt.empty()) {
    oss << "--" << boundary << "\r\n";
    oss << "Content-Disposition: form-data; name=\"prompt\"\r\n\r\n";
    oss << cfg.stt_prompt << "\r\n";
  }
  oss << "--" << boundary << "\r\n";
  oss << "Content-Disposition: form-data; name=\"response_format\"\r\n\r\n";
  oss << "json\r\n";
  oss << "--" << boundary << "--\r\n";
  return oss.str();
}

std::string ToString(const Json::Value& value) {
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  return Json::writeString(builder, value);
}

float RmsDbfs(const std::vector<int16_t>& frame) {
  if (frame.empty()) return -120.0f;
  const float inv = 1.0f / 32768.0f;
  double sum_sq = 0.0;
  for (int16_t s : frame) {
    float v = static_cast<float>(s) * inv;
    sum_sq += static_cast<double>(v * v);
  }
  double rms = std::sqrt(sum_sq / frame.size());
  return 20.0f * std::log10(std::max(rms, 1e-9));
}

class VadSegmenter {
 public:
  explicit VadSegmenter(const ServiceConfig& cfg)
      : cfg_(cfg),
        frame_samples_(cfg.sample_rate * cfg.vad_frame_ms / 1000),
        start_frames_(std::max(1, cfg.vad_start_ms / cfg.vad_frame_ms)),
        end_frames_(std::max(1, cfg.vad_end_ms / cfg.vad_frame_ms)),
        post_frames_(std::max(0, cfg.vad_postpad_ms / cfg.vad_frame_ms)),
        prepad_samples_(cfg.sample_rate * cfg.vad_prepad_ms / 1000),
        min_samples_(cfg.sample_rate * cfg.vad_min_utt_ms / 1000),
        max_samples_(cfg.sample_rate * cfg.vad_max_utt_ms / 1000),
        overlap_samples_(cfg.sample_rate * cfg.vad_overlap_ms / 1000),
        noise_calib_frames_(std::max(0, cfg.vad_noise_calib_ms / cfg.vad_frame_ms)) {}

  int frame_samples() const { return frame_samples_; }

  template <typename F>
  void PushFrame(const std::vector<int16_t>& frame, F on_segment) {
    if (frame.empty()) return;

    const float db = RmsDbfs(frame);

    // noise calibration
    if (cfg_.vad_auto_calib && !calibrated_ && noise_calib_frames_ > 0) {
      noise_accum_ += std::pow(10.0f, db / 20.0f);
      calib_frames_++;
      if (calib_frames_ >= noise_calib_frames_) {
        float avg_rms = noise_accum_ / static_cast<float>(calib_frames_);
        noise_floor_db_ = 20.0f * std::log10(std::max(avg_rms, 1e-9f));
        calibrated_ = true;
      }
    }

    float threshold_db = cfg_.vad_threshold_db;
    if (cfg_.vad_auto_calib && calibrated_) {
      threshold_db = std::max(threshold_db, noise_floor_db_ + cfg_.vad_noise_margin_db);
    }
    const bool speech = db > threshold_db;
    if (cfg_.vad_auto_calib && calibrated_ && !in_speech_ && !speech) {
      noise_floor_db_ = 0.99f * noise_floor_db_ + 0.01f * db;
    }

    if (!in_speech_) {
      if (speech) {
        speech_frames_++;
      } else {
        speech_frames_ = 0;
      }
      if (speech_frames_ >= start_frames_) {
        in_speech_ = true;
        current_.clear();
        if (!prepad_.empty()) {
          current_.insert(current_.end(), prepad_.begin(), prepad_.end());
        }
        current_.insert(current_.end(), frame.begin(), frame.end());
        silence_frames_ = 0;
        speech_frames_ = 0;
      }
      // update prepad buffer (keep only previous audio)
      if (!in_speech_ && prepad_samples_ > 0) {
        prepad_.insert(prepad_.end(), frame.begin(), frame.end());
        if (prepad_.size() > static_cast<size_t>(prepad_samples_)) {
          prepad_.erase(prepad_.begin(), prepad_.end() - prepad_samples_);
        }
      }
      return;
    }

    // in speech
    current_.insert(current_.end(), frame.begin(), frame.end());
    if (speech) {
      silence_frames_ = 0;
    } else {
      silence_frames_++;
    }

    // forced cut for max length with overlap
    if (static_cast<int>(current_.size()) >= max_samples_) {
      if (static_cast<int>(current_.size()) >= min_samples_) {
        auto segment = current_;
        on_segment(std::move(segment));
      }
      // keep overlap for next segment continuation
      if (overlap_samples_ > 0 && static_cast<int>(current_.size()) > overlap_samples_) {
        std::vector<int16_t> overlap(current_.end() - overlap_samples_, current_.end());
        current_.swap(overlap);
      } else {
        current_.clear();
      }
      silence_frames_ = 0;
      return;
    }

    // end of speech + postpad
    if (silence_frames_ >= end_frames_ + post_frames_) {
      EmitSegment(on_segment);
      current_.clear();
      in_speech_ = false;
      silence_frames_ = 0;
      speech_frames_ = 0;
      // after end, allow prepad to collect new silence frames
      if (prepad_samples_ > 0) {
        prepad_.clear();
      }
    }
  }

 private:
  template <typename F>
  void EmitSegment(F on_segment) {
    if (static_cast<int>(current_.size()) < min_samples_) {
      return;
    }
    on_segment(std::move(current_));
    current_.clear();
  }

  const ServiceConfig& cfg_;
  int frame_samples_;
  int start_frames_;
  int end_frames_;
  int post_frames_;
  int prepad_samples_;
  int min_samples_;
  int max_samples_;
  int overlap_samples_;
  int noise_calib_frames_;

  bool in_speech_ = false;
  int speech_frames_ = 0;
  int silence_frames_ = 0;
  std::deque<int16_t> prepad_;
  std::vector<int16_t> current_;

  bool calibrated_ = false;
  int calib_frames_ = 0;
  float noise_accum_ = 0.0f;
  float noise_floor_db_ = -60.0f;
};

}  // namespace

AudioStreamController::AudioStreamController() {
  std::cout << "[AudioStreamController] Initialized" << std::endl;
}

AudioStreamController::~AudioStreamController() {
  std::cout << "[AudioStreamController] Destroyed" << std::endl;
}

void AudioStreamController::handleNewConnection(
    const drogon::HttpRequestPtr&,
    const drogon::WebSocketConnectionPtr& wsConnPtr) {
  std::cout << "[AudioStreamController] New WebSocket connection from "
            << wsConnPtr->peerAddr().toIpPort() << std::endl;

  wsConnPtr->setContext(std::make_shared<SessionState>());

  Json::Value payload;
  payload["type"] = "connected";
  payload["message"] = "Audio streaming ready";
  payload["sample_rate"] = 16000;
  payload["channels"] = 1;
  payload["bit_depth"] = 16;
  payload["mode"] = "lecture";
  wsConnPtr->send(ToString(payload));
}

void AudioStreamController::handleNewMessage(
    const drogon::WebSocketConnectionPtr& wsConnPtr,
    std::string&& message,
    const drogon::WebSocketMessageType& type) {
  if (message.empty()) {
    return;
  }

  auto session = wsConnPtr->getContext<SessionState>();
  if (!session) {
    wsConnPtr->send(create_error_response("Missing session state"));
    return;
  }

  if (type == drogon::WebSocketMessageType::Text) {
    Json::Value root;
    std::string err;
    if (!ParseJson(message, root, err)) {
      wsConnPtr->send(create_error_response("Invalid control JSON: " + err));
      return;
    }

    const std::string msg_type = root.get("type", "").asString();
    if (msg_type == "control") {
      const std::string mode = root.get("mode", "").asString();
      if (mode == "note") session->mode = SessionMode::kNote;
      if (mode == "lecture") session->mode = SessionMode::kLecture;
      if (mode == "defense") session->mode = SessionMode::kDefense;

      auto& cfg = GetConfig();
      if (root.isMember("tts_ref_audio_path")) {
        const std::string v = root["tts_ref_audio_path"].asString();
        if (!v.empty()) cfg.tts_ref_audio_path = v;
      }
      if (root.isMember("tts_prompt_text")) {
        const std::string v = root["tts_prompt_text"].asString();
        if (!v.empty()) cfg.tts_prompt_text = v;
      }

      wsConnPtr->send(create_info_response("control", "mode updated"));
      return;
    }

    if (msg_type == "summary") {
      const std::string provided_text = root.get("text", "").asString();
      std::weak_ptr<drogon::WebSocketConnection> weak_conn = wsConnPtr;
      std::thread([session, weak_conn, provided_text]() {
        auto conn = weak_conn.lock();
        if (!conn || !conn->connected()) return;

        std::string full_text;
        if (!provided_text.empty()) {
          full_text = provided_text;
        } else {
          std::lock_guard<std::mutex> lock(session->lock);
          full_text = JoinText(session->full, 200000);
        }

        if (full_text.empty()) {
          if (conn && conn->connected()) {
            Json::Value payload;
            payload["type"] = "summary";
            payload["text"] = "(no transcript)";
            conn->send(ToString(payload));
          }
          return;
        }

        auto& cfg = GetConfig();
        auto client = drogon::HttpClient::newHttpClient(cfg.llama_base);

        const std::string SYSTEM_PROMPT = R"(
          You are a "Versatile Content Analyst" designed to support university students.
          Your task is to summarize ANY input provided, whether it is a formal academic lecture or a casual daily conversation.

          [Operational Rules]
          1. No Refusal: You must summarize the input regardless of its nature (academic, casual, or technical). NEVER say the content is irrelevant.
          2. Language: You MUST respond in Korean (한국어) only. If technical terms appear, you may include the English term in parentheses.
          3. Content Filtering: Ignore stuttering, verbal fillers (uh, um, etc.), and repetitive noise from the STT transcript.
          4. Style: Maintain a professional yet helpful tone as a dedicated assistant.

          [Output Format]
          🎓 핵심 주제 (Core Theme)
          - (주제를 한 줄로 명확하게 정리)

          📌 주요 내용 (Key Points)
          * (강의라면 핵심 지식을, 대화라면 논의된 주요 안건이나 흐름을 요약)
          * (중요한 세부 사항이나 결론)

          💻 용어 및 키워드 (Keywords)
          * (중요 단어): (간결한 설명)

          ✍️ 한 줄 메모 (Note)
          * (학생이 기억해야 할 실용적인 조언이나 다음 할 일)
          )";
        auto summarize_chunk = [&](const std::string& chunk) -> std::string {
          Json::Value req;
          req["model"] = cfg.llama_model;
          Json::Value messages(Json::arrayValue);
          Json::Value system;
          system["role"] = "system";
          system["content"] = SYSTEM_PROMPT;
          Json::Value user;
          user["role"] = "user";
          user["content"] = "Summarize:\n" + chunk;
          messages.append(system);
          messages.append(user);
          req["messages"] = messages;
          req["temperature"] = 0.3;

          auto request = drogon::HttpRequest::newHttpJsonRequest(req);
          request->setMethod(drogon::Post);
          request->setPath("/v1/chat/completions");

          std::promise<std::string> promise;
          client->sendRequest(request, [&promise](drogon::ReqResult r,
                                                  const drogon::HttpResponsePtr& resp) {
            if (r != drogon::ReqResult::Ok || !resp) {
              promise.set_value("");
              return;
            }
            Json::Value json;
            std::string err;
            if (!ParseJson(resp->getBody(), json, err)) {
              promise.set_value("");
              return;
            }
            const auto& choices = json["choices"];
            if (!choices.isArray() || choices.empty()) {
              promise.set_value("");
              return;
            }
            promise.set_value(choices[0]["message"]["content"].asString());
          });
          return promise.get_future().get();
        };

        const size_t chunk_size = 4000;
        std::vector<std::string> partials;
        for (size_t offset = 0; offset < full_text.size(); offset += chunk_size) {
          partials.push_back(summarize_chunk(full_text.substr(offset, chunk_size)));
        }

        std::string final_summary;
        if (partials.size() == 1) {
          final_summary = partials[0];
        } else {
          std::string combined;
          for (const auto& p : partials) {
            if (!combined.empty()) combined += "\n";
            combined += p;
          }
          final_summary = summarize_chunk(combined);
        }

        if (conn && conn->connected()) {
          Json::Value payload;
          payload["type"] = "summary";
          payload["text"] = final_summary;
          conn->send(ToString(payload));
        }
      }).detach();
      return;
    }

    wsConnPtr->send(create_error_response("Unknown control message"));
    return;
  }

  if (type != drogon::WebSocketMessageType::Binary) {
    return;
  }

  std::vector<int16_t> pcm_data = parse_pcm_data(message);
  if (pcm_data.empty()) {
    wsConnPtr->send(create_error_response("Invalid PCM data"));
    return;
  }

  session->audio.write(pcm_data.data(), pcm_data.size());

  const int frame_samples = session->vad->frame_samples();

  auto send_segment = [session, wsConnPtr](std::vector<int16_t>&& segment) {
    if (segment.empty()) return;
    if (session->stt_in_flight.load() >= GetConfig().stt_max_in_flight) {
      return;
    }

    std::string wav = BuildWav(segment, 16000);
    std::string boundary = MakeBoundary();
    std::string body = BuildMultipartBody(boundary, wav);

    auto client = drogon::HttpClient::newHttpClient(GetConfig().whisper_base);
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->setPath("/inference");
    req->setContentTypeString("multipart/form-data; boundary=" + boundary);
    req->setBody(std::move(body));

    session->stt_in_flight.fetch_add(1);

    std::weak_ptr<drogon::WebSocketConnection> weak_conn = wsConnPtr;
    client->sendRequest(
        req, [session, weak_conn](drogon::ReqResult result,
                                  const drogon::HttpResponsePtr& resp) {
          session->stt_in_flight.fetch_sub(1);
          auto conn = weak_conn.lock();
          if (!conn || !conn->connected()) return;
          if (result != drogon::ReqResult::Ok || !resp) {
            Json::Value payload;
            payload["type"] = "error";
            payload["error"] = "STT request failed";
            conn->send(ToString(payload));
            return;
          }

          Json::Value json;
          std::string err;
          if (!ParseJson(resp->getBody(), json, err)) {
            Json::Value payload;
            payload["type"] = "error";
            payload["error"] = "STT JSON parse failed";
            conn->send(ToString(payload));
            return;
          }

          std::string text = json.get("text", "").asString();
          if (text.empty()) return;

          int64_t now = NowMs();
          {
            std::lock_guard<std::mutex> lock(session->lock);
            session->full.push_back({now, text});
            session->recent.push_back({now, text});
            PruneRecent(session->recent, now, GetConfig().recent_window_ms);
          }

          {
            Json::Value payload;
            payload["type"] = "stt";
            payload["text"] = text;
            conn->send(ToString(payload));
          }

          auto send_tts_only = [session, weak_conn](const std::string& reply) {
            auto& cfg = GetConfig();
            auto conn = weak_conn.lock();
            if (!conn || !conn->connected()) {
              return;
            }
            Json::Value payload;
            payload["type"] = "answer";
            payload["text"] = reply;
            conn->send(ToString(payload));
            if (cfg.attendance_audio_path.empty()) {
              std::cout << "[Attendance] audio path empty" << std::endl;
              return;
            }
            const std::string audio_bytes = ReadFileBinary(cfg.attendance_audio_path);
            std::cout << "[Attendance] audio bytes=" << audio_bytes.size()
                      << " path=" << cfg.attendance_audio_path << std::endl;
            if (!audio_bytes.empty() && conn && conn->connected()) {
              conn->send(audio_bytes, drogon::WebSocketMessageType::Binary);
            } else if (conn && conn->connected()) {
              Json::Value err;
              err["type"] = "error";
              err["error"] = "attendance audio file not found";
              conn->send(ToString(err));
            }
          };

          if (session->mode == SessionMode::kNote) {
            return;
          }
          // D 모드(lecture)는 STT만 수행하고 질문/답변/TTS는 하지 않음.
          if (session->mode != SessionMode::kDefense) {
            return;
          }

          if (IsAttendanceStart(text) && !session->attendance_mode) {
            session->attendance_mode = true;
            session->attendance_started_ms = now;
            std::cout << "[Attendance] start detected: " << text << std::endl;
          }
          if (session->attendance_mode && IsNameCalled(text)) {
            if (now - session->last_attendance_tts_ms >= 10000) {
              session->last_attendance_tts_ms = now;
              std::cout << "[Attendance] name matched -> TTS" << std::endl;
              send_tts_only("네!");
              // 출석 완료 - 더 이상 출석 모드 아님
              session->attendance_mode = false;
              std::cout << "[Attendance] completed, mode off" << std::endl;
              // 프론트엔드에 출석 완료 알림
              if (conn && conn->connected()) {
                Json::Value complete_msg;
                complete_msg["type"] = "attendance_complete";
                conn->send(ToString(complete_msg));
              }
            }
          }

          // "질문 있나요?" 감지 시 LLM으로 추천 질문 생성
          if (IsAskingForQuestions(text) && now - session->last_question_suggest_ts > 30000) {
            session->last_question_suggest_ts = now;
            std::cout << "[QuestionSuggest] detected: " << text << std::endl;
            
            // 비동기로 LLM 호출하여 질문 생성
            std::thread([session, weak_conn]() {
              auto conn = weak_conn.lock();
              if (!conn || !conn->connected()) return;
              
              auto& cfg = GetConfig();
              std::string transcript;
              {
                std::lock_guard<std::mutex> lg(session->lock);
                transcript = JoinText(session->full, 4000);
              }
              
              if (transcript.empty()) {
                std::cout << "[QuestionSuggest] No transcript to analyze" << std::endl;
                return;
              }
              
              const std::string QUESTION_PROMPT = R"(
당신은 대학생을 위한 학습 도우미입니다.
아래는 수업 내용의 STT 기록입니다. 이 내용을 바탕으로 학생이 교수님께 질문할 만한 좋은 질문 3개를 생성해주세요.

규칙:
1. 반드시 한국어로 작성하세요.
2. 각 질문은 새 줄에 "- "로 시작합니다.
3. 질문은 수업 내용의 핵심 개념을 이해하는 데 도움이 되어야 합니다.
4. 질문은 간결하게 1-2문장으로 작성하세요.

형식:
- 첫 번째 추천 질문
- 두 번째 추천 질문
- 세 번째 추천 질문
)";
              
              Json::Value req;
              req["model"] = cfg.llama_model;
              Json::Value messages(Json::arrayValue);
              Json::Value system;
              system["role"] = "system";
              system["content"] = QUESTION_PROMPT;
              Json::Value user;
              user["role"] = "user";
              user["content"] = "수업 내용:\n" + transcript;
              messages.append(system);
              messages.append(user);
              req["messages"] = messages;
              req["temperature"] = 0.7;
              
              auto request = drogon::HttpRequest::newHttpJsonRequest(req);
              request->setMethod(drogon::Post);
              request->setPath("/v1/chat/completions");
              
              auto client = drogon::HttpClient::newHttpClient(cfg.llama_base);
              client->sendRequest(request, [weak_conn](drogon::ReqResult r,
                                                        const drogon::HttpResponsePtr& resp) {
                auto conn = weak_conn.lock();
                if (!conn || !conn->connected()) return;
                
                if (r != drogon::ReqResult::Ok || !resp) {
                  std::cerr << "[QuestionSuggest] LLM request failed" << std::endl;
                  return;
                }
                
                Json::Value json;
                std::string err;
                if (!ParseJson(resp->getBody(), json, err)) {
                  std::cerr << "[QuestionSuggest] JSON parse error: " << err << std::endl;
                  return;
                }
                
                const auto& choices = json["choices"];
                if (!choices.isArray() || choices.empty()) return;
                
                std::string content = choices[0u]["message"]["content"].asString();
                std::cout << "[QuestionSuggest] LLM response: " << content << std::endl;
                
                // 질문들을 파싱하여 배열로 만들기
                Json::Value msg;
                msg["type"] = "question_suggestions";
                Json::Value questions(Json::arrayValue);
                
                std::istringstream iss(content);
                std::string line;
                while (std::getline(iss, line)) {
                  // "- " 로 시작하는 줄만 추출
                  size_t pos = line.find("- ");
                  if (pos != std::string::npos) {
                    std::string q = line.substr(pos + 2);
                    // 앞뒤 공백 제거
                    size_t start = q.find_first_not_of(" \t\r\n");
                    size_t end = q.find_last_not_of(" \t\r\n");
                    if (start != std::string::npos && end != std::string::npos) {
                      questions.append(q.substr(start, end - start + 1));
                    }
                  }
                }
                
                if (questions.empty()) {
                  std::cerr << "[QuestionSuggest] No questions parsed from response" << std::endl;
                  return;
                }
                
                msg["questions"] = questions;
                std::cout << "[QuestionSuggest] Sending " << questions.size() << " questions" << std::endl;
                conn->send(ToString(msg));
              });
            }).detach();
          }

          auto maybe_answer = [session, text, weak_conn]() {
            auto& cfg = GetConfig();
            std::cout << "[TTS] maybe_answer called. ref=" << (cfg.tts_ref_audio_path.empty() ? "EMPTY" : "OK")
                      << " prompt=" << (cfg.tts_prompt_text.empty() ? "EMPTY" : "OK") << std::endl;
            if (cfg.tts_ref_audio_path.empty() || cfg.tts_prompt_text.empty()) {
              return;
            }
            int64_t now = NowMs();
            if (now - session->last_question_ts < 3000) {
              std::cout << "[TTS] skipped: cooldown" << std::endl;
              return;
            }
            if (session->llm_busy.exchange(true)) {
              std::cout << "[TTS] skipped: llm_busy" << std::endl;
              return;
            }
            session->last_question_ts = now;

            std::thread([session, text, weak_conn, cfg]() mutable {
              auto conn = weak_conn.lock();
              if (!conn || !conn->connected()) {
                session->llm_busy.store(false);
                return;
              }

              std::string context;
              std::string prev_block;
              {
                std::lock_guard<std::mutex> lock(session->lock);
                context = JoinText(session->recent, 6000);
                // last two lines before current (if available)
                const int n = static_cast<int>(session->recent.size());
                std::vector<std::string> prev_lines;
                if (n >= 2) prev_lines.push_back(session->recent[n - 2].text);
                if (n >= 3) prev_lines.insert(prev_lines.begin(), session->recent[n - 3].text);
                // Build a compact block: prev2 + current
                std::ostringstream oss;
                if (!prev_lines.empty()) {
                  for (const auto& line : prev_lines) {
                    if (!line.empty()) oss << line << "\n";
                  }
                }
                if (!text.empty()) oss << text;
                prev_block = oss.str();
              }

              Json::Value req;
              req["model"] = cfg.llama_model;
              Json::Value messages(Json::arrayValue);
              Json::Value system;
              system["role"] = "system";
              system["content"] =
                  "당신은 대학 강의를 듣고 있는 학생입니다. "
                  "교수님이 질문하시면, 방금 들은 수업 내용을 바탕으로 자신있게 답변하세요.\n\n"
                  "규칙:\n"
                  "1. 수업에서 언급된 핵심 개념과 용어를 활용하여 답변\n"
                  "2. 2~3문장, 100자 내외로 답변\n"
                  "3. '~입니다', '~라고 배웠습니다', '~라고 이해했습니다' 같은 학생 말투 사용\n"
                  "4. 한국어로만 답변\n\n"
                  "주의:\n"
                  "- '네', '알아요', '모르겠어요' 같은 단답은 절대 금지\n"
                  "- 반드시 수업 내용에서 관련 개념을 찾아서 설명";
              Json::Value user;
              user["role"] = "user";
              user["content"] =
                  "[교수님 질문]\n" + prev_block +
                  "\n\n[최근 수업 내용]\n" + "\n" + context +
                  "\n\n위 수업 내용을 참고하여 교수님 질문에 답변하세요. "
                  "관련 개념을 포함한 2~3문장으로 답변하세요.";
              messages.append(system);
              messages.append(user);
              req["messages"] = messages;
              req["temperature"] = 0.5;
              std::cout << "[LLM] system: " << system["content"].asString() << std::endl;
              std::cout << "[LLM] user: " << user["content"].asString() << std::endl;

              auto client = drogon::HttpClient::newHttpClient(cfg.llama_base);
              auto request = drogon::HttpRequest::newHttpJsonRequest(req);
              request->setMethod(drogon::Post);
              request->setPath("/v1/chat/completions");

              std::promise<std::string> promise;
              client->sendRequest(request, [&promise](drogon::ReqResult r,
                                                      const drogon::HttpResponsePtr& resp) {
                if (r != drogon::ReqResult::Ok || !resp) {
                  promise.set_value("");
                  return;
                }
                Json::Value json;
                std::string err;
                if (!ParseJson(resp->getBody(), json, err)) {
                  promise.set_value("");
                  return;
                }
                const auto& choices = json["choices"];
                if (!choices.isArray() || choices.empty()) {
                  promise.set_value("");
                  return;
                }
                promise.set_value(choices[0]["message"]["content"].asString());
              });

              std::string answer = promise.get_future().get();
              if (answer.empty()) {
                if (conn && conn->connected()) {
                  Json::Value payload;
                  payload["type"] = "error";
                  payload["error"] = "LLM answer failed";
                  conn->send(ToString(payload));
                }
                std::cout << "[TTS] LLM answer failed" << std::endl;
                session->llm_busy.store(false);
                return;
              }

              std::cout << "[LLM] answer: " << answer << std::endl;
              std::cout << "[TTS] LLM answer OK, length=" << answer.size() << std::endl;
              if (conn && conn->connected()) {
                Json::Value payload;
                payload["type"] = "answer";
                payload["text"] = answer;
                conn->send(ToString(payload));
              }

              Json::Value tts_req;
              tts_req["text"] = answer;
              tts_req["text_lang"] = cfg.tts_text_lang;
              tts_req["ref_audio_path"] = cfg.tts_ref_audio_path;
              tts_req["prompt_text"] = cfg.tts_prompt_text;
              tts_req["prompt_lang"] = cfg.tts_prompt_lang;
              tts_req["streaming_mode"] = false;

              auto tts_client = drogon::HttpClient::newHttpClient(cfg.tts_base);
              auto tts_request = drogon::HttpRequest::newHttpJsonRequest(tts_req);
              tts_request->setMethod(drogon::Post);
              tts_request->setPath("/tts");

              std::promise<std::string> tts_promise;
              tts_client->sendRequest(
                  tts_request, [&tts_promise](drogon::ReqResult r,
                                              const drogon::HttpResponsePtr& resp) {
                    if (r != drogon::ReqResult::Ok || !resp) {
                      tts_promise.set_value("");
                      return;
                    }
                    tts_promise.set_value(std::string(resp->getBody()));
                  });

              std::string audio_bytes = tts_promise.get_future().get();
              std::cout << "[TTS] TTS bytes=" << audio_bytes.size() << std::endl;
              if (!audio_bytes.empty() && conn && conn->connected()) {
                conn->send(audio_bytes, drogon::WebSocketMessageType::Binary);
              }

              session->llm_busy.store(false);
            }).detach();
          };

          if (session->trigger.IsTriggered(text)) {
            std::cout << "[Trigger] keyword matched, text: " << text << std::endl;
            maybe_answer();
          }
        });
  };

  while (session->audio.get_stored_samples() >= static_cast<size_t>(frame_samples)) {
    std::vector<int16_t> frame;
    if (session->audio.read(frame_samples, frame) == 0) {
      break;
    }
    session->vad->PushFrame(frame, send_segment);
  }
}

void AudioStreamController::handleConnectionClosed(
    const drogon::WebSocketConnectionPtr& wsConnPtr) {
  std::cout << "[AudioStreamController] Connection closed from "
            << wsConnPtr->peerAddr().toIpPort() << std::endl;
}

std::vector<int16_t> AudioStreamController::parse_pcm_data(
    const std::string& binary_data) {
  std::vector<int16_t> pcm_data;
  if (binary_data.size() % 2 != 0) {
    std::cerr << "[AudioStreamController] Invalid binary data size" << std::endl;
    return pcm_data;
  }

  size_t num_samples = binary_data.size() / sizeof(int16_t);
  pcm_data.resize(num_samples);
  std::memcpy(pcm_data.data(), binary_data.data(), binary_data.size());
  return pcm_data;
}

std::string AudioStreamController::create_error_response(const std::string& error) {
  Json::Value payload;
  payload["type"] = "error";
  payload["error"] = error;
  return ToString(payload);
}

std::string AudioStreamController::create_info_response(const std::string& type,
                                                        const std::string& message) {
  Json::Value payload;
  payload["type"] = type;
  payload["message"] = message;
  return ToString(payload);
}

std::string AudioStreamController::create_text_response(const std::string& type,
                                                        const std::string& text,
                                                        const std::string& request_id) {
  Json::Value payload;
  payload["type"] = type;
  payload["text"] = text;
  if (!request_id.empty()) {
    payload["request_id"] = request_id;
  }
  return ToString(payload);
}
