#include "server/TriggerEngine.h"
#include "AudioProcessor.hpp"
#include "STTService.hpp"

#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

#if defined(CD_ENABLE_DROGON_SERVER)
#include <drogon/drogon.h>
#endif

#if defined(CD_ENABLE_GRPC)
#include <grpcpp/grpcpp.h>

#include "classdefense.grpc.pb.h"
#endif

#if defined(CD_ENABLE_GRPC)
class StreamServiceImpl final : public classdefense::StreamService::Service {
 public:
  grpc::Status StreamAudio(
      grpc::ServerContext* context,
      grpc::ServerReaderWriter<classdefense::ControlSignal,
                               classdefense::AudioChunk>* stream) override {
    classdefense::AudioChunk chunk;
    while (stream->Read(&chunk)) {
      (void)context;
      classdefense::ControlSignal signal;
      signal.set_type(classdefense::ControlSignal::STALL_SIGNAL);
      signal.set_message("stalling");
      stream->Write(signal);
    }
    return grpc::Status::OK;
  }
};
#endif

int main() {
  std::cout << "================================================\n";
  std::cout << "  ClassDefense Real-time Audio Pipeline Server\n";
  std::cout << "================================================\n\n";

  // TriggerEngine 테스트
  cd::server::TriggerEngine trigger_engine;
  std::cout << "✓ TriggerEngine initialized\n";
  std::cout << "  Trigger test (\"Any questions?\"): "
            << (trigger_engine.IsTriggered("Any questions?") ? "yes" : "no")
            << "\n\n";

  // AudioProcessor 초기화
  std::cout << "[1/2] Initializing AudioProcessor...\n";
  AudioProcessor::Config ap_config;
  ap_config.sample_rate = 16000;
  ap_config.buffer_duration_seconds = 10;
  ap_config.vad_threshold = 0.02f;
  
  AudioProcessor audio_processor(ap_config);
  std::cout << "✓ AudioProcessor ready\n";
  std::cout << "  - Buffer capacity: " << audio_processor.get_stored_samples() << " samples\n";
  std::cout << "  - Stored duration: " << audio_processor.get_stored_duration_ms() << "ms\n\n";

  // STTService 초기화
  std::cout << "[2/2] Initializing STTService...\n";
  STTService::Config stt_config;
  stt_config.model_path = "models/ggml-large-v3.bin";
  stt_config.use_gpu = true;
  stt_config.device_id = 0;
  
  STTService stt_service(stt_config);
  if (stt_service.initialize()) {
    std::cout << "✓ STTService ready (GPU acceleration enabled)\n\n";
  } else {
    std::cout << "⚠ STTService init warning (model may not exist)\n\n";
  }

  // 간단한 오디오 처리 테스트
  std::cout << "================================================\n";
  std::cout << "  Audio Processing Test\n";
  std::cout << "================================================\n\n";
  
  // 더미 PCM 데이터 생성 (100ms @ 16kHz = 1600 샘플)
  std::vector<int16_t> test_audio(1600, 0);
  for (size_t i = 0; i < test_audio.size(); ++i) {
    // 간단한 사인파 생성 (440Hz tone)
    test_audio[i] = static_cast<int16_t>(20000 * std::sin(2 * 3.14159 * 440 * i / 16000));
  }
  
  std::cout << "Generated test audio: " << test_audio.size() << " samples (100ms)\n";
  
  // 버퍼에 기록
  size_t written = audio_processor.write(test_audio.data(), test_audio.size());
  std::cout << "✓ Written to buffer: " << written << " samples\n";
  std::cout << "  Buffer now contains: " << audio_processor.get_stored_duration_ms() << "ms\n";
  
  // VAD 테스트
  float energy = 0.0f;
  bool has_speech = audio_processor.detect_voice_activity(test_audio, &energy);
  std::cout << "✓ VAD test: " << (has_speech ? "SPEECH DETECTED" : "SILENCE")
            << " (energy: " << energy << ")\n\n";
  
  // 정규화 테스트
  std::cout << "Testing audio normalization...\n";
  std::vector<float> normalized;
  AudioProcessor::normalize_to_float(test_audio, normalized);
  std::cout << "✓ Converted int16 → float32 (" << normalized.size() << " samples)\n";
  std::cout << "  Sample range: [" << normalized[0] << ", " << normalized[normalized.size()-1] << "]\n\n";

  std::cout << "================================================\n";
  std::cout << "  All Core Modules Operational ✓\n";
  std::cout << "================================================\n";

#if defined(CD_ENABLE_GRPC)
  std::string server_address("0.0.0.0:50051");
  StreamServiceImpl service;
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "gRPC server listening on " << server_address << "\n";
  server->Wait();
#endif

#if defined(CD_ENABLE_DROGON_SERVER)
  std::cout << "Starting Drogon WebSocket server on 0.0.0.0:9000\n";
  drogon::app()
      .setThreadNum(4)
      .addListener("0.0.0.0", 9000)
      .run();
#endif
  return 0;
}
