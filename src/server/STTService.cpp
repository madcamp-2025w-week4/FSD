#include "STTService.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <cmath>

// whisper.cpp 라이브러리 헤더 (실제 환경에서 설치 필요)
// #include "whisper.h"

STTService::STTService(const Config& config)
    : config_(config) {
    std::cout << "[STTService] Initialized with config: "
              << "model=" << config.model_path << ", "
              << "language=" << config.language << ", "
              << "use_gpu=" << (config.use_gpu ? "true" : "false") << std::endl;
}

STTService::STTService() : STTService(Config()) {}

STTService::~STTService() {
    shutdown();
}

bool STTService::initialize() {
    std::lock_guard<std::mutex> lock(state_lock_);

    // 모델 파일 검증
    if (!validate_model_file()) {
        std::cerr << "[STTService] Model file not found: " << config_.model_path << std::endl;
        return false;
    }

    // whisper 컨텍스트 생성
    // ctx_ = whisper_init_from_file(config_.model_path.c_str());
    // 
    // if (!ctx_) {
    //     std::cerr << "[STTService] Failed to initialize whisper context" << std::endl;
    //     return false;
    // }

    // GPU 가속 설정 (CUDA 사용 가능 시)
    // if (config_.use_gpu) {
    //     whisper_ctx_init_with_gpu_device(ctx_, config_.device_id);
    // }

    is_ready_ = true;
    std::cout << "[STTService] ✓ Service initialized and ready" << std::endl;

    return true;
}

bool STTService::validate_model_file() const {
    std::ifstream file(config_.model_path, std::ios::binary);
    return file.good();
}

void STTService::setup_whisper_params(whisper_params& params) const {
    // whisper 파라미터 설정
    // params.language = config_.language.c_str();
    // params.n_threads = config_.n_threads;
    // params.use_gpu = config_.use_gpu;
    // params.device = config_.device_id;
    // params.beam_size = config_.beam_size;
}

STTService::TranscriptionResult STTService::transcribe_sync(
    const std::vector<float>& audio_data) {
    
    if (!is_ready_) {
        return TranscriptionResult{
            .success = false,
            .error_message = "STT service is not ready"
        };
    }

    return do_inference(audio_data);
}

std::future<STTService::TranscriptionResult> STTService::transcribe_async(
    const std::vector<float>& audio_data) {
    
    return std::async(std::launch::async, [this, audio_data]() {
        return do_inference(audio_data);
    });
}

STTService::TranscriptionResult STTService::do_inference(
    const std::vector<float>& audio_data) {
    
    auto start_time = std::chrono::high_resolution_clock::now();

    TranscriptionResult result;
    result.duration_seconds = audio_data.size() / static_cast<double>(config_.sample_rate);

    // whisper 추론 실행
    // whisper_full(ctx_, params, audio_data.data(), audio_data.size());
    //
    // // 결과 추출
    // int n_segments = whisper_full_n_segments(ctx_);
    // std::string full_text;
    // for (int i = 0; i < n_segments; ++i) {
    //     const char* text = whisper_full_get_segment_text(ctx_, i);
    //     if (text) {
    //         full_text += text;
    //     }
    // }
    //
    // result.text = full_text;
    // result.language = config_.language;
    // result.success = true;

    // 추론 시간 측정
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    ).count();

    std::cout << "[STTService] Inference completed in " << elapsed_ms << "ms" << std::endl;

    return result;
}

void STTService::shutdown() {
    std::lock_guard<std::mutex> lock(state_lock_);

    if (ctx_) {
        // whisper_free(ctx_);
        ctx_ = nullptr;
    }

    is_ready_ = false;
    std::cout << "[STTService] Service shut down" << std::endl;
}
