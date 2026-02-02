#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <functional>
#include <future>
#include <mutex>

// Forward declaration (whisper.cpp 라이브러리)
struct whisper_context;
struct whisper_params;

/**
 * @class STTService
 * @brief whisper.cpp를 래핑하여 실시간 음성 인식을 수행하는 서비스
 * 
 * 특징:
 * - whisper.h 라이브러리와 직접 연동
 * - GPU(CUDA) 가속 지원
 * - 비동기 추론 (std::async / std::future)
 * - 콜백 지원
 */
class STTService {
public:
    /**
     * @struct Config
     * @brief STTService 설정
     */
    struct Config {
        std::string model_path = "models/ggml-large-v3.bin";
        std::string language = "ko";  // 한국어
        int device_id = 0;            // GPU 디바이스 ID
        int n_threads = 4;            // CPU 스레드 수
        int beam_size = 5;            // 빔 서치 크기
        int sample_rate = 16000;      // Hz
        bool use_gpu = true;          // GPU 가속 활성화
    };

    /**
     * @struct TranscriptionResult
     * @brief STT 결과
     */
    struct TranscriptionResult {
        std::string text;
        std::string language;
        double duration_seconds = 0.0;
        bool success = false;
        std::string error_message;
    };

    explicit STTService(const Config& config);
    STTService();
    ~STTService();

    /**
     * @brief 서비스 초기화 (모델 로드)
     * @return 성공 여부
     */
    bool initialize();

    /**
     * @brief 오디오를 텍스트로 변환 (동기)
     * @param audio_data float32 오디오 데이터 (16kHz, Mono)
     * @return TranscriptionResult
     */
    TranscriptionResult transcribe_sync(const std::vector<float>& audio_data);

    /**
     * @brief 오디오를 텍스트로 변환 (비동기)
     * @param audio_data float32 오디오 데이터
     * @return std::future<TranscriptionResult>
     */
    std::future<TranscriptionResult> transcribe_async(
        const std::vector<float>& audio_data
    );

    /**
     * @brief 서비스 상태 확인
     * @return 준비 여부
     */
    bool is_ready() const { return is_ready_; }

    /**
     * @brief 현재 설정 반환
     */
    const Config& get_config() const { return config_; }

    /**
     * @brief 서비스 종료
     */
    void shutdown();

private:
    // 설정 및 상태
    Config config_;
    bool is_ready_ = false;
    whisper_context* ctx_ = nullptr;
    mutable std::mutex state_lock_;

    /**
     * @brief 실제 추론 실행 (내부 함수)
     */
    TranscriptionResult do_inference(const std::vector<float>& audio_data);

    /**
     * @brief 모델 파일 검증
     */
    bool validate_model_file() const;

    /**
     * @brief whisper 파라미터 설정
     */
    void setup_whisper_params(whisper_params& params) const;
};
