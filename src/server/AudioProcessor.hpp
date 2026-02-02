#pragma once

#include <vector>
#include <cstdint>
#include <mutex>
#include <memory>
#include <atomic>
#include <cmath>

/**
 * @class AudioProcessor
 * @brief Thread-safe 순환 버퍼 및 오디오 전처리(정규화, VAD)를 담당합니다.
 * 
 * 특징:
 * - std::vector 기반 순환 버퍼 (Lock-free 아님, std::mutex 사용)
 * - int16_t -> float32 변환 및 정규화 (LUFS 기반)
 * - 에너지 기반 음성 활동 감지 (VAD)
 */
class AudioProcessor {
public:
    /**
     * @struct Config
     * @brief AudioProcessor 설정
     */
    struct Config {
        int sample_rate = 16000;           // Hz
        int buffer_duration_seconds = 10;  // 버퍼가 보유할 최대 오디오 길이
        int channels = 1;                  // Mono
        float vad_threshold = 0.02f;       // 음성 활동 감지 임계값 (0~1)
        int vad_duration_ms = 300;         // 음성 활동 감지 최소 지속 시간
    };

    explicit AudioProcessor(const Config& config);
    AudioProcessor();
    ~AudioProcessor() = default;

    // 버퍼 관리
    /**
     * @brief 바이너리 PCM 데이터를 버퍼에 기록
     * @param data 16-bit 리틀 엔디안 PCM 데이터 포인터
     * @param num_samples 샘플 수
     * @return 기록된 샘플 수
     */
    size_t write(const int16_t* data, size_t num_samples);

    /**
     * @brief 버퍼에서 지정된 수의 샘플을 읽음
     * @param num_samples 읽을 샘플 수
     * @param output 출력 버퍼 (호출자가 할당해야 함)
     * @return 읽은 샘플 수 (부족하면 0)
     */
    size_t read(size_t num_samples, std::vector<int16_t>& output);

    /**
     * @brief 현재 버퍼에 저장된 샘플 수
     */
    size_t get_stored_samples() const;

    /**
     * @brief 현재 버퍼에 저장된 오디오의 지속 시간 (밀리초)
     */
    double get_stored_duration_ms() const;

    /**
     * @brief 버퍼 초기화
     */
    void clear();

    // 오디오 처리 유틸리티
    /**
     * @brief int16_t 데이터를 float32로 정규화 변환
     * @param input int16_t 입력 데이터
     * @param output float32 출력 데이터
     * @param target_db 목표 데시벨 (기본값: -20.0 LUFS)
     */
    static void normalize_to_float(
        const std::vector<int16_t>& input,
        std::vector<float>& output,
        float target_db = -20.0f
    );

    /**
     * @brief float32 데이터를 int16_t로 클리핑하며 변환
     * @param input float32 입력 데이터
     * @param output int16_t 출력 데이터
     */
    static void convert_to_int16(
        const std::vector<float>& input,
        std::vector<int16_t>& output
    );

    /**
     * @brief 음성 활동 감지 (VAD)
     * @param audio int16_t 오디오 데이터
     * @param energy_out 계산된 에너지 (선택사항)
     * @return 음성 활동 감지 여부
     */
    bool detect_voice_activity(
        const std::vector<int16_t>& audio,
        float* energy_out = nullptr
    );

    /**
     * @brief 오디오 에너지 계산
     * @param audio int16_t 오디오 데이터
     * @return 정규화된 에너지 (0~1)
     */
    static float compute_energy(const std::vector<int16_t>& audio);

private:
    // 버퍼 멤버
    std::vector<int16_t> buffer_;
    size_t write_pos_ = 0;
    size_t read_pos_ = 0;
    size_t samples_stored_ = 0;
    size_t max_samples_;

    // 설정 멤버
    Config config_;

    // 동기화 멤버
    mutable std::mutex buffer_lock_;

    // VAD 상태
    bool is_speaking_ = false;
    size_t silence_counter_ = 0;
    size_t vad_duration_samples_;
};
