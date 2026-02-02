#include "AudioProcessor.hpp"
#include <cstring>
#include <algorithm>
#include <numeric>
#include <iostream>

AudioProcessor::AudioProcessor(const Config& config)
    : config_(config) {
    max_samples_ = config.sample_rate * config.buffer_duration_seconds * config.channels;
    buffer_.resize(max_samples_, 0);
    vad_duration_samples_ = (config.vad_duration_ms * config.sample_rate) / 1000;

    std::cout << "[AudioProcessor] Initialized: "
              << "sample_rate=" << config.sample_rate << "Hz, "
              << "buffer_size=" << max_samples_ << " samples, "
              << "vad_threshold=" << config.vad_threshold << std::endl;
}

AudioProcessor::AudioProcessor() : AudioProcessor(Config()) {}

size_t AudioProcessor::write(const int16_t* data, size_t num_samples) {
    if (!data || num_samples == 0) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(buffer_lock_);

    size_t remaining = num_samples;
    size_t offset = 0;

    while (remaining > 0) {
        size_t space_to_end = max_samples_ - write_pos_;
        size_t to_write = std::min(remaining, space_to_end);

        std::memcpy(
            buffer_.data() + write_pos_,
            data + offset,
            to_write * sizeof(int16_t)
        );

        write_pos_ = (write_pos_ + to_write) % max_samples_;
        remaining -= to_write;
        offset += to_write;
    }

    samples_stored_ = std::min(samples_stored_ + num_samples, max_samples_);

    return num_samples;
}

size_t AudioProcessor::read(size_t num_samples, std::vector<int16_t>& output) {
    std::lock_guard<std::mutex> lock(buffer_lock_);

    if (samples_stored_ < num_samples) {
        return 0;
    }

    output.resize(num_samples);
    size_t remaining = num_samples;
    size_t offset = 0;

    while (remaining > 0) {
        size_t available_to_end = max_samples_ - read_pos_;
        size_t to_read = std::min(remaining, available_to_end);

        std::memcpy(
            output.data() + offset,
            buffer_.data() + read_pos_,
            to_read * sizeof(int16_t)
        );

        read_pos_ = (read_pos_ + to_read) % max_samples_;
        remaining -= to_read;
        offset += to_read;
    }

    samples_stored_ -= num_samples;

    return num_samples;
}

size_t AudioProcessor::get_stored_samples() const {
    std::lock_guard<std::mutex> lock(buffer_lock_);
    return samples_stored_;
}

double AudioProcessor::get_stored_duration_ms() const {
    std::lock_guard<std::mutex> lock(buffer_lock_);
    return (static_cast<double>(samples_stored_) / config_.sample_rate) * 1000.0;
}

void AudioProcessor::clear() {
    std::lock_guard<std::mutex> lock(buffer_lock_);
    write_pos_ = 0;
    read_pos_ = 0;
    samples_stored_ = 0;
    std::cout << "[AudioProcessor] Buffer cleared" << std::endl;
}

void AudioProcessor::normalize_to_float(
    const std::vector<int16_t>& input,
    std::vector<float>& output,
    float target_db) {
    
    output.resize(input.size());

    // Float로 변환 (정규화)
    const float int16_max = 32768.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<float>(input[i]) / int16_max;
    }

    // RMS 계산
    float sum_sq = 0.0f;
    for (float sample : output) {
        sum_sq += sample * sample;
    }
    float rms = std::sqrt(sum_sq / output.size());

    if (rms < 1e-10f) {
        return;  // 무음 처리
    }

    // 데시벨 계산 및 게인 적용
    float current_db = 20.0f * std::log10(rms);
    float gain_db = target_db - current_db;
    float gain_linear = std::pow(10.0f, gain_db / 20.0f);

    // 게인 적용 및 클리핑
    for (float& sample : output) {
        sample = std::clamp(sample * gain_linear, -1.0f, 1.0f - 1e-6f);
    }
}

void AudioProcessor::convert_to_int16(
    const std::vector<float>& input,
    std::vector<int16_t>& output) {
    
    output.resize(input.size());

    const float int16_max = 32767.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        float clamped = std::clamp(input[i], -1.0f, 1.0f);
        output[i] = static_cast<int16_t>(clamped * int16_max);
    }
}

bool AudioProcessor::detect_voice_activity(
    const std::vector<int16_t>& audio,
    float* energy_out) {
    
    float energy = compute_energy(audio);
    
    if (energy_out) {
        *energy_out = energy;
    }

    bool has_speech = energy > config_.vad_threshold;

    if (has_speech) {
        silence_counter_ = 0;
        is_speaking_ = true;
    } else {
        silence_counter_ += audio.size();
        if (silence_counter_ > vad_duration_samples_) {
            is_speaking_ = false;
        }
    }

    return is_speaking_;
}

float AudioProcessor::compute_energy(const std::vector<int16_t>& audio) {
    if (audio.empty()) {
        return 0.0f;
    }

    const float int16_max = 32768.0f;
    float sum_sq = 0.0f;

    for (int16_t sample : audio) {
        float normalized = static_cast<float>(sample) / int16_max;
        sum_sq += normalized * normalized;
    }

    float energy = std::sqrt(sum_sq / audio.size());
    return std::clamp(energy, 0.0f, 1.0f);
}
