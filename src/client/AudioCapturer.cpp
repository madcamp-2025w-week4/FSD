#include "client/AudioCapturer.h"

#include <chrono>

namespace cd::client {

AudioCapturer::AudioCapturer() : running_(false) {}

AudioCapturer::~AudioCapturer() {
  Stop();
}

void AudioCapturer::Start() {
  if (running_.exchange(true)) {
    return;
  }
#if defined(CD_ENABLE_PORTAUDIO)
  if (Pa_Initialize() == paNoError) {
    pa_initialized_ = true;
    PaStreamParameters input_params{};
    input_params.device = Pa_GetDefaultInputDevice();
    input_params.channelCount = 1;
    input_params.sampleFormat = paFloat32;
    input_params.suggestedLatency =
        Pa_GetDeviceInfo(input_params.device)->defaultLowInputLatency;
    input_params.hostApiSpecificStreamInfo = nullptr;

    constexpr double kSampleRate = 16000.0;
    constexpr unsigned long kFramesPerBuffer = 160;
    if (Pa_OpenStream(&stream_, &input_params, nullptr, kSampleRate,
                      kFramesPerBuffer, paClipOff, nullptr, nullptr) == paNoError) {
      Pa_StartStream(stream_);
    }
  }
#endif
  worker_ = std::thread(&AudioCapturer::CaptureLoop, this);
}

void AudioCapturer::Stop() {
  if (!running_.exchange(false)) {
    return;
  }
  if (worker_.joinable()) {
    worker_.join();
  }
#if defined(CD_ENABLE_PORTAUDIO)
  if (stream_ != nullptr) {
    Pa_StopStream(stream_);
    Pa_CloseStream(stream_);
    stream_ = nullptr;
  }
  if (pa_initialized_) {
    Pa_Terminate();
    pa_initialized_ = false;
  }
#endif
}

bool AudioCapturer::IsRunning() const {
  return running_.load();
}

void AudioCapturer::SetAudioCallback(AudioCallback callback) {
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  callback_ = std::move(callback);
}

void AudioCapturer::CaptureLoop() {
  using namespace std::chrono_literals;
  while (running_.load()) {
    AudioCallback callback_copy;
    std::vector<float> local_buffer;
    {
      std::lock_guard<std::mutex> lock(buffer_mutex_);
#if defined(CD_ENABLE_PORTAUDIO)
      constexpr unsigned long kFramesPerBuffer = 160;
      buffer_.resize(kFramesPerBuffer);
      if (stream_ != nullptr) {
        Pa_ReadStream(stream_, buffer_.data(), kFramesPerBuffer);
      }
      local_buffer = buffer_;
#else
      buffer_.clear();
#endif
      callback_copy = callback_;
    }
    if (callback_copy && !local_buffer.empty()) {
      callback_copy(local_buffer, 16000, 1);
    }
    std::this_thread::sleep_for(10ms);
  }
}

}  // namespace cd::client
