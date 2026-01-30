#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(CD_ENABLE_PORTAUDIO)
#include <portaudio.h>
#endif

namespace cd::client {

class AudioCapturer {
 public:
  using AudioCallback =
      std::function<void(const std::vector<float>&, int sample_rate, int channels)>;

  AudioCapturer();
  ~AudioCapturer();

  void Start();
  void Stop();
  bool IsRunning() const;
  void SetAudioCallback(AudioCallback callback);

 private:
  void CaptureLoop();

  std::atomic<bool> running_;
  std::thread worker_;
  std::mutex buffer_mutex_;
  std::vector<float> buffer_;
  AudioCallback callback_;

#if defined(CD_ENABLE_PORTAUDIO)
  PaStream* stream_ = nullptr;
  bool pa_initialized_ = false;
#endif
};

}  // namespace cd::client
