#include "client/SleepDetector.h"

#include <cmath>

namespace cd::client {

SleepDetector::SleepDetector(float ear_threshold, int consecutive_frames)
    : ear_threshold_(ear_threshold),
      consecutive_frames_(consecutive_frames),
      closed_frames_(0),
      sleeping_(false) {}

float SleepDetector::ComputeEAR(const std::vector<cv::Point2f>& eye) const {
  if (eye.size() != 6) {
    return 0.0f;
  }
  const float a = cv::norm(eye[1] - eye[5]);
  const float b = cv::norm(eye[2] - eye[4]);
  const float c = cv::norm(eye[0] - eye[3]);
  if (c <= 0.0f) {
    return 0.0f;
  }
  return (a + b) / (2.0f * c);
}

bool SleepDetector::Update(const std::vector<cv::Point2f>& left_eye,
                           const std::vector<cv::Point2f>& right_eye) {
  const float left_ear = ComputeEAR(left_eye);
  const float right_ear = ComputeEAR(right_eye);
  const float ear = (left_ear + right_ear) * 0.5f;

  if (ear > 0.0f && ear < ear_threshold_) {
    closed_frames_++;
  } else {
    closed_frames_ = 0;
  }

  sleeping_ = (closed_frames_ >= consecutive_frames_);
  return sleeping_;
}

bool SleepDetector::IsSleeping() const {
  return sleeping_;
}

}  // namespace cd::client
