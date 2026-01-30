#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace cd::client {

class SleepDetector {
 public:
  SleepDetector(float ear_threshold, int consecutive_frames);

  float ComputeEAR(const std::vector<cv::Point2f>& eye) const;
  bool Update(const std::vector<cv::Point2f>& left_eye,
              const std::vector<cv::Point2f>& right_eye);

  bool IsSleeping() const;

 private:
  float ear_threshold_;
  int consecutive_frames_;
  int closed_frames_;
  bool sleeping_;
};

}  // namespace cd::client
