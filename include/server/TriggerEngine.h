#pragma once

#include <regex>
#include <string>

namespace cd::server {

class TriggerEngine {
 public:
  TriggerEngine();

  bool IsTriggered(const std::string& transcript);

 private:
  std::regex trigger_regex_;
};

}  // namespace cd::server
