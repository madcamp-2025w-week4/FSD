#include "server/TriggerEngine.h"

namespace cd::server {

TriggerEngine::TriggerEngine()
    : trigger_regex_("(do you know|any questions\\?|알아요)",
                     std::regex_constants::icase) {}

bool TriggerEngine::IsTriggered(const std::string& transcript) {
  return std::regex_search(transcript, trigger_regex_);
}

}  // namespace cd::server
