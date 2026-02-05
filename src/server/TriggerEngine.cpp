#include "server/TriggerEngine.h"

namespace cd::server {

TriggerEngine::TriggerEngine()
    : trigger_regex_("(대답해 볼까요?|알아요?|이거 알아요?|이거 아나요?|설명할 수 있어요?|이해했어요?|이해했나요?|이해했습니까?|알고 있나요?|알고 있습니까?|이해할 수 있어요?|이해할 수 있나요?|이해할 수 있습니까?|설명할 수 있나요?|설명할 수 있습니까?)",
                     std::regex_constants::icase) {}

bool TriggerEngine::IsTriggered(const std::string& transcript) {
  return std::regex_search(transcript, trigger_regex_);
}

}  // namespace cd::server
