#pragma once

#include <drogon/WebSocketController.h>
#include <memory>
#include <vector>
#include <cstdint>
#include <deque>
#include <mutex>
#include <atomic>
#include <string>

// Forward declarations
class AudioProcessor;
class STTService;

/**
 * @class AudioStreamController
 * @brief Drogon 기반 WebSocket 컨트롤러
 * 
 * 경로: /ws/audio
 * 기능:
 * - 클라이언트로부터 16kHz 16-bit Mono PCM 바이너리 메시지 수신
 * - AudioProcessor 버퍼에 데이터 전달
 * - STT 결과를 JSON으로 클라이언트에 응답
 */
class AudioStreamController : public drogon::WebSocketController<AudioStreamController> {
public:
    AudioStreamController();
    ~AudioStreamController();

    // WebSocket 엔드포인트 정의
    WS_PATH_LIST_BEGIN
        WS_PATH_ADD("/ws/audio");
    WS_PATH_LIST_END

    // WebSocket 콜백 메서드들
    /**
     * @brief 클라이언트 연결 시 호출
     */
    virtual void handleNewConnection(
        const drogon::HttpRequestPtr& req,
        const drogon::WebSocketConnectionPtr& wsConnPtr
    ) override;

    /**
     * @brief 텍스트/바이너리 메시지 수신 시 호출
     */
    virtual void handleNewMessage(
        const drogon::WebSocketConnectionPtr& wsConnPtr,
        std::string&& message,
        const drogon::WebSocketMessageType& type
    ) override;

    /**
     * @brief 클라이언트 연결 종료 시 호출
     */
    virtual void handleConnectionClosed(
        const drogon::WebSocketConnectionPtr& wsConnPtr
    ) override;

private:
    // 추론 임계값 (1초 분량의 오디오 = 16000 샘플)
    static constexpr int INFERENCE_SAMPLES = 16000;

    /**
     * @brief 수신된 바이너리 데이터를 int16_t 배열로 파싱
     */
    std::vector<int16_t> parse_pcm_data(const std::string& binary_data);

    std::string create_error_response(const std::string& error);
    std::string create_info_response(const std::string& type, const std::string& message);
    std::string create_text_response(const std::string& type, const std::string& text,
                                     const std::string& request_id = "");
};
