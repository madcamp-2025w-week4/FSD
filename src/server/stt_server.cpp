#include "server/TriggerEngine.h"

#include <iostream>
#include <memory>

#if defined(CD_ENABLE_GRPC)
#include <grpcpp/grpcpp.h>

#include "classdefense.grpc.pb.h"
#endif

#if defined(CD_ENABLE_GRPC)
class StreamServiceImpl final : public classdefense::StreamService::Service {
 public:
  grpc::Status StreamAudio(
      grpc::ServerContext* context,
      grpc::ServerReaderWriter<classdefense::ControlSignal,
                               classdefense::AudioChunk>* stream) override {
    classdefense::AudioChunk chunk;
    while (stream->Read(&chunk)) {
      (void)context;
      classdefense::ControlSignal signal;
      signal.set_type(classdefense::ControlSignal::STALL_SIGNAL);
      signal.set_message("stalling");
      stream->Write(signal);
    }
    return grpc::Status::OK;
  }
};
#endif

int main() {
  cd::server::TriggerEngine trigger_engine;
  std::cout << "ClassDefense server skeleton running.\n";
  std::cout << "Trigger test: "
            << (trigger_engine.IsTriggered("Any questions?") ? "yes" : "no")
            << "\n";

#if defined(CD_ENABLE_GRPC)
  std::string server_address("0.0.0.0:50051");
  StreamServiceImpl service;
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "gRPC server listening on " << server_address << "\n";
  server->Wait();
#endif
  return 0;
}
