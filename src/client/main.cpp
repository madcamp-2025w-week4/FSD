#include "client/AudioCapturer.h"

#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#if defined(CD_ENABLE_GRPC)
#include <grpcpp/grpcpp.h>

#include "classdefense.grpc.pb.h"
#endif

#if defined(CD_ENABLE_IMGUI)
#include "imgui.h"
#if defined(CD_ENABLE_IMGUI_BACKEND)
#include <GLFW/glfw3.h>

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#endif
#endif

#if defined(CD_ENABLE_OPENCV)
#include <opencv2/opencv.hpp>

#include "client/SleepDetector.h"
#endif

#if defined(CD_ENABLE_GRPC)
std::unique_ptr<classdefense::StreamService::Stub> CreateGrpcStub(
    const std::string& address) {
  auto channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  return classdefense::StreamService::NewStub(channel);
}
#endif

int main() {
  cd::client::AudioCapturer capturer;
#if defined(CD_ENABLE_IMGUI) && defined(CD_ENABLE_IMGUI_BACKEND)
  GLFWwindow* window = nullptr;
#endif

#if defined(CD_ENABLE_GRPC)
  std::unique_ptr<classdefense::StreamService::Stub> stub;
  std::unique_ptr<grpc::ClientReaderWriter<classdefense::AudioChunk,
                                           classdefense::ControlSignal>>
      stream;
  std::mutex stream_mutex;
  std::thread signal_thread;

  stub = CreateGrpcStub("localhost:50051");
  grpc::ClientContext context;
  stream = stub->StreamAudio(&context);
  signal_thread = std::thread([&stream]() {
    classdefense::ControlSignal signal;
    while (stream->Read(&signal)) {
      std::cout << "Control signal: " << signal.message() << "\n";
    }
  });
#endif

#if defined(CD_ENABLE_GRPC)
  capturer.SetAudioCallback(
      [&stream, &stream_mutex](const std::vector<float>& samples, int sample_rate,
                               int channels) {
        if (!stream) {
          return;
        }
        classdefense::AudioChunk chunk;
        chunk.set_pcm_f32le(samples.data(),
                            samples.size() * static_cast<int>(sizeof(float)));
        chunk.set_sample_rate_hz(sample_rate);
        chunk.set_channels(channels);
        std::lock_guard<std::mutex> lock(stream_mutex);
        stream->Write(chunk);
      });
#else
  capturer.SetAudioCallback(nullptr);
#endif

  capturer.Start();

#if defined(CD_ENABLE_IMGUI) && defined(CD_ENABLE_IMGUI_BACKEND)
  if (!glfwInit()) {
    std::cerr << "GLFW init failed.\n";
    return 1;
  }
  const char* glsl_version = "#version 150";
  window = glfwCreateWindow(800, 600, "ClassDefense", nullptr, nullptr);
  if (!window) {
    std::cerr << "GLFW window creation failed.\n";
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);
#elif defined(CD_ENABLE_IMGUI)
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
#endif

#if defined(CD_ENABLE_OPENCV)
  cd::client::SleepDetector detector(0.25f, 15);
  cv::VideoCapture cap(0);
  if (cap.isOpened()) {
    cv::Mat frame;
    cap >> frame;
    std::vector<cv::Point2f> dummy_eye(6, cv::Point2f(0.0f, 0.0f));
    detector.Update(dummy_eye, dummy_eye);
  }
  cap.release();
#endif

  std::cout << "ClassDefense client skeleton running.\n";
  std::cout << "Press Enter to exit.\n";
  std::cin.get();

  capturer.Stop();

#if defined(CD_ENABLE_IMGUI) && defined(CD_ENABLE_IMGUI_BACKEND)
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
#elif defined(CD_ENABLE_IMGUI)
  ImGui::DestroyContext();
#endif

#if defined(CD_ENABLE_GRPC)
  if (stream) {
    std::lock_guard<std::mutex> lock(stream_mutex);
    stream->WritesDone();
  }
  if (signal_thread.joinable()) {
    signal_thread.join();
  }
#endif
  return 0;
}
