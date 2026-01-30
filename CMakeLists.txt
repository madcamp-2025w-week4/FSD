cmake_minimum_required(VERSION 3.20)

project(ClassDefense LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_GRPC "Enable gRPC/Protobuf build" OFF)
option(ENABLE_OPENCV "Enable OpenCV build" OFF)
option(ENABLE_PORTAUDIO "Enable PortAudio build" OFF)
option(ENABLE_IMGUI "Enable Dear ImGui build" OFF)
option(ENABLE_GLFW "Enable GLFW for ImGui backend" OFF)
option(ENABLE_OPENGL "Enable OpenGL for ImGui backend" OFF)

set(CD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CD_PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)

add_library(cd_shared STATIC
  src/shared/Placeholder.cpp
)

target_include_directories(cd_shared PUBLIC ${CD_INCLUDE_DIR})

add_executable(cd_client
  src/client/main.cpp
  src/client/AudioCapturer.cpp
)

target_include_directories(cd_client PRIVATE ${CD_INCLUDE_DIR})
target_compile_definitions(cd_client PRIVATE
  $<$<BOOL:${ENABLE_GRPC}>:CD_ENABLE_GRPC>
  $<$<BOOL:${ENABLE_PORTAUDIO}>:CD_ENABLE_PORTAUDIO>
  $<$<BOOL:${ENABLE_IMGUI}>:CD_ENABLE_IMGUI>
  $<$<BOOL:${ENABLE_OPENCV}>:CD_ENABLE_OPENCV>
  $<$<AND:$<BOOL:${ENABLE_IMGUI}>,$<BOOL:${ENABLE_GLFW}>,$<BOOL:${ENABLE_OPENGL}>>:CD_ENABLE_IMGUI_BACKEND>
)
target_link_libraries(cd_client PRIVATE cd_shared)

add_executable(cd_server
  src/server/stt_server.cpp
  src/server/TriggerEngine.cpp
)

target_include_directories(cd_server PRIVATE ${CD_INCLUDE_DIR})
target_compile_definitions(cd_server PRIVATE
  $<$<BOOL:${ENABLE_GRPC}>:CD_ENABLE_GRPC>
  $<$<BOOL:${ENABLE_OPENCV}>:CD_ENABLE_OPENCV>
)
target_link_libraries(cd_server PRIVATE cd_shared)

if(ENABLE_GRPC)
  find_package(Protobuf CONFIG REQUIRED)
  find_package(gRPC CONFIG REQUIRED)

  set(PROTO_FILES ${CD_PROTO_DIR}/classdefense.proto)

  protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
  grpc_generate_cpp(GRPC_SRCS GRPC_HDRS ${PROTO_FILES})

  add_library(cd_proto STATIC ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS})
  target_include_directories(cd_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(cd_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)

  target_link_libraries(cd_client PRIVATE cd_proto)
  target_link_libraries(cd_server PRIVATE cd_proto)
endif()

if(ENABLE_OPENCV)
  find_package(OpenCV REQUIRED)
  target_sources(cd_client PRIVATE src/client/SleepDetector.cpp)
  target_link_libraries(cd_client PRIVATE ${OpenCV_LIBS})
endif()

if(ENABLE_PORTAUDIO)
  find_package(PortAudio REQUIRED)
  target_link_libraries(cd_client PRIVATE PortAudio::PortAudio)
endif()

if(ENABLE_IMGUI)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui/CMakeLists.txt)
    add_subdirectory(third_party/imgui)
    target_link_libraries(cd_client PRIVATE imgui)
    if(ENABLE_GLFW AND ENABLE_OPENGL)
      target_sources(cd_client PRIVATE
        third_party/imgui/backends/imgui_impl_glfw.cpp
        third_party/imgui/backends/imgui_impl_opengl3.cpp
      )
      target_include_directories(cd_client PRIVATE third_party/imgui/backends)
    endif()
  else()
    message(FATAL_ERROR "ImGui not found. Place it under third_party/imgui or disable ENABLE_IMGUI.")
  endif()
endif()

if(ENABLE_GLFW)
  find_package(glfw3 REQUIRED)
  target_link_libraries(cd_client PRIVATE glfw)
endif()

if(ENABLE_OPENGL)
  find_package(OpenGL REQUIRED)
  target_link_libraries(cd_client PRIVATE OpenGL::GL)
endif()

include(CPack)

set(CPACK_PACKAGE_NAME "ClassDefense")
set(CPACK_PACKAGE_VENDOR "ClassDefense")
set(CPACK_PACKAGE_CONTACT "dev@classdefense.local")
set(CPACK_PACKAGE_VERSION "0.1.0")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}")

if(APPLE)
  set(CPACK_GENERATOR "DragNDrop")
elseif(WIN32)
  set(CPACK_GENERATOR "NSIS")
endif()

install(TARGETS cd_client cd_server
  RUNTIME DESTINATION .
)
