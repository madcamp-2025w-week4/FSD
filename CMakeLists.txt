cmake_minimum_required(VERSION 3.20)

project(ClassDefense LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add system cmake paths
list(APPEND CMAKE_PREFIX_PATH 
  /usr/lib/x86_64-linux-gnu/cmake
  /usr/lib/cmake
)

# Simple approach: manually link Drogon libraries
# Skip CONFIG mode to avoid PostgreSQL dependency hell
set(DROGON_INCLUDE_DIRS "/usr/include" "/usr/include/jsoncpp")
set(DROGON_LIBRARIES "drogon" "trantor" "jsoncpp")
set(JSONCPP_LIBRARIES "jsoncpp")

option(ENABLE_GRPC "Enable gRPC/Protobuf build" OFF)
option(ENABLE_OPENCV "Enable OpenCV build" OFF)
option(ENABLE_PORTAUDIO "Enable PortAudio build" OFF)
option(ENABLE_IMGUI "Enable Dear ImGui build" OFF)
option(ENABLE_GLFW "Enable GLFW for ImGui backend" OFF)
option(ENABLE_OPENGL "Enable OpenGL for ImGui backend" OFF)
option(ENABLE_WHISPER "Enable whisper.cpp integration" ON)
option(ENABLE_DROGON_SERVER "Enable Drogon-based audio server" ON)
option(BUILD_DROGON_SHARED_LIBS "Build Drogon as shared libraries" ON)
option(BUILD_POSTGRESQL_SUPPORT "Build with PostgreSQL support" OFF)

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
  src/server/AudioProcessor.cpp
  src/server/STTService.cpp
  src/server/AudioStreamController.cpp
)

target_include_directories(cd_server PRIVATE ${CD_INCLUDE_DIR})
target_compile_definitions(cd_server PRIVATE
  $<$<BOOL:${ENABLE_GRPC}>:CD_ENABLE_GRPC>
  $<$<BOOL:${ENABLE_OPENCV}>:CD_ENABLE_OPENCV>
  $<$<BOOL:${ENABLE_DROGON_SERVER}>:CD_ENABLE_DROGON_SERVER>
  $<$<BOOL:${ENABLE_WHISPER}>:CD_ENABLE_WHISPER>
)
target_link_libraries(cd_server PRIVATE cd_shared)

# Link Drogon and jsoncpp for audio server
if(ENABLE_DROGON_SERVER)
  target_include_directories(cd_server PRIVATE ${DROGON_INCLUDE_DIRS})
  target_link_libraries(cd_server PRIVATE ${DROGON_LIBRARIES} ${JSONCPP_LIBRARIES})
  message(STATUS "✓ Drogon and jsoncpp linked manually")
  message(STATUS "⏳ AudioStreamController: Requires Drogon WebSocket method signature fixes")
endif()

# Link whisper.cpp if enabled
if(ENABLE_WHISPER)
  # whisper.cpp 라이브러리 찾기 (third_party에 설치되어 있다고 가정)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp)
    add_subdirectory(third_party/whisper.cpp)
    target_link_libraries(cd_server PRIVATE whisper)
    target_include_directories(cd_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp)
  else()
    message(WARNING "whisper.cpp not found in third_party. Assuming system installation.")
    find_library(WHISPER_LIB whisper)
    if(WHISPER_LIB)
      target_link_libraries(cd_server PRIVATE ${WHISPER_LIB})
    else()
      message(WARNING "whisper library not found. Build may fail.")
    endif()
  endif()
  
  # CUDA support for whisper (optional)
  if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_TOOLKIT_ROOT_DIR}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_86")  # RTX 3090 compute capability
    target_compile_definitions(cd_server PRIVATE CD_ENABLE_CUDA)
  endif()
endif()

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
