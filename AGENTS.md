Act as an expert C++ Software Engineer and AI Architect. I am building a real-time lecture assistance system named "ClassDefense".

### 1. Project Overview & Environment
- Goal: Real-time lecture transcription, automated response for professor's questions, sleep detection, and lecture summarization.
- Target Platforms: Client (macOS & Windows), Server (Linux/WSL2 with RTX 3090 GPU).
- Tech Stack: 
  - Language: C++20
  - Build System: CMake
  - Audio: PortAudio (Cross-platform audio I/O)
  - Vision: OpenCV & MediaPipe (Sleep detection via EAR - Eye Aspect Ratio)
  - UI: Dear ImGui (Transparent overlay for transcript)
  - STT: whisper.cpp (Running on Server GPU)
  - LLM: llama.cpp (Llama 3 8B, 8-bit quantized on Server GPU)
  - Communication: gRPC (Low-latency streaming between Client and Server)

### 2. Core Features & Logic Flow
1. Ears (Client): Captures 16kHz Mono audio via PortAudio and streams to Server via gRPC.
2. Brain (Server): 
   - Converts audio to text using whisper.cpp.
   - Monitors text for triggers: "Name (User)", "Do you know...", "Any questions?".
   - If triggered:
     - Immediate Action: Send 'STALL_SIGNAL' to Client.
     - Secondary Action: Generate response using llama.cpp based on the last 5 minutes of lecture transcript.
3. Guardian (Client): 
   - Uses OpenCV/MediaPipe to monitor if the user is sleeping or not responding.
   - If 'STALL_SIGNAL' is received: Play pre-recorded stalling audio (e.g., "Umm... let me think...").
   - If user is sleeping or silent for 3 seconds after a question: Play the AI-generated answer using pre-synthesized voice (RVC).
4. Secretary (Server): At the end of the session, summarize all transcripts into a "Exam Prep Study Note".

### 3. Task: Generate Initial Project Structure
Please generate the following:

#### A. Modular Folder Structure
Suggest a clean directory tree (e.g., /src/client, /src/server, /proto, /include, /third_party).

#### B. Comprehensive CMakeLists.txt
- Must support macOS (Apple Silicon/Intel) and Windows.
- Find and link: PortAudio, OpenCV, gRPC, Protobuf, and Dear ImGui.
- Setup CPack for generating .dmg (Mac) and .exe (NSIS/Windows) installers.

#### C. Core Client Logic (main.cpp & AudioCapturer.cpp)
- Initialize a transparent Dear ImGui overlay for the transcript.
- Implement a thread-safe PortAudio capture loop (16kHz, Float32).
- Setup the gRPC client stub for streaming audio to the server.

#### D. Core Server Logic (stt_server.cpp & TriggerEngine.cpp)
- Implement gRPC service to receive audio chunks.
- Setup a placeholder for whisper.cpp and llama.cpp integration.
- Implement the 'TriggerEngine' that uses regex to detect questions and manage the 'Stalling' state.

#### E. Vision & Response Handler
- Provide C++ code for EAR calculation to detect sleep.
- Implement the 'ResponseHandler' that manages audio playback (Stalling sound vs. AI Answer).

### 4. Special Instructions
- Use Modern C++ features (Smart pointers, std::thread, std::mutex).
- Ensure all file paths are handled relatively for cross-platform compatibility.
- For macOS, include necessary 'Info.plist' permission strings for Microphone and Camera.
- Focus on minimizing latency between 'Professor Question' and 'Stalling Audio Playback'.