Act as a Senior Backend & AI Systems Engineer. Develop the backend for "ClassDefense Web".

### 1. Server Environment
- Hardware: Linux Server with RTX 3090 GPU (24GB VRAM) and 40-core vCPU.
- Tech Stack: FastAPI (Python) or Drogon (C++), gRPC/WebSockets, whisper.cpp, llama.cpp, RVC.

### 2. Core Backend Requirements
1. [Real-time Audio Pipeline]:
   - Implement a WebSocket endpoint to receive binary PCM audio chunks.
   - Use a thread-safe circular buffer to store incoming audio for the STT engine.
   - Integrate 'whisper.cpp' as a sub-process or library to perform inference on the GPU.

2. [Intelligence & Trigger Logic]:
   - Continuously pipe STT results into a Llama 3 8B model.
   - Use regex-based and LLM-based detection for:
     - User's Name (Trigger: Attendance Response)
     - Interrogative sentences (Trigger: Stalling Audio & Answer Generation)
   - Implement a 'Context Buffer' that stores the last 5-10 minutes of text for RAG (Retrieval-Augmented Generation).

3. [Autonomous Attendance & Command System]:
   - Voice Attendance: On name detection, send a signal to the client to play local audio or stream a pre-synthesized RVC voice packet.
   - Electronic Attendance: Implement a headless automation bridge. Upon detection of "attendance check" keywords, execute a script to log into 'KLMS' or a designated portal and perform the check-in.

4. [Session Management & Secretary]:
   - Handle 3-mode states: 'Note' (Idle), 'Lecture' (Transcription), 'Active FSD' (Defense & Vision status).
   - On 'Summary' request, process the entire session log via Llama 3 to output a structured Markdown summary.

### 3. Technical Constraints
- High Concurrency: Leverage 40-core CPU using asynchronous I/O and thread pools.
- Low Latency: Ensure the path from "Name Detection" to "Response Signal" is under 500ms.
- Memory Safety: Manage GPU VRAM carefully to prevent OOM (Out of Memory) when multiple models are loaded.

### 4. Task
Generate the 'ServerController' class that coordinates the Audio-STT-LLM pipeline and the 'AttendanceService' for automated check-ins. Provide the API/WebSocket schema for client communication.