# 🎓 FSD - Full Self Defense

> 수업 중 졸아도, 자리를 비워도 걱정 NO!  
> AI가 대신 수업을 듣고, 출석도 대답해주는 출석도 대답해주는 당신의 학점 지키미!

---

## 📖 프로젝트 소개

> "교수님: 이상범?" → AI(이상범): "**네!**"

**FSD (Full Self Defense)** 는 강의 중 학생의 상태(졸음, 자리비움)를 실시간으로 감지하고, 호명 출석에 자동으로 대답하며, 수업 내용을 STT로 기록하고 AI로 요약해주는 **올인원 스마트 학습 보조 시스템**입니다.

### 🎯 기획 의도

누구나 한 번쯤은 경험해본 상황들:
- 😴 수업 시간에 졸다가 출석 호명을 놓친 적
- 🚽 잠깐 화장실 다녀온 사이 출석 체크가 끝난 적
- 📝 필기하느라 정작 수업 내용을 놓친 적

**FSD**는 이 모든 고민을 해결합니다. AI가 당신의 눈과 귀가 되어 수업을 대신 들어줍니다.

---

## ✨ 주요 기능

| 기능 | 설명 |
| --- | --- |
| **😴 졸음 감지** | MediaPipe Face Landmarker로 눈 감김(EAR)을 실시간 분석, 2초 이상 졸면 FSD 모드 자동 활성화 |
| **🚶 자리비움 감지** | 얼굴이 5초 이상 감지되지 않으면 자동으로 FSD 모드 전환 |
| **🎤 실시간 STT** | Whisper 모델을 활용한 고품질 한국어 음성 인식, 강의 내용 실시간 기록 |
| **📢 자동 출석 대답** | "이상범" 호명 감지 시 자동으로 "네!" 음성 재생 (TTS) |
| **📄 AI 요약** | LLM(Qwen2.5)을 활용한 강의 내용 자동 요약 및 핵심 정리 |
| **💡 질문 추천** | "질문 있나요?" 감지 시 수업 내용 기반 추천 질문 자동 생성 |
| **📊 신호등 UI** | 현재 상태를 직관적으로 표시 (🟢수업진행중 / 🟡출석대기중 / 🔴출석대답중) |

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            💻 Web Client (React + Vite)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  🎥 MediaPipe Face Detection    🎤 Web Audio API    📱 Real-time UI         │
│  (졸음/자리비움 감지)              (음성 캡처)         (신호등 상태)           │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ WebSocket (Real-time Audio)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ⚡ C++ Backend (Drogon Framework)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  📡 WebSocket Handler  →  🔊 VAD Segmenter  →  🗣️ Whisper STT              │
│                                                      │                       │
│                              ┌───────────────────────┼───────────────────┐   │
│                              ▼                       ▼                   ▼   │
│                    🤖 Qwen2.5 LLM           📢 GPT-SoVITS TTS      💡 Q&A   │
│                    (요약/질문생성)            (음성 합성)           (추천)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 💡 Core Workflow

### 1. 졸음/자리비움 감지 (Frontend)

- **MediaPipe Face Landmarker**: 웹캠에서 얼굴 랜드마크 468개 실시간 추출
- **EAR (Eye Aspect Ratio)**: 눈 깜빡임 비율 계산으로 졸음 감지
- **Calibration**: 최초 2초간 개인별 눈 크기 보정

### 2. 실시간 음성 처리 (Backend)

```
🎤 Audio Stream → VAD Segmenter → Whisper STT → 키워드 감지
                                                    │
                      ┌─────────────────────────────┼─────────────────────────┐
                      ▼                             ▼                         ▼
                 "출석" 감지                   "이상범" 감지              "질문있나요"
                      │                             │                         │
                      ▼                             ▼                         ▼
                출석 모드 ON               GPT-SoVITS TTS 재생          LLM 질문 생성
```

### 3. 자동 출석 응답 흐름

| 단계 | 신호등 | 동작 |
| --- | --- | --- |
| 1. "출석 체크" 감지 | 🟢 → 🟡 | 출석 대기 모드 진입 |
| 2. "이상범" 호명 | 🟡 → 🔴 | 출석 대답 오버레이 표시 |
| 3. TTS "네!" 재생 | 🔴 | 자동 음성 출력 |
| 4. 완료 | 🔴 → 🟢 | 수업 진행 모드 복귀 |

---

## 📱 User Interface

### 신호등 상태 표시

| 상태 | 색상 | 의미 |
| --- | --- | --- |
| 수업 진행중 | 🟢 | 정상 수업 모드 |
| 출석 대기중 | 🟡 | "출석" 감지됨, 호명 대기 |
| 출석 대답중 | 🔴 | 이름 호명됨, TTS 재생 중 |

### 강의 내용 패널

- **회색 박스**: 과거 STT 내용
- **파란색 박스**: 현재 STT 내용  
- **빨간색 박스**: AI 추천 질문

### 경고 오버레이

- 😴 **졸음 감지**: "졸음 감지! 곧 FSD가 자동 실행됩니다."
- 🚶 **자리비움 감지**: "자리 비움 감지!"
- 🎤 **호명 출석 감지**: "호명 출석 감지! 곧 자동으로 대답합니다."

---

## 🛠️ Tech Stack

| Category | Technology |
| --- | --- |
| **Frontend** | React 18, Vite, Zustand |
| **Styling** | Vanilla CSS, Glass Morphism |
| **Face Detection** | MediaPipe Face Landmarker |
| **Audio Processing** | Web Audio API, AudioWorklet |
| **Backend** | C++17, Drogon Framework |
| **STT** | Whisper (whisper.cpp) |
| **LLM** | Qwen2.5-14B (llama.cpp) |
| **TTS** | GPT-SoVITS |
| **Communication** | WebSocket (Real-time) |

---

## 🚀 Getting Started

### Prerequisites

- Node.js 18+
- CMake 3.16+
- CUDA Toolkit (GPU acceleration)
- Python 3.10+ (for GPT-SoVITS)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/FSD.git
cd FSD

# Build backend
cd FSD
mkdir build && cd build
cmake ..
make -j4

# Install frontend dependencies
cd ../FSD-UI
npm install

# Start all services
./run_all.sh
```

### Environment Variables

```bash
export CD_WHISPER_BASE="http://127.0.0.1:8080"
export CD_LLM_BASE="http://127.0.0.1:8000"
export CD_TTS_BASE="http://127.0.0.1:9880"
export CD_TTS_REF_AUDIO="/path/to/reference.wav"
export CD_TTS_PROMPT_TEXT="참조 음성 텍스트"
```

---

## 👥 Team

<table>
  <tr>
    <td align="center">
      <b>이상범 (Minseok Yoon)</b><br/>
      <sub>KAIST CS 21</sub><br/>
      <sub>KAIST 전산학부</sub>
    </td>
    <td align="center">
      <b>박정우 (Jungwoo Park)</b><br/>
      <sub>HYU CSE 21</sub><br/>
      <sub>한양대학교 컴퓨터소프트웨어학부</sub>
    </td>
  </tr>
</table>
