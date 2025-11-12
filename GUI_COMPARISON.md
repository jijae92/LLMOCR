# LLMOCR GUI 비교 가이드

LLMOCR은 두 가지 강력한 GUI를 제공합니다. 당신의 필요에 맞는 것을 선택하세요!

## 📱 옵션 1: Desktop App (PyQt5)

![Desktop App](https://img.shields.io/badge/Platform-Desktop-blue)
![Status](https://img.shields.io/badge/Status-Stable-green)

### 개요
네이티브 데스크톱 애플리케이션으로, 독립 실행형 앱처럼 작동합니다.

### 장점
✅ **독립 실행형** - 브라우저 불필요
✅ **빠른 성능** - 네이티브 코드
✅ **오프라인** - 인터넷 연결 불필요
✅ **멀티태스킹** - 백그라운드 처리 완벽 지원
✅ **실행 파일 생성 가능** - PyInstaller로 배포 가능
✅ **리소스 효율적** - 낮은 메모리 사용

### 단점
❌ 일부 고급 기능 아직 미구현 (진행 중)
❌ PyQt5 추가 설치 필요

### 실행 방법
```bash
# 간단!
./run_gui.sh           # macOS/Linux
run_gui.bat            # Windows
```

### 스크린샷
```
┌─────────────────────────────────────────┐
│  📝 LLMOCR - Korean OCR Platform        │
├─────────────────────────────────────────┤
│ 🗂️ Dataset | 🔄 Processing | 🖼️ OCR ... │
├─────────────────────────────────────────┤
│                                         │
│  [Dataset Management Interface]         │
│                                         │
│  ┌────────────────────────────────┐    │
│  │ Download SynthDoG-ko           │    │
│  │ Output: datasets/raw/          │    │
│  │ Samples: [1000]                │    │
│  │ [📥 Download]                  │    │
│  └────────────────────────────────┘    │
│                                         │
│  Output Log:                            │
│  ┌────────────────────────────────┐    │
│  │ [10:30:15] Starting download... │    │
│  │ ✓ Downloaded successfully!      │    │
│  └────────────────────────────────┘    │
│                                         │
├─────────────────────────────────────────┤
│ Ready                                   │
└─────────────────────────────────────────┘
```

### 기술 스택
- **프레임워크**: PyQt5
- **언어**: Python 3.8+
- **멀티스레딩**: QThread
- **패키징**: PyInstaller, py2app

---

## 🌐 옵션 2: Streamlit Web App

![Platform](https://img.shields.io/badge/Platform-Web-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

### 개요
웹 브라우저에서 실행되는 풀 기능 애플리케이션.

### 장점
✅ **완전한 기능** - 모든 기능 구현 완료
✅ **풍부한 시각화** - 차트, 그래프, 테이블
✅ **쉬운 사용** - 직관적인 웹 UI
✅ **반응형** - 다양한 화면 크기 지원
✅ **빠른 개발** - Streamlit의 빠른 프로토타이핑
✅ **원격 접속 가능** - 서버 배포 시

### 단점
❌ 브라우저 필요
❌ Streamlit 런타임 오버헤드
❌ 상태 관리 제한

### 실행 방법
```bash
streamlit run gui/streamlit_app.py
```

### 스크린샷
```
┌──────────────────────────────────────────────┐
│ 📝 LLMOCR - Complete Korean OCR Platform     │
│ Comprehensive data management, benchmarking  │
├──────────────────────────────────────────────┤
│ 🗂️ Dataset Management                        │
│ 🔄 Data Processing                           │
│ 🚀 Benchmark Execution                       │
│ 🔁 Continuous Learning                       │
│ 🖼️ Single Image OCR                          │
│ 📊 Error Analysis                            │
│ 📋 Audit Logs                                │
│ ⚡ Batch Processing                          │
├──────────────────────────────────────────────┤
│                                              │
│  Download SynthDoG-ko Dataset                │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━        │
│                                              │
│  Output Directory: [datasets/raw/synthdog_ko]│
│  Sample Limit: [▓▓▓▓▓▓▓░░░] 1000            │
│  Start Index: [0]                            │
│                                              │
│  [📥 Download SynthDoG-ko]                   │
│                                              │
│  ✓ Downloaded 1000 samples successfully!    │
│                                              │
│  📄 Download Log                             │
│  ┌──────────────────────────────────────┐   │
│  │ Downloading SynthDoG-ko dataset...   │   │
│  │ Processing samples: 100%              │   │
│  │ ✓ Successfully downloaded 1000 samples│   │
│  └──────────────────────────────────────┘   │
│                                              │
└──────────────────────────────────────────────┘
```

### 기술 스택
- **프레임워크**: Streamlit
- **언어**: Python 3.8+
- **시각화**: Plotly, Pandas
- **배포**: Streamlit Cloud, Docker

---

## 📊 상세 비교표

### 일반 특성

| 항목 | Desktop App | Streamlit Web |
|------|-------------|---------------|
| **플랫폼** | Windows, macOS, Linux | 웹 브라우저 (모든 OS) |
| **설치** | PyQt5 | Streamlit |
| **실행** | 앱 실행 | 서버 시작 |
| **브라우저** | 불필요 | 필수 |
| **오프라인** | ✅ 완전 지원 | ✅ 로컬에서 지원 |
| **배포** | 실행 파일 (.exe, .app) | 웹 서버 / Docker |
| **업데이트** | 재배포 필요 | 코드 업데이트만 |

### 성능

| 항목 | Desktop App | Streamlit Web |
|------|-------------|---------------|
| **시작 속도** | ⚡⚡⚡⚡⚡ (매우 빠름) | ⚡⚡⚡ (보통) |
| **응답 속도** | ⚡⚡⚡⚡⚡ (즉각) | ⚡⚡⚡⚡ (빠름) |
| **메모리 사용** | 💾💾 (낮음) | 💾💾💾 (보통) |
| **CPU 사용** | ⚙️⚙️ (낮음) | ⚙️⚙️⚙️ (보통) |
| **멀티태스킹** | ✅ 완벽 지원 | ⚠️ 제한적 |

### 사용자 경험

| 항목 | Desktop App | Streamlit Web |
|------|-------------|---------------|
| **학습 곡선** | ⭐⭐⭐ (보통) | ⭐⭐ (쉬움) |
| **직관성** | ⭐⭐⭐⭐ (좋음) | ⭐⭐⭐⭐⭐ (매우 좋음) |
| **반응성** | ⭐⭐⭐⭐⭐ (즉각) | ⭐⭐⭐⭐ (빠름) |
| **시각화** | ⭐⭐⭐ (기본) | ⭐⭐⭐⭐⭐ (풍부) |
| **커스터마이징** | ⭐⭐⭐⭐⭐ (완전) | ⭐⭐⭐ (제한적) |

### 기능 구현

| 기능 | Desktop App | Streamlit Web |
|------|-------------|---------------|
| Dataset Management | ✅ 완료 | ✅ 완료 |
| Data Processing | ✅ 완료 | ✅ 완료 |
| Single Image OCR | ✅ 완료 | ✅ 완료 |
| Benchmark Execution | 🔜 진행 중 | ✅ 완료 |
| Continuous Learning | 🔜 진행 중 | ✅ 완료 |
| Error Analysis | 🔜 진행 중 | ✅ 완료 |
| Audit Logs | 🔜 진행 중 | ✅ 완료 |
| Batch Processing | 🔜 진행 중 | ✅ 완료 |

---

## 🎯 사용 사례별 추천

### 개인 사용자
**Desktop App** 추천
- 빠른 로컬 작업
- 간단한 데이터셋 처리
- 오프라인 환경

### 연구자/개발자
**Streamlit Web** 추천
- 풍부한 시각화 필요
- 벤치마크 및 분석
- 실험 및 프로토타이핑

### 엔터프라이즈
**둘 다 사용**
- Desktop App: 일상적인 OCR 작업
- Streamlit: 분석 및 리포팅

### 데모/프레젠테이션
**Streamlit Web** 추천
- 아름다운 UI
- 실시간 시각화
- 쉬운 접근성

---

## 🚀 시작하기

### Desktop App

```bash
# 1. 저장소 클론
git clone <repository-url>
cd LLMOCR

# 2. PyQt5 설치
pip3 install PyQt5

# 3. 실행
./run_gui.sh          # macOS/Linux
run_gui.bat           # Windows
```

### Streamlit Web

```bash
# 1. 저장소 클론 (위와 동일)

# 2. Streamlit 설치
pip3 install streamlit

# 3. 실행
streamlit run gui/streamlit_app.py
```

---

## 💡 Pro Tips

### Desktop App
- `Ctrl+Q` 또는 `Cmd+Q`로 종료
- 로그는 실시간으로 업데이트됨
- 백그라운드 작업 중에도 다른 탭 사용 가능

### Streamlit Web
- `R` 키로 페이지 새로고침
- `C` 키로 캐시 클리어
- 사이드바는 `>` 버튼으로 숨기기/표시

---

## 🔮 로드맵

### Desktop App
- [ ] 모든 탭 완전 구현
- [ ] 다크 모드
- [ ] 설정 저장/로드
- [ ] 플러그인 시스템

### Streamlit Web
- [x] 모든 기능 완료
- [ ] 실시간 모델 추론
- [ ] 고급 시각화 추가
- [ ] 멀티유저 지원

---

## 📞 지원

문제가 있나요?

1. **Documentation**: `gui/DESKTOP_README.md` 또는 `gui/README_v2.md`
2. **Quick Start**: `QUICKSTART_GUI.md`
3. **Issues**: GitHub Issues에 문의

---

## 🏆 권장 사항

**처음 사용자**: Desktop App으로 시작 → 필요시 Streamlit으로 전환

**이유**:
1. Desktop App이 더 빠르게 시작
2. 기본 기능 충분
3. 고급 기능 필요시 Streamlit 사용

---

**양쪽 다 사용해보고 자신에게 맞는 것을 선택하세요!** 🎉
