# LLMOCR Desktop Application

완전한 독립 실행형 데스크톱 애플리케이션입니다!

## 특징

✅ **네이티브 GUI** - PyQt5 기반의 완전한 데스크톱 애플리케이션
✅ **브라우저 불필요** - 독립 실행형 앱으로 실행
✅ **크로스 플랫폼** - Windows, macOS, Linux 모두 지원
✅ **빠른 성능** - 네이티브 코드로 빠른 실행
✅ **직관적 UI** - 탭 기반의 사용하기 쉬운 인터페이스
✅ **백그라운드 처리** - 멀티스레딩으로 UI 블로킹 없음

## 설치 방법

### 1. Python 설치 확인

Python 3.8 이상이 필요합니다.

```bash
python3 --version
```

### 2. PyQt5 설치

```bash
pip3 install PyQt5
```

### 3. 기타 의존성 설치

```bash
pip3 install -r requirements.txt
```

## 실행 방법

### 방법 1: 실행 스크립트 사용 (권장)

#### macOS/Linux:
```bash
./run_gui.sh
```

#### Windows:
```cmd
run_gui.bat
```

실행 스크립트가 자동으로:
- Python 버전 확인
- PyQt5 설치 여부 확인
- 필요한 패키지 자동 설치 제안
- 애플리케이션 실행

### 방법 2: 직접 실행

```bash
python3 gui/desktop_app.py
```

## 화면 구성

애플리케이션은 8개의 탭으로 구성되어 있습니다:

### 1. 🗂️ Dataset Management
- **SynthDoG-ko 다운로드**
  - 출력 디렉토리 설정
  - 샘플 수 설정 (10 ~ 100,000)
  - 시작 인덱스 설정
  - 다운로드 버튼 클릭
  - 실시간 로그 출력

- **AI-Hub 데이터셋**
  - 다운로드 가이드 표시
  - 수동 다운로드 안내

### 2. 🔄 Data Processing
- **데이터 정제**
  - 소스/출력 디렉토리 설정
  - 텍스트 길이 필터 (최소/최대)
  - 이미지 크기 필터
  - 블러 임계값 설정
  - 이미지 복사/링크 옵션
  - 정제 실행 및 로그 확인

- **데이터 분할**
  - Train/Val/Test 비율 설정
  - 랜덤 시드 설정
  - 분할 실행

### 3. 🖼️ Single Image OCR
- **이미지 선택**
  - 파일 선택 대화상자
  - 이미지 미리보기

- **처리**
  - Ground Truth 입력 (선택사항)
  - 이미지 처리 실행

- **결과 표시**
  - 예측 텍스트
  - 신뢰도, 처리 시간, CER
  - 단어별 상세 정보

### 4-8. 기타 기능 (곧 구현 예정)
- Benchmark Execution
- Continuous Learning
- Error Analysis
- Audit Logs
- Batch Processing

## 사용 예시

### 데이터셋 다운로드 및 처리

1. **Dataset Management 탭**
   - SynthDoG-ko 다운로드 (1000 samples)
   - 로그 확인

2. **Data Processing 탭**
   - Clean Dataset 실행
   - Create Splits 실행

3. **처리 완료**
   - datasets/processed/ 폴더에 정제된 데이터 생성

### 단일 이미지 OCR

1. **Single Image OCR 탭**
2. "Select Image" 버튼 클릭
3. 이미지 파일 선택
4. Ground Truth 입력 (선택사항)
5. "Process Image" 버튼 클릭
6. 결과 확인

## 주요 기능

### 멀티스레딩
- 장시간 실행 작업은 백그라운드 스레드에서 실행
- UI가 블로킹되지 않음
- 실시간 진행 상황 표시

### 에러 처리
- 명확한 에러 메시지
- 로그 출력으로 디버깅 용이
- 사용자 친화적인 에러 대화상자

### 프로그레스 바
- 작업 진행 상황 실시간 표시
- 불확정 진행 표시 (다운로드 등)

### 스타일링
- 현대적인 UI 디자인
- 색상 코딩 (성공/에러)
- 반응형 레이아웃

## 아키텍처

### 메인 구조
```
MainWindow
├── Header (타이틀)
├── TabWidget
│   ├── DatasetManagementTab
│   ├── DataProcessingTab
│   ├── SingleImageOCRTab
│   └── ... (기타 탭들)
└── StatusBar
```

### 워커 스레드
```python
class WorkerThread(QThread):
    """백그라운드 작업 처리"""
    - 서브프로세스 실행
    - 진행 상황 시그널
    - 완료 시그널
```

### 탭 클래스
각 탭은 독립적인 QWidget:
```python
class DatasetManagementTab(QWidget):
    - UI 초기화
    - 이벤트 핸들러
    - 워커 스레드 관리
```

## 커스터마이징

### 스타일 변경

`MainWindow.init_ui()`의 `setStyleSheet()` 부분 수정:

```python
self.setStyleSheet("""
    QPushButton {
        background-color: #3498db;  /* 버튼 색상 */
        color: white;
    }
    /* ... */
""")
```

### 새 탭 추가

1. 새 클래스 생성:
```python
class MyNewTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        # UI 구성
        pass
```

2. MainWindow에 추가:
```python
self.tabs.addTab(MyNewTab(), "🆕 My Feature")
```

### 실제 모델 통합

`SingleImageOCRTab.process_image()` 메서드 수정:

```python
def process_image(self):
    # 실제 모델 로드 및 추론
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    if not hasattr(self, 'model'):
        self.model = VisionEncoderDecoderModel.from_pretrained("model_path")
        self.processor = TrOCRProcessor.from_pretrained("model_path")

    # 추론 실행
    pixel_values = self.processor(self.image, return_tensors="pt").pixel_values
    generated_ids = self.model.generate(pixel_values)
    prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 결과 표시
    self.prediction_text.setText(prediction)
```

## 패키징 (실행 파일 생성)

### PyInstaller로 실행 파일 만들기

```bash
# PyInstaller 설치
pip install pyinstaller

# 실행 파일 생성
pyinstaller --onefile --windowed --name="LLMOCR" gui/desktop_app.py
```

생성된 실행 파일: `dist/LLMOCR` 또는 `dist/LLMOCR.exe`

### macOS 앱 번들 생성

```bash
# py2app 설치
pip install py2app

# setup.py 생성 및 빌드
python setup.py py2app
```

### Windows 인스톨러 생성

```bash
# Inno Setup 사용
# 또는 NSIS 사용
```

## 트러블슈팅

### PyQt5 설치 실패 (macOS)

```bash
# Homebrew로 설치
brew install pyqt5

# 또는
pip3 install --upgrade pip
pip3 install PyQt5
```

### PyQt5 설치 실패 (Linux)

```bash
# Ubuntu/Debian
sudo apt-get install python3-pyqt5

# Fedora
sudo dnf install python3-qt5
```

### "No module named PyQt5" 에러

```bash
# 올바른 Python 인터프리터 확인
which python3
python3 -m pip install PyQt5
```

### 애플리케이션이 실행되지 않음

```bash
# 직접 실행하여 에러 확인
python3 gui/desktop_app.py

# 의존성 확인
python3 -c "import PyQt5; print(PyQt5.__version__)"
```

### 고해상도 디스플레이 문제

환경 변수 설정:
```bash
export QT_AUTO_SCREEN_SCALE_FACTOR=1
python3 gui/desktop_app.py
```

## 성능 최적화

### 모델 캐싱
```python
# 세션 상태에 모델 캐시
if not hasattr(self, 'cached_model'):
    self.cached_model = load_model()
```

### 이미지 리사이징
```python
# 큰 이미지는 미리보기용으로 리사이즈
pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
```

### 멀티스레딩
```python
# 장시간 작업은 항상 WorkerThread 사용
worker = WorkerThread(command, description)
worker.finished.connect(callback)
worker.start()
```

## 기능 비교

### Desktop App vs Streamlit

| 기능 | Desktop App | Streamlit Web |
|------|-------------|---------------|
| 브라우저 필요 | ❌ 불필요 | ✅ 필요 |
| 설치 | PyQt5만 | Streamlit + 브라우저 |
| 성능 | ⚡ 빠름 | 보통 |
| UI 반응성 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 배포 | 실행 파일 | 서버 필요 |
| 오프라인 사용 | ✅ 가능 | ❌ 불가능 |
| 멀티태스킹 | ✅ 완벽 지원 | 제한적 |

## 향후 계획

- [ ] 모든 탭 완전 구현
- [ ] 실시간 모델 추론 통합
- [ ] 설정 저장/로드
- [ ] 다크 모드 지원
- [ ] 플러그인 시스템
- [ ] 자동 업데이트
- [ ] 다국어 지원

## 라이센스

프로젝트 라이센스를 따릅니다.

## 기여

Issue 및 Pull Request 환영합니다!

---

**Made with PyQt5 ❤️**
