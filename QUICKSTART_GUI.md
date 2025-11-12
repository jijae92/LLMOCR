# LLMOCR GUI 빠른 시작 가이드

LLMOCR은 **두 가지 GUI** 옵션을 제공합니다!

## 🖥️ Desktop App (데스크톱 앱) - 추천!

### 특징
✅ **네이티브 애플리케이션** - 브라우저 없이 실행
✅ **빠른 성능** - PyQt5 기반
✅ **오프라인 사용 가능**
✅ **직관적인 인터페이스**

### 실행 방법

#### macOS/Linux:
```bash
# 1. 실행 스크립트에 권한 부여 (최초 1회만)
chmod +x run_gui.sh

# 2. 실행
./run_gui.sh
```

#### Windows:
```cmd
run_gui.bat
```

실행 스크립트가 자동으로:
- Python 버전 확인
- PyQt5 설치 여부 확인
- 필요한 패키지 자동 설치
- 앱 실행

### 수동 실행

```bash
# PyQt5 설치
pip3 install PyQt5

# 앱 실행
python3 gui/desktop_app.py
```

---

## 🌐 Streamlit Web App

### 특징
✅ **모든 기능 완전 구현**
✅ **웹 브라우저에서 실행**
✅ **실시간 결과 시각화**
✅ **풍부한 차트 및 그래프**

### 실행 방법

```bash
# Streamlit 설치
pip3 install streamlit

# 웹 앱 실행
streamlit run gui/streamlit_app.py
```

자동으로 브라우저가 열리며 `http://localhost:8501`에서 접속됩니다.

---

## 📊 기능 비교

| 기능 | Desktop App | Streamlit Web |
|------|-------------|---------------|
| **설치** | PyQt5만 필요 | Streamlit + 브라우저 |
| **실행 방식** | 독립 실행형 앱 | 웹 서버 |
| **인터넷 연결** | 불필요 | 불필요 (로컬) |
| **성능** | ⚡ 매우 빠름 | 빠름 |
| **UI 반응성** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **멀티태스킹** | ✅ 완벽 지원 | 제한적 |
| **오프라인** | ✅ 완전 지원 | ✅ 지원 |
| **배포** | 실행 파일 생성 가능 | 서버 배포 |

### 구현 상태

| 탭 | Desktop App | Streamlit |
|----|-------------|-----------|
| Dataset Management | ✅ 구현 | ✅ 구현 |
| Data Processing | ✅ 구현 | ✅ 구현 |
| Single Image OCR | ✅ 구현 | ✅ 구현 |
| Benchmark Execution | 🔜 곧 구현 | ✅ 구현 |
| Continuous Learning | 🔜 곧 구현 | ✅ 구현 |
| Error Analysis | 🔜 곧 구현 | ✅ 구현 |
| Audit Logs | 🔜 곧 구현 | ✅ 구현 |
| Batch Processing | 🔜 곧 구현 | ✅ 구현 |

---

## 🎯 어떤 것을 선택해야 하나요?

### Desktop App을 선택하세요:
- ✅ 빠른 성능이 필요한 경우
- ✅ 독립 실행형 앱을 원하는 경우
- ✅ 브라우저를 사용하고 싶지 않은 경우
- ✅ 기본 기능만 필요한 경우
- ✅ 실행 파일로 배포하고 싶은 경우

### Streamlit Web App을 선택하세요:
- ✅ 모든 기능을 사용하고 싶은 경우
- ✅ 풍부한 시각화가 필요한 경우
- ✅ 여러 차트와 테이블이 필요한 경우
- ✅ 웹 기반 인터페이스를 선호하는 경우
- ✅ 원격 서버에 배포하고 싶은 경우

---

## 💡 빠른 팁

### Desktop App 실행이 안 되나요?

```bash
# PyQt5 재설치
pip3 uninstall PyQt5
pip3 install PyQt5

# 직접 실행
python3 gui/desktop_app.py
```

### Streamlit 실행이 안 되나요?

```bash
# Streamlit 재설치
pip3 uninstall streamlit
pip3 install streamlit

# 포트 변경
streamlit run gui/streamlit_app.py --server.port 8502
```

### 의존성 문제가 있나요?

```bash
# 모든 의존성 재설치
pip3 install -r requirements.txt
```

---

## 📚 상세 문서

- **Desktop App**: `gui/DESKTOP_README.md`
- **Streamlit Web**: `gui/README_v2.md`
- **프로젝트 전체**: `README.md`

---

## 🚀 첫 실행 추천

1. **Desktop App으로 시작** (빠르고 간단)
   ```bash
   ./run_gui.sh
   ```

2. **Dataset Management** 탭에서 작은 데이터셋 다운로드
   - SynthDoG-ko: 100 samples

3. **Data Processing** 탭에서 데이터 정제

4. **Single Image OCR** 탭에서 이미지 테스트

5. 더 많은 기능이 필요하면 **Streamlit**으로 전환:
   ```bash
   streamlit run gui/streamlit_app.py
   ```

---

**즐거운 OCR 작업 되세요! 🎉**
