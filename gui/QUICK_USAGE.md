# ⚡ LLMOCR Desktop App - 빠른 사용법

5분 안에 시작하기!

## 🚀 1단계: 앱 실행

```bash
./run_gui.sh
```

앱이 열리면 다크모드 UI를 볼 수 있습니다! 🌙

---

## 📊 2단계: 데이터셋 다운로드 (선택)

### 🗂️ Dataset Management 탭

#### 빠른 테스트용:
```
Sample Limit: 100
→ [📥 Download SynthDoG-ko] 클릭
→ 약 1분 대기
```

#### 실제 사용용:
```
Sample Limit: 1000
→ [📥 Download SynthDoG-ko] 클릭
→ 약 5분 대기
```

**생성 위치**: `datasets/raw/synthdog_ko/`

---

## 🔄 3단계: 데이터 정제 (선택)

### Data Processing 탭

#### 1. Clean Dataset
```
Source: datasets/raw/synthdog_ko
Output: datasets/processed/synthdog_ko_clean
→ [🧹 Clean Dataset] 클릭
→ 완료 대기
```

#### 2. Create Splits
```
Train: 0.8
Val: 0.1
Test: 0.1
→ [✂️ Create Splits] 클릭
```

---

## 🖼️ 4단계: 이미지 OCR 테스트

### Single Image OCR 탭

1. **[📁 Select Image]** 클릭
2. 이미지 파일 선택
3. **[🚀 Process Image]** 클릭
4. 결과 확인:
   - 예측 텍스트
   - 신뢰도
   - 처리 시간

---

## 🎨 다크모드 특징

### 색상
- 🖤 **배경**: 진한 검정 (#1e1e1e)
- 💎 **강조**: 밝은 청록색 (#61dafb)
- 🔵 **버튼**: 청록색 (#0d7377)
- ⚪ **텍스트**: 밝은 회색 (#e0e0e0)

### 효과
- ✨ 호버 시 밝아지는 버튼
- 🎯 포커스 시 시안 테두리
- 📊 그라데이션 프로그레스 바
- 🔄 부드러운 스크롤바

---

## 💡 핵심 기능

### 이미 구현됨 ✅
- 🗂️ **Dataset Management**: SynthDoG-ko 다운로드
- 🔄 **Data Processing**: 정제 & 분할
- 🖼️ **Single Image OCR**: 이미지 처리

### 곧 구현 예정 🔜
- 🚀 Benchmark Execution
- 🔁 Continuous Learning
- 📊 Error Analysis
- 📋 Audit Logs
- ⚡ Batch Processing

---

## 🎯 추천 워크플로우

### 초보자 🔰
```
1. Single Image OCR 탭
2. 테스트 이미지 업로드
3. 결과 확인
```

### 개발자 👨‍💻
```
1. Dataset Management → 100 samples 다운로드
2. Data Processing → 정제 및 분할
3. Single Image → 샘플 테스트
```

### 연구자 🎓
```
1. Dataset Management → 10,000 samples 다운로드
2. Data Processing → 전체 정제
3. Streamlit 버전으로 전환 (고급 기능)
```

---

## ⌨️ 단축키

| 키 | 기능 |
|----|------|
| `Ctrl+Q` / `Cmd+Q` | 앱 종료 |
| `Ctrl+Tab` | 다음 탭 |
| `Ctrl+Shift+Tab` | 이전 탭 |

---

## 🆘 문제 해결

### 앱이 안 열려요
```bash
# PyQt5 재설치
pip install --upgrade PyQt5

# 직접 실행
python gui/desktop_app.py
```

### 다운로드가 안 돼요
```bash
# 의존성 확인
pip install datasets huggingface-hub

# 네트워크 확인
ping huggingface.co
```

### 버튼을 눌러도 반응이 없어요
- 로그 출력 확인
- 백그라운드 작업 진행 중일 수 있음
- 프로그레스 바 확인

---

## 📚 더 알아보기

- **상세 가이드**: `DATASET_MANAGEMENT_GUIDE.md`
- **전체 문서**: `DESKTOP_README.md`
- **비교**: `GUI_COMPARISON.md`

---

## 🎉 시작하기

```bash
# 1. 앱 실행
./run_gui.sh

# 2. Dataset Management 탭
# 3. 100 samples 다운로드
# 4. Single Image OCR 테스트

# 끝! 🎊
```

**즐거운 OCR 작업 되세요!** ✨
