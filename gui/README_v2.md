# LLMOCR Complete GUI v2.0 - 사용 가이드

완전히 새로워진 LLMOCR GUI입니다. 모든 프로젝트 기능이 통합되었습니다!

## 새로운 기능 (v2.0)

### 🆕 추가된 기능

1. **🗂️ Dataset Management (데이터셋 관리)**
   - SynthDoG-ko 자동 다운로드
   - AI-Hub 데이터셋 다운로드 가이드
   - 다운로드 이력 추적

2. **🔄 Data Processing (데이터 처리)**
   - 데이터 정제 (이미지 품질 검사, 텍스트 필터링)
   - Train/Val/Test 데이터 분할
   - 난이도 분석 및 통계

3. **🚀 Benchmark Execution (벤치마크 실행)**
   - 다중 모델 벤치마크 실행
   - CER, WER, 처리량, 지연시간 측정
   - 결과 자동 저장 (JSON, CSV, Markdown)

4. **🔁 Continuous Learning (지속적 학습)**
   - 자동화된 학습 파이프라인
   - 회귀 검사 (Regression Testing)
   - 모델 자동 승격 (Auto-promotion)

### ✅ 기존 기능 (v1.0에서 유지)

5. **🖼️ Single Image OCR**
   - 단일 이미지 처리
   - Bounding box 시각화
   - High DPI 재시도

6. **📊 Error Analysis**
   - 에러 패턴 분석
   - Top N 에러 샘플 표시

7. **📋 Audit Logs**
   - 모든 작업 로그 추적
   - 통계 및 리포트 생성

8. **⚡ Batch Processing**
   - 다중 이미지 배치 처리
   - CSV 결과 다운로드

## 설치 방법

### 1. 필수 패키지 설치

```bash
# 기본 패키지
pip install -r requirements.txt

# GUI 실행에 필요한 추가 패키지 확인
pip install streamlit pandas plotly
```

### 2. 디렉토리 구조 확인

프로젝트 루트에서 실행해야 합니다:

```
LLMOCR/
├── gui/
│   └── streamlit_app.py  <- 이 파일 실행
├── datasets/
├── benchmarks/
├── tools/
└── requirements.txt
```

## 실행 방법

### 기본 실행

```bash
# 프로젝트 루트에서
streamlit run gui/streamlit_app.py
```

브라우저가 자동으로 열리며 `http://localhost:8501`에서 접속됩니다.

### 포트 변경

```bash
streamlit run gui/streamlit_app.py --server.port 8502
```

## 사용 가이드

### 1. 데이터셋 다운로드 (Dataset Management)

**SynthDoG-ko 다운로드:**
1. "Dataset Management" 탭으로 이동
2. 샘플 수 설정 (예: 1000개)
3. "Download SynthDoG-ko" 버튼 클릭
4. 다운로드 완료 후 로그 확인

**AI-Hub 데이터셋:**
1. "Show Download Instructions" 버튼으로 가이드 확인
2. 웹사이트에서 수동 다운로드
3. 지정된 디렉토리에 배치

### 2. 데이터 처리 (Data Processing)

**Step 1: 데이터 정제**
1. "Data Processing" 탭으로 이동
2. Source Directory 설정 (예: `datasets/raw/synthdog_ko`)
3. Output Directory 설정 (예: `datasets/processed/synthdog_ko_clean`)
4. 필터링 옵션 설정:
   - Min Text Length: 5
   - Max Text Length: 1000
   - Min Image Dimension: 32
   - Blur Threshold: 100.0
5. "Clean Dataset" 버튼 클릭
6. 통계 확인 (Total Samples, Valid Samples, Success Rate)

**Step 2: 데이터 분할**
1. Input Directory 설정 (정제된 데이터)
2. Split 비율 설정:
   - Train: 80%
   - Validation: 10%
   - Test: 10%
3. "Create Splits" 버튼 클릭
4. 각 split의 통계 확인

### 3. 벤치마크 실행 (Benchmark Execution)

1. "Benchmark Execution" 탭으로 이동
2. 설정:
   - Model Paths: `models/baseline` (콤마로 구분하여 여러 모델 가능)
   - Datasets: `synthdog_ko_clean`
   - Sample Limit: 0 (전체) 또는 원하는 수
   - Device: cuda 또는 cpu
3. "Run Benchmark" 버튼 클릭
4. 결과 확인:
   - CER (Character Error Rate)
   - WER (Word Error Rate)
   - Throughput (처리량)
   - Latency (지연시간)
5. JSON 또는 CSV로 결과 다운로드

### 4. 지속적 학습 (Continuous Learning)

**전체 파이프라인 실행:**
1. "Continuous Learning" 탭으로 이동
2. 설정:
   - Base Model Path: 기준 모델
   - New Data Path: 새로운 학습 데이터
   - Dataset Name: 데이터셋 이름
   - Benchmark Datasets: 평가할 데이터셋들
   - Training Epochs: 학습 에폭 수
   - Regression Threshold: 회귀 감지 임계값 (0.02 = 2%)
3. Auto-promote 체크박스: 성능 향상 시 자동으로 production 모델로 승격
4. "Run Continuous Learning Pipeline" 버튼 클릭
5. 결과 확인:
   - Improvements: 성능 향상된 데이터셋
   - Regressions: 성능 저하된 데이터셋
   - 전체 Status (PASS/FAIL)

### 5. 단일 이미지 OCR (Single Image OCR)

1. "Single Image OCR" 탭으로 이동
2. 이미지 업로드
3. Ground Truth 입력 (선택사항)
4. 사이드바에서 전처리 옵션 설정
5. "Process Image" 버튼 클릭
6. 결과 확인:
   - Annotated image (bounding boxes)
   - Prediction text
   - Confidence, Inference Time, CER
   - Word-level 상세 정보
7. 신뢰도가 낮으면 "Retry with High DPI" 버튼으로 재시도

### 6. 에러 분석 (Error Analysis)

1. "Error Analysis" 탭으로 이동
2. 벤치마크 결과 JSON 파일 업로드
3. Top N 에러 수 설정 (슬라이더)
4. 에러 샘플 확인:
   - Ground Truth vs Prediction
   - Error Breakdown (Substitutions, Insertions, Deletions)
5. 에러 패턴 분석:
   - Total Errors
   - Average CER/WER
   - Top Substitution Patterns

### 7. 감사 로그 (Audit Logs)

1. "Audit Logs" 탭으로 이동
2. 필터 설정:
   - Start Date / End Date
   - Model Name
3. "Query Logs" 버튼 클릭
4. 통계 확인:
   - Total Inferences
   - Mean Inference Time
   - Mean Confidence
   - Mean CER
5. 최근 작업 로그 테이블 확인
6. "Export Report" 버튼으로 Markdown 리포트 생성

### 8. 배치 처리 (Batch Processing)

1. "Batch Processing" 탭으로 이동
2. 여러 이미지 파일 업로드
3. "Process Batch" 버튼 클릭
4. 진행 상황 확인
5. 결과 테이블 확인
6. CSV 파일로 다운로드

## 주요 기능별 워크플로우

### 완전한 데이터 파이프라인

```
1. Dataset Management → SynthDoG-ko 다운로드 (1000 samples)
2. Data Processing → 데이터 정제
3. Data Processing → Train/Val/Test 분할
4. Benchmark Execution → 모델 평가
5. Error Analysis → 에러 패턴 분석
```

### 지속적 학습 워크플로우

```
1. Continuous Learning → 새 데이터로 전체 파이프라인 실행
   → 자동으로 데이터 정제 → 학습 → 벤치마크 → 회귀 검사
2. 결과 확인 및 모델 자동 승격
```

### 단순 이미지 처리 워크플로우

```
1. Single Image OCR → 이미지 업로드 및 처리
2. 신뢰도 낮으면 → High DPI Retry
3. Audit Logs → 작업 로그 확인
```

## 설정 (Sidebar)

모든 탭에서 공통으로 사용되는 설정:

**Model Configuration:**
- Model: TrOCR-Korean, EasyOCR, PaddleOCR, Custom
- Adapter: LoRA adapter 이름 (선택사항)
- Engine: PyTorch, ONNX, OpenVINO, TensorRT

**Preprocessing:**
- DPI Scale: 1.0 ~ 3.0 (해상도 스케일)
- Denoise: 노이즈 제거
- Sharpen: 샤프닝

**Visualization:**
- Show Bounding Boxes: bbox 표시
- Highlight Low Confidence: 낮은 신뢰도 강조
- Low Confidence Threshold: 임계값 설정

**Audit Logging:**
- Enable Audit Logging: 자동 로그 기록

## 실제 모델 통합

현재 GUI는 mock OCR 모델을 사용합니다. 실제 모델을 통합하려면:

### TrOCR 통합 예시

`gui/streamlit_app.py`의 `mock_ocr_inference()` 함수를 수정:

```python
def real_ocr_inference(image, model_name, engine):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    # Load model (cache in session state for performance)
    if 'ocr_model' not in st.session_state:
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        st.session_state.ocr_model = model
        st.session_state.ocr_processor = processor

    processor = st.session_state.ocr_processor
    model = st.session_state.ocr_model

    # Run inference
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return {
        'text': text,
        'confidence': 0.95,  # Calculate actual confidence
        'words': [],  # Extract word-level results if available
        'inference_time_ms': 0,  # Measured separately
    }
```

### EasyOCR 통합 예시

```python
def easyocr_inference(image, model_name, engine):
    import easyocr

    if 'easyocr_reader' not in st.session_state:
        st.session_state.easyocr_reader = easyocr.Reader(['ko', 'en'])

    reader = st.session_state.easyocr_reader
    results = reader.readtext(np.array(image))

    # Combine all text
    text = ' '.join([r[1] for r in results])

    # Convert to word format
    words = [
        {
            'text': r[1],
            'box': [r[0][0][0], r[0][0][1], r[0][2][0], r[0][2][1]],
            'confidence': r[2],
        }
        for r in results
    ]

    return {
        'text': text,
        'confidence': np.mean([r[2] for r in results]) if results else 0,
        'words': words,
        'inference_time_ms': 0,
    }
```

## 트러블슈팅

### GUI가 실행되지 않음
```bash
# Streamlit 재설치
pip install --upgrade streamlit

# 포트 변경
streamlit run gui/streamlit_app.py --server.port 8502
```

### Import 에러
```bash
# 모든 의존성 재설치
pip install -r requirements.txt

# 프로젝트 루트에서 실행하는지 확인
pwd  # LLMOCR 디렉토리여야 함
```

### 데이터셋 다운로드 실패
```bash
# datasets 라이브러리 확인
pip install --upgrade datasets huggingface-hub

# 수동으로 스크립트 실행
python datasets/scripts/download_synthdog_ko.py --limit 100
```

### 벤치마크 실행 실패
- 모델 경로 확인
- 데이터셋 경로 확인 (datasets/processed/<dataset_name>)
- test.jsonl 또는 train.jsonl 파일 존재 확인

### 메모리 부족
- Sample Limit 설정하여 샘플 수 제한
- DPI Scale 낮추기
- 배치 처리 시 한 번에 적은 이미지 처리

## 성능 최적화

### 빠른 프로토타이핑
```
- Sample Limit 사용 (100~1000 samples)
- Device: CPU (CUDA 설정 불필요)
- DPI Scale: 1.0
```

### 프로덕션 사용
```
- 전체 데이터셋 사용
- Device: CUDA
- ONNX 또는 TensorRT 엔진 사용
- 모델 캐싱 (session state 활용)
```

## 변경 이력

### v2.0.0 (현재)
- ✅ Dataset Management 추가
- ✅ Data Processing 추가
- ✅ Benchmark Execution 추가
- ✅ Continuous Learning 추가
- ✅ 8개 탭으로 확장
- ✅ 모든 CLI 기능 GUI 통합

### v1.0.0
- Single Image OCR
- Error Analysis
- Audit Logs
- Batch Processing

## 문의 및 기여

이슈나 기능 제안은 GitHub Issues를 통해 제출해주세요.

## 라이센스

프로젝트 라이센스를 따릅니다.
