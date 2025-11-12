# macOS 개발환경 부트스트랩 가이드

이 문서는 macOS에서 LLMOCR 프로젝트 개발환경을 설정하는 방법을 안내합니다.

## 개요

`mac_bootstrap.sh` 스크립트는 macOS에서 LLMOCR 개발에 필요한 모든 도구와 라이브러리를 자동으로 설치합니다. **이 스크립트는 최초 한 번만 실행하면 됩니다.**

## 설치 항목

### 1. Homebrew
- macOS용 패키지 관리자
- 설치되어 있지 않으면 자동으로 설치됩니다

### 2. 개발 도구
- **git**: 버전 관리
- **ffmpeg**: 멀티미디어 처리 (이미지/비디오 전처리에 유용)
- **pkg-config**: 컴파일러 설정 도구

### 3. Python 가상환경
- Python 3.11 기반 가상환경 (`.venv`)
- pip, wheel, setuptools 최신 버전

### 4. ML/AI 라이브러리

#### PyTorch 스택 (MPS 지원)
- **torch, torchvision, torchaudio**: PyTorch 핵심 라이브러리
- **MPS (Metal Performance Shaders)**: macOS Metal 가속 지원
- Apple Silicon (M1/M2/M3) 칩에서 GPU 가속 가능

#### Hugging Face 생태계
- **transformers**: 사전학습 모델 및 파이프라인
- **datasets**: 데이터셋 로딩 및 전처리
- **accelerate**: 분산 학습 및 혼합 정밀도 지원
- **peft**: Parameter-Efficient Fine-Tuning (LoRA 등)

#### FastAPI 스택
- **fastapi**: 고성능 웹 API 프레임워크
- **uvicorn**: ASGI 서버

#### 평가 도구
- **jiwer**: Word Error Rate (WER) 계산
- **python-Levenshtein**: 편집 거리 계산

#### 모델 최적화
- **onnx**: Open Neural Network Exchange 포맷
- **onnxruntime**: ONNX 모델 추론 엔진 (macOS arm64 공식 지원)
- **optimum**: Hugging Face 모델 최적화 및 내보내기

#### OCR 백엔드 (선택)
- **PaddleOCR**: 경량 OCR 엔진
- **PaddlePaddle**: PaddleOCR 백엔드

## 사용 방법

### 초기 설정 (최초 1회)

```bash
# 저장소 클론
git clone <your-repo-url>
cd LLMOCR

# 부트스트랩 스크립트 실행
./mac_bootstrap.sh
```

### 이후 개발 세션

```bash
# 가상환경 활성화
source .venv/bin/activate

# 개발 작업...
python your_script.py

# 작업 완료 후
deactivate
```

## MPS (Metal Performance Shaders) 지원

### MPS란?

MPS는 Apple의 Metal API를 활용한 GPU 가속 기술입니다. PyTorch에서 MPS를 사용하면:
- Apple Silicon (M1/M2/M3) 칩의 GPU를 활용 가능
- CPU 대비 훨씬 빠른 학습 및 추론 속도
- CUDA(NVIDIA GPU)와 유사한 사용법

### MPS 확인

부트스트랩 스크립트는 자동으로 MPS 가용성을 확인합니다. 수동으로 확인하려면:

```python
import torch
print(torch.backends.mps.is_available())  # True면 MPS 사용 가능
```

### 코드에서 MPS 사용

```python
import torch

# 디바이스 설정
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 모델을 디바이스로 이동
model = model.to(device)

# 텐서도 동일한 디바이스로
input_tensor = input_tensor.to(device)
```

## 참고 자료

### PyTorch MPS
- [PyTorch 공식 MPS 문서](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Metal Performance Shaders](https://developer.apple.com/metal/pytorch/)

### ONNX Runtime
- [ONNX Runtime 공식 사이트](https://onnxruntime.ai/)
- macOS arm64 공식 휠 제공 (별도 비공식 휠 불필요)

### PaddleOCR
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddlePaddle 공식 사이트](https://www.paddlepaddle.org.cn/)

## 문제 해결

### PaddleOCR 설치 실패

PaddleOCR/PaddlePaddle은 macOS와 Python 버전 조합에 따라 간헐적으로 설치 문제가 발생할 수 있습니다.

**해결 방법:**
1. 우선 TrOCR/Donut 경로로 프로젝트를 진행
2. PaddleOCR는 선택적 의존성이므로 없어도 핵심 기능 사용 가능
3. 추후 필요시 다음 방법 시도:
   ```bash
   # Python 버전 변경 (예: 3.10)
   # 또는 PaddleOCR 버전 다운그레이드
   pip install paddleocr==2.6.0 paddlepaddle==2.5.0
   ```

### Homebrew 설치 오류

Homebrew 설치 중 권한 문제가 발생하면:
```bash
sudo chown -R $(whoami) /opt/homebrew
```

### MPS 사용 불가

- **macOS 12.3 이상** 및 **Apple Silicon (M1/M2/M3)** 칩이 필요합니다
- Intel 기반 Mac에서는 MPS를 사용할 수 없습니다 (CPU 모드로 동작)

## 추가 정보

### 디스크 공간

전체 설치에는 약 **5-10GB**의 디스크 공간이 필요합니다:
- PyTorch: ~2GB
- PaddlePaddle: ~500MB
- 기타 라이브러리: ~1-2GB
- Homebrew 패키지: ~500MB

### 설치 시간

네트워크 속도에 따라 **10-30분** 소요됩니다.

## 라이선스

이 프로젝트의 라이선스를 따릅니다.
