# 📚 데이터셋 관리 탭 사용 가이드

LLMOCR Desktop App의 Dataset Management 탭 완벽 사용 가이드입니다.

## 🎯 개요

Dataset Management 탭에서는 한국어 OCR 학습을 위한 데이터셋을 다운로드하고 관리할 수 있습니다.

## 📊 지원하는 데이터셋

### 1. SynthDoG-ko (자동 다운로드 지원)
- **제공**: Naver Clova AI
- **출처**: Hugging Face
- **크기**: 약 800,000개 샘플
- **특징**: 합성 한국어 문서 이미지
- **용도**: 사전 학습, 데이터 증강

### 2. AI-Hub (수동 다운로드)
- **제공**: 한국지능정보사회진흥원
- **출처**: AI-Hub 웹사이트
- **종류**:
  - 공공행정문서 OCR
  - 한국어 글자체 이미지
- **용도**: 실제 문서 학습

---

## 🚀 SynthDoG-ko 다운로드 방법

### Step 1: 탭 열기
1. LLMOCR Desktop App 실행
2. 상단 탭에서 **"🗂️ Dataset Management"** 클릭

### Step 2: 다운로드 설정

#### 2-1. Output Directory (출력 디렉토리)
```
기본값: datasets/raw/synthdog_ko
```

**변경 방법:**
- 텍스트 필드를 클릭
- 원하는 경로 입력
- 예: `datasets/raw/my_synthdog`

**권장 사항:**
- 프로젝트 루트의 `datasets/raw/` 하위에 저장
- 충분한 디스크 공간 확인 (1,000 샘플 ≈ 100MB)

#### 2-2. Sample Limit (샘플 수)
```
범위: 10 ~ 100,000
기본값: 1,000
```

**설정 방법:**
- 숫자 입력 필드 클릭
- 원하는 숫자 입력 또는 화살표 버튼으로 조정

**추천 샘플 수:**
```
🔰 테스트/프로토타입:     100 ~ 1,000 samples
📚 소규모 학습:         1,000 ~ 5,000 samples
🎓 중규모 학습:         5,000 ~ 20,000 samples
🏆 대규모 학습:        20,000 ~ 100,000 samples
```

**예상 다운로드 시간 & 용량:**
```
100 samples   →  ~1분,   ~10MB
1,000 samples →  ~5분,   ~100MB
5,000 samples →  ~20분,  ~500MB
10,000 samples → ~40분,  ~1GB
```

#### 2-3. Start Index (시작 인덱스)
```
범위: 0 ~ 무제한
기본값: 0
```

**사용 사례:**
- **재개**: 중단된 다운로드 이어받기
- **분할**: 여러 번에 나눠서 다운로드
- **샘플링**: 특정 구간의 데이터만 다운로드

**예시:**
```bash
# 첫 번째 다운로드
Start Index: 0
Sample Limit: 5000
→ 샘플 0 ~ 4999 다운로드

# 두 번째 다운로드 (이어받기)
Start Index: 5000
Sample Limit: 5000
→ 샘플 5000 ~ 9999 다운로드
```

### Step 3: 다운로드 실행

1. **"📥 Download SynthDoG-ko"** 버튼 클릭
2. 진행 상황 확인:
   - 프로그레스 바 표시
   - 실시간 로그 출력
3. 완료 대기 (시간은 샘플 수에 따라 다름)

### Step 4: 다운로드 중 확인사항

#### 로그 출력 예시:
```
[10:30:15] Starting download...
Command: python3 datasets/scripts/download_synthdog_ko.py --output_dir datasets/raw/synthdog_ko --limit 1000 --start_idx 0

Downloading SynthDoG-ko dataset (split: train)...
Output directory: datasets/raw/synthdog_ko
Processing samples (start: 0, limit: 1000)...
Downloading: 100%|████████████████| 1000/1000 [02:34<00:00, 6.47it/s]

Saving 1000 annotations to datasets/raw/synthdog_ko/annotations.jsonl...

✓ Successfully downloaded 1000 samples
  Images: datasets/raw/synthdog_ko/images
  Annotations: datasets/raw/synthdog_ko/annotations.jsonl
  Metadata: datasets/raw/synthdog_ko/metadata.json
```

#### 프로그레스 바:
- **불확정 모드**: 계속 움직이며 작업 진행 중을 표시
- 완료 시 자동으로 사라짐

### Step 5: 완료 확인

#### 성공 메시지:
```
✓ Download completed successfully!
```

#### 다이얼로그:
팝업 창에 "Dataset downloaded successfully!" 메시지 표시

#### 생성된 파일 확인:
```
datasets/raw/synthdog_ko/
├── images/                          # 다운로드된 이미지들
│   ├── synthdog_ko_00000000.jpg
│   ├── synthdog_ko_00000001.jpg
│   └── ...
├── annotations.jsonl                # 텍스트 레이블
└── metadata.json                    # 메타데이터
```

---

## 🏢 AI-Hub 데이터셋 다운로드

### Why Manual Download?
AI-Hub 데이터셋은:
- 로그인 필요
- 사용 약관 동의 필요
- 라이센스 확인 필요
→ 자동 다운로드 불가능

### Step 1: 데이터셋 선택

드롭다운 메뉴에서 선택:
- **admin_docs**: 공공행정문서 OCR
- **korean_fonts**: 한국어 글자체 이미지

### Step 2: 다운로드 가이드 확인

**"📋 Show Instructions"** 버튼 클릭

#### 표시되는 안내:
```
============================================
MANUAL DOWNLOAD INSTRUCTIONS
============================================

1. Visit: https://aihub.or.kr/aihubdata/data/view.do?currMenu=...

2. Click 'Download' button (requires login)

3. Extract downloaded ZIP file

4. Organize files as follows:

   datasets/raw/aihub_admin_docs/
   ├── images/           # All image files
   └── annotations/      # JSON/XML annotation files

5. Run the parsing script:

   python scripts/parse_aihub_format.py --input datasets/raw/aihub_admin_docs

============================================
```

### Step 3: 수동 다운로드 진행

1. **AI-Hub 웹사이트 방문**
   ```
   https://aihub.or.kr
   ```

2. **로그인/회원가입**
   - 계정 생성
   - 사용 약관 동의

3. **데이터셋 검색**
   - 검색창에 "OCR" 또는 "글자체" 입력
   - 원하는 데이터셋 선택

4. **신청 및 승인**
   - "데이터 신청" 버튼 클릭
   - 활용 목적 입력
   - 승인 대기 (보통 즉시 승인)

5. **다운로드**
   - "다운로드" 버튼 클릭
   - ZIP 파일 저장

6. **압축 해제**
   ```bash
   unzip AI-Hub-Dataset.zip -d datasets/raw/aihub_admin_docs/
   ```

7. **파일 정리**
   - 이미지를 `images/` 폴더로
   - 레이블을 `annotations/` 폴더로

---

## 💡 실전 팁

### 1. 인터넷 연결 확인
```bash
# 다운로드 전 네트워크 테스트
ping huggingface.co
```

### 2. 디스크 공간 확인
```bash
# macOS/Linux
df -h

# 권장: 최소 5GB 여유 공간
```

### 3. 백그라운드 작업
- 다운로드 중에도 다른 탭 사용 가능
- UI가 블로킹되지 않음
- 진행 상황은 로그에서 실시간 확인

### 4. 다운로드 중단 시
- 앱을 닫아도 안전
- Start Index를 조정하여 이어받기 가능

### 5. 네트워크 오류 시
```
Error: Connection timeout
→ 인터넷 연결 확인
→ 방화벽 설정 확인
→ VPN 사용 시 비활성화 시도
```

### 6. 권한 오류 시
```
Error: Permission denied
→ 출력 디렉토리 쓰기 권한 확인
→ 필요시 sudo 권한으로 실행
```

---

## 📁 다운로드 후 파일 구조

### SynthDoG-ko 구조:
```
datasets/raw/synthdog_ko/
├── images/
│   ├── synthdog_ko_00000000.jpg    # 이미지 파일
│   ├── synthdog_ko_00000001.jpg
│   └── ...
├── annotations.jsonl               # 레이블 (JSON Lines)
└── metadata.json                   # 메타데이터
```

### annotations.jsonl 형식:
```json
{"image_path": "images/synthdog_ko_00000000.jpg", "text": "한국어 텍스트", "source": "synthdog_ko", "original_idx": 0}
{"image_path": "images/synthdog_ko_00000001.jpg", "text": "다음 텍스트", "source": "synthdog_ko", "original_idx": 1}
```

### metadata.json 형식:
```json
{
  "dataset": "synthdog-ko",
  "split": "train",
  "source": "https://huggingface.co/datasets/naver-clova-ix/synthdog-ko",
  "total_samples": 1000,
  "start_idx": 0
}
```

---

## 🔄 다음 단계

데이터셋 다운로드 후:

### 1. Data Processing 탭으로 이동
```
🔄 Data Processing 탭에서:
1. Clean Dataset: 데이터 정제
2. Create Splits: Train/Val/Test 분할
```

### 2. 데이터 확인
```bash
# 이미지 개수 확인
ls datasets/raw/synthdog_ko/images/ | wc -l

# 레이블 개수 확인
wc -l datasets/raw/synthdog_ko/annotations.jsonl
```

### 3. 샘플 이미지 보기
```bash
# macOS
open datasets/raw/synthdog_ko/images/synthdog_ko_00000000.jpg

# Linux
xdg-open datasets/raw/synthdog_ko/images/synthdog_ko_00000000.jpg
```

---

## ❓ FAQ

### Q1: 다운로드가 너무 느려요
**A:**
- 샘플 수를 줄여서 테스트
- 인터넷 연결 확인
- 시간대를 변경해서 재시도 (서버 부하가 적은 시간)

### Q2: 다운로드가 중간에 멈췄어요
**A:**
- 로그에서 마지막 다운로드된 인덱스 확인
- Start Index를 그 다음 값으로 설정
- 다시 다운로드 실행

### Q3: 디스크 공간이 부족해요
**A:**
- Sample Limit을 낮춤
- 불필요한 파일 삭제
- 외장 하드로 출력 경로 변경

### Q4: Hugging Face 인증이 필요하다고 나와요
**A:**
```bash
# Hugging Face 토큰 설정
huggingface-cli login

# 또는
pip install huggingface_hub
python -c "from huggingface_hub import login; login()"
```

### Q5: 다운로드 속도를 높이고 싶어요
**A:**
- 유선 랜 사용
- 여러 번에 나눠서 다운로드
- 서버와 가까운 시간대에 다운로드

---

## 🎓 예제 시나리오

### 시나리오 1: 빠른 테스트
```
목적: 파이프라인 테스트
샘플 수: 100
예상 시간: 1분
용량: 10MB
```

### 시나리오 2: 소규모 프로젝트
```
목적: PoC, 데모
샘플 수: 1,000
예상 시간: 5분
용량: 100MB
```

### 시나리오 3: 실제 학습
```
목적: 모델 학습
샘플 수: 10,000
예상 시간: 40분
용량: 1GB
```

### 시나리오 4: 대규모 학습
```
목적: 프로덕션 모델
샘플 수: 50,000+
예상 시간: 3시간+
용량: 5GB+
```

---

## 🔧 트러블슈팅

### 문제: ModuleNotFoundError: No module named 'datasets'
**해결:**
```bash
pip install datasets huggingface-hub
```

### 문제: SSL Certificate Error
**해결:**
```bash
pip install --upgrade certifi
# 또는
export SSL_CERT_FILE=$(python -m certifi)
```

### 문제: Out of Memory
**해결:**
- 샘플 수를 줄임
- 다른 애플리케이션 종료
- 시스템 재시작

### 문제: Permission Denied
**해결:**
```bash
# 디렉토리 권한 변경
chmod -R 755 datasets/

# 또는 다른 경로 사용
Output Directory: ~/Desktop/datasets/
```

---

## 📚 관련 문서

- **다음 단계**: `DATA_PROCESSING_GUIDE.md`
- **전체 가이드**: `DESKTOP_README.md`
- **스크립트 문서**: `datasets/README.md`

---

**Happy Dataset Downloading! 📊**
