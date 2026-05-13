# LoRA 학습 가이드 (SD 1.5 + Kohya_ss)

> GTX 1650 4GB VRAM 기준 | 캐릭터 반복 생성 목적

---

## 1. LoRA 개념

| 종류 | 설명 | 예시 |
|---|---|---|
| 캐릭터 LoRA | 특정 캐릭터를 학습 → 동일 캐릭터 반복 생성 | NarmayaLoRA |
| 품질 LoRA | 범용 디테일/스타일 향상 | more_details |

**캐릭터 반복 생성은 캐릭터 전용 LoRA가 필수.** IP-Adapter는 보조 역할.

---

## 2. 사전 준비

### 필수 소프트웨어
- Python 3.10 또는 3.11
- Git
- NVIDIA 드라이버 최신 버전
- CUDA Toolkit 11.8

### CUDA 확인
```bash
nvidia-smi
# 우측 상단에 CUDA Version 표시됨
```

---

## 3. Kohya_ss 설치

### 다운로드
```bash
git clone https://github.com/bmaltais/kohya_ss
cd kohya_ss
```

### setup.bat 실행 순서
- setup.bat 실행
> 1번 → Install kohya_ss GUI          (필수) 완료되면 자동으로 메뉴로 돌아옴  
> 2번 → Install CuDNN                 (권장, 속도 향상)
> 4번 → Install bitsandbytes          (권장, VRAM 절약)
>       → 옵션 선택: 4번 (Recommended) 0.41.2
> 6번 → Launch GUI in browser         (설치 확인) 브라우저 열림 

|번호 |	             항목          |	필요성 |	이유 | 
|----|----------------------------|--------|--------------| 
| 1 |	Install kohya_ss GUI        |	필수    |	본체 설치 | 
| 2	| Install CuDNN	              | 권장    |	학습 속도 향상 | 
| 3	| Install Triton	            | 하지 마세요 |	GTX 1650 불필요, 오히려 망가질 수 있음 | 
| 4	| Install bitsandbytes        |	권장    |	VRAM 절약 — 4GB에서 중요 | 
| 5	| Manually configure Accelerate |	나중에 |	자동설정으로 대부분 OK | 
| 6	| Launch GUI in browser         | 설치 후 |	설치 완료 확인용 | 

> ⚠️ 3번 Triton은 절대 설치하지 말 것 (GTX 1650 비호환)

### GUI 실행
```bash
gui.bat
# 브라우저에서 http://127.0.0.1:7860 자동 오픈
```

---

## 4. 이미지 준비

### 수량 및 조건
- 최소 12장, 권장 20~40장
- 최소 해상도: 768, 1024 이상(정사각형)
- 정면·측면·다양한 표정 포함
- 배경 단순한 것 선호
- mychar.safetensor : 50M 이상

### 해상도 부족 시 업스케일
`waifu2x-caffe` (애니 이미지에 최적)
> → 입력 폴더 선택   
> → 스케일: 2x   
> → 노이즈 제거: 2   
> → 변환 실행   
> → 결과: 약 1024×1024 (충분)   


### 폴더 구조 생성
```
📁 my_character_lora/
  📁 img/
    📁 12_mychar/        ← 반드시 "숫자_이름" 형식 숫자는 반복횟수 이미지겟수 아님
      🖼️ 001.jpg
      🖼️ 002.jpg
      ...
  📁 model/              ← 학습된 LoRA 저장 위치
  📁 log/
```

> `12_mychar` → 이미지 12장을 12번 반복 학습 (총 144 steps/epoch)

---

## 0. 전체 작업 순서 (빠른 참조)

### STEP 0 — GUI 실행
```
gui.bat → 브라우저 http://127.0.0.1:7860 자동 오픈
```

### STEP 1 — 캡션 자동 생성
상단 탭: **Utilities → Captioning → WD14 Captioning**

| 항목 | 설정 |
|---|---|
| Image folder | `절대경로/my_character_lora/img/39_mychar/` |
| Captioning method | WD14 Tagger |
| Prefix to add | `mychar,` (트리거 단어) |
| Threshold | `0.35` |
| Use onnx | ✅ |
| Remove underscore | ✅ |

→ **Caption images** 클릭  
→ 각 이미지 옆 `.txt` 파일 자동 생성

-  캡션 파일 확인
```
001.txt 예시:
mychar solo, looking at viewer,...
```
맨 앞에 트리거 단어 `mychar,` 확인



### STEP 2 — LoRA 학습 설정
상단 탭: **LoRA**

**Model 섹션**

| 항목 | 설정 |
|---|---|
| Pretrained model | `SD 1.5 체크포인트 .safetensors 파일 경로` (폴더 아닌 파일) |
| Trained model output name | `mychar` |
| Image folder | `절대경로/my_character_lora/img` |
| SD 버전 버튼 | **아무것도 선택 안 함** (SD 1.5 기본값) |  

> ⚠️ Pretrained model은 **폴더가 아닌 파일**을 지정해야 함  
> 예: `...\checkpoints\anythingV5nijimix_25BEST.safetensors`

**Folders 섹션**

| 항목 | 설정 |
|---|---|
| Output directory | `절대경로/my_character_lora/model` |
| Regularisation directory  | `비워두기` |
| Logging directory | `절대경로/my_character_lora/log` |

**Parameters 섹션 (GTX 1650 최적)**

| 항목 | 설정 | 비고 |
|---|---|---|
| Train batch size | `1` | VRAM 한계 |
| Epoch | `10` | Max train steps는 건드리지 말 것 |
| Max resolution | `1024,1024` | 세로 이미지 대응 |
| Enable buckets | ✅ | 다양한 비율 자동 처리 |
| Network Rank (Dimension) | `32` | 낮을수록 파일 작음·품질 낮음 |
| Network Alpha | `16` | Rank의 절반 권장 |
| Optimizer | `AdamW8bit` | bitsandbytes 필요, VRAM 절약 |
| Learning rate | `0.0001` | 1e-4 |
| LR Scheduler | `cosine` | |

> **Epoch vs Max train steps**  
> Epoch > 0 이면 Max train steps 무시됨. Epoch = 0 이면 Max train steps 기준.  
> 캐릭터 LoRA는 **Epoch = 10, Max train steps = 0** 권장.  
> 총 steps 예시: 이미지 12장 × 12반복 × 10epoch = **1,440 steps**

### STEP 3 — 학습 실행
화면 하단 **Start Training** 클릭

- 터미널 정상 출력 확인:
```
INFO  144 train images with repeats.   ← 이미지 인식 확인
INFO  class_tokens: mychar             ← 트리거 단어 확인
INFO  bucket 0: resolution (448, 832)  ← 해상도 버킷 확인
[1/10] loss: 0.08 ...                  ← 학습 진행 중
```
- 소요 시간
  GTX 1650 기준 약 30분 ~ 1시간 30분 소요  
- 터미널 창 닫기 ❌
- 브라우저 닫기 ❌
- PC 절전 모드 ❌

**완료**
```
Training complete → model/ 폴더에 mychar.safetensors 생성
```


### STEP 4 — 완료 후 파일 배포
```
my_character_lora/model/mychar.safetensors
       ↓ 복사
comfyUI_service/models/loras/mychar.safetensors
```

---

#### 문제 해결

| 오류 | 원인 | 해결 |
|---|---|---|
| `0 train images` | 이미지 폴더 경로 잘못됨 | `img` 폴더까지 지정, `12_mychar` 하위 폴더 확인 |
| `model is not found` | 모델 경로가 폴더로 지정됨 | `.safetensors` 파일까지 지정 |
| VRAM OOM | 메모리 부족 | batch size 1 유지, 다른 프로그램 종료 |
| 학습 완료 후 캐릭터 안 닮음 | Epoch 부족 또는 이미지 다양성 부족 | Epoch 15로 증가, 이미지 추가 |


####  재학습 필요한 경우

- 의상·헤어스타일 추가 시
- 결과물 품질 불만족 시 (이미지 추가 후 재학습)
- 다른 SD 1.5 체크포인트로 변경 시

한 번 학습한 LoRA는 동일 체크포인트에서 계속 재사용 가능.





















