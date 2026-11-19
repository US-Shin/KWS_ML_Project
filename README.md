# Keyword Spotting for Smart Home Voice Commands

## 0. 프로젝트 정보
- **학번·이름**: 202235271 / 신의서
- **GitHub URL**: https://github.com/US-Shin/ML_Projects_KWS
- **데모 URL**: https://3c30fb6111d026cfaa.gradio.live/ (임시 Gradio Share 링크이며 실행 중인 앱에 따라 만료될 수 있음)

## 1. 문제 정의
- **Who**: 스마트홈 사용자, 음성 비서 사용자, 손이 자유롭지 않은 사용자
- **What**: 짧은 음성 명령어(예: on, off, stop, go 등)를 실시간으로 인식하여 기기를 제어
- **When**: 일상생활 중 스마트 스피커, 홈 IoT 기기 제어 시
- **Why**: 짧은 키워드만 처리하는 음성 명령 시스템을 통해 빠르고 프라이버시 친화적인 제어를 목표로 함
- **차별점**: 짧은 명령어 10개에 특화한 소규모 음성 분류 모델을 구현하고, 불확실한 입력은 예측을 보류
- **정량 목표**: 테스트셋 기준 Accuracy ≥ 0.85, Macro F1 ≥ 0.80

## 2. 데이터
- **출처**: Google Speech Commands Dataset v0.02 (코드가 TensorFlow 공식 다운로드 주소에서 직접 다운로드)
- **라이선스**: Creative Commons BY 4.0
- **규모**: 전체 105,829개 1초 음성 파일. 타깃 키워드별 최대 300개를 샘플링하고 unknown 및 silence 클래스를 별도 구성
- **구성**: 10개 타겟 키워드 (yes, no, on, off, stop, go, up, down, left, right) + unknown + silence 총 12 클래스
- **선정 이유**: 키워드 스팟팅 표준 벤치마크 데이터셋으로, 다양한 화자와 배경 소음이 포함되어 있어 현실적인 모델 학습에 적합

## 3. 데이터 전처리
전처리 기법 3가지 이상:
1. **음량 보존**: 입력 전체를 최대 진폭으로 나누지 않아 작은 마이크 잡음이 과도하게 증폭되는 것을 줄임
2. **Log-Mel Spectrogram 추출**: n_mels=40, n_fft=512, hop_length=160으로 특징 추출
3. **고정 길이 패딩/트리밍**: 1초(16kHz) 기준으로 시간 축을 맞춤. 긴 입력은 에너지가 큰 1초 구간을 선택
4. **무음 입력 처리**: RMS 음량 기준으로 매우 작은 입력을 음성 미검출로 안내

**분할**: train / val / test = 70% / 15% / 15% (stratified, random_state=42)



## 4. 모델
- **베이스라인 모델**: Simple CNN (3 Conv layers + Fully Connected)
  - 학습 방식: Scratch (처음부터 학습)
  - 선정 이유: 구현이 간단하고 빠른 베이스라인으로 적합
- **제안 모델**: Residual CNN (Residual Blocks 포함)
  - 학습 방식: Scratch (더 깊은 구조로 성능 향상 기대)
  - 선정 이유: Residual connection으로 더 깊은 네트워크 학습 가능, 베이스라인 대비 성능 개선 목표

두 모델 모두 PyTorch로 직접 학습.

## 5. 평가 결과
테스트셋 기준 지표 비교:

| 모델               | Accuracy | Macro F1 |
|-------------------|----------|-----------|
| Baseline CNN      |   0.8222  |   0.8237   |
| Proposed ResCNN   |   0.9185  |   0.9176   |

- 최종 모델: Proposed ResCNN (현재 테스트 결과에서 정확도와 Macro F1이 모두 더 높음)
- 분석: Confusion Matrix, 학습 곡선은 `results/` 폴더에 저장됨
- 오류 사례: silence와 unknown 클래스 간 혼동, 비슷한 발음 키워드(on/off 등) 혼동 가능성

상세 리포트는 `src/evaluate.py` 실행으로 생성.

## 6. 서비스 프로토타입
- **모델 연동 방식**: Gradio 웹 인터페이스에서 마이크 또는 업로드된 오디오를 받아 전처리 후 학습된 모델로 추론
- **화면 캡처**: `기말_캡처/` 폴더에 메인 화면 1장, 음성 입력 예시 3장, 플래그 화면 1장 포함
- **사용 안내**: 마이크 버튼 클릭 후 명령어 말하기, 또는 WAV 파일 업로드
- **예외 처리**: 입력이 없거나 음량이 매우 낮으면 안내하고, 신뢰도 또는 상위 예측 차이가 작으면 키워드 예측을 보류

실행: `python app/app.py` (share=True로 공개 링크 생성 가능)

## 7. 배포
- **플랫폼**: Gradio Share (임시 공개 링크). 지속 공개에는 Gradio 앱 호스팅 플랫폼 사용
- **공개 URL**: 위 데모 URL 참조 (임시 링크이며 영구 주소가 아님)
- 모델 가중치는 `models/` 폴더에 저장, 대용량은 Hugging Face Hub 링크 사용 가능

## 8. 실행 방법
```bash
# 환경 설정
pip install -r requirements.txt

# 1. 데이터 준비
python src/data_preparation.py

# 2. 베이스라인 학습
python src/baseline_model.py

# 3. 제안 모델 학습
python src/proposed_model.py

# 4. 평가
python src/evaluate.py

# 5. 데모 실행
python app/app.py
```

- **실행 환경**: Python 3.10+, VS CODE, CUDA 권장 (CPU도 가능하나 느림)
- **가중치 위치**: `models/baseline_cnn.pth`, `models/proposed_rescnn.pth`

## 9. Lessons Learned
### PMI
- **Plus**:
  - 키워드 스팟팅은 데이터 전처리와 특징 추출이 성능에 큰 영향을 미침을 확인
  - Residual 구조가 베이스라인보다 안정적으로 학습됨
- **Minus**:
  - silence와 unknown 클래스 구분 어려움
  - 소규모 샘플링 시 과적합 위험
- **Interesting**:
  - Mel Spectrogram만으로도 상당히 높은 정확도 달성 가능
  - 온디바이스 배포 가능성에 대한 인사이트

### 문제 해결 사례
- **문제**: Hugging Face 데이터셋 로딩 시 오디오 디코딩 오류
- **원인**: torchcodec / FFmpeg 의존성 문제
- **시도**: 대안 데이터셋 사용, librosa 기반 전처리로 전환
- **결과**: 안정적인 로컬 전처리 파이프라인 구축

### 향후 개선 계획
1. **데이터**: KWS-> 문장 인식으로 확장하고 LibriSpeech `train-clean-100`으로 문장 인식 모델 학습
2. **모델**: Whisper-tiny / Whisper-base 파인튜닝으로 인식 기능 향상
3. **서비스**: 실제 IoT 기기 연동 또는 엣지 디바이스(ONNX) 배포
4. **증강**: 작은 소리도 인식 가능하게끔 음량 증폭 / 마이크나 실제 환경 등 잡음에도 인식 가능한 전처리 파이프라인 구현하기

---
