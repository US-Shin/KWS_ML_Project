# Hugging Face Spaces 배포 방법

배포용 앱 파일은 `hf_space/` 아래에 독립적으로 구성했습니다. 학습 데이터 전체가 아니라 추론 코드와 제안 ResCNN 가중치만 포함합니다.

1. Hugging Face 계정으로 로그인한 뒤 새 Space를 만듭니다.
2. SDK를 **Gradio**로 선택하고, 제출 기준에 맞게 visibility를 **Public**으로 설정합니다.
3. `hf_space/` 안의 모든 파일을 Space 저장소 루트에 업로드합니다. `README.md`, `app.py`, `requirements.txt`, `models/`, `src/` 경로가 루트에 보여야 합니다.
4. 빌드가 완료된 뒤 Space의 페이지에서 마이크/업로드 입력으로 추론을 확인합니다.
5. 생성된 페이지 주소를 프로젝트 `README.md`의 공개 데모 항목과 `submission.json`의 `demo_url`에 기록하고, 서비스 화면 세 장을 `screenshots/`에 저장합니다.

Hugging Face Spaces는 계정 요금제/사용 가능 하드웨어에 따라 생성 가능한 런타임이 다를 수 있습니다. 이 앱은 일반 CPU 추론용으로 구성되어 있으므로, 계정에서 선택 가능한 Gradio 실행 하드웨어를 확인하세요. `hf_space/README.md`의 Space 메타데이터가 SDK 및 엔트리포인트 설정을 지정합니다.
