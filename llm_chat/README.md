# 오픈소스 LLM 채팅 (과제: Opensource LLM 설치 및 실행)

오픈소스 언어모델을 웹 서버에 직접 설치해 실행하는 채팅 앱입니다. ChatGPT 같은 외부 AI API를 쓰지 않고, 모델 파일을 내려받아 서버 CPU에서 계산합니다. 사용자는 아무것도 설치하지 않고 웹 주소로 바로 대화할 수 있습니다.

## 구성

| 구분 | 사용한 것 |
| --- | --- |
| 언어모델 | Qwen2.5-1.5B-Instruct (Alibaba, Apache 2.0) / EXAONE-3.5-2.4B-Instruct (LG AI연구원, 연구·교육용 라이선스) |
| 모델 파일 | GGUF 형식, Q4_K_M 4비트 양자화 (약 1.0GB / 1.6GB) |
| 실행 도구 | llama.cpp의 파이썬 바인딩 `llama-cpp-python` (CPU 실행) |
| 화면 | Streamlit 채팅 UI |
| 서버 | Streamlit Community Cloud (CPU 2코어, 메모리 최대 약 2.7GB, GPU 없음) |

## 모델을 고른 이유

- 서버에 GPU가 없고 메모리가 약 2.7GB라서, 수십억~수천억 개 매개변수 모델(예: DeepSeek 원본 671B)은 실행할 수 없습니다.
- 그래서 매개변수 15억~24억 개 모델을 4비트로 압축한 파일을 골랐습니다. 16비트로 저장하면 3~4.8GB라 메모리에 들어가지 않습니다.
- Qwen2.5-1.5B는 작고 빨라서 기본 모델로, EXAONE-3.5-2.4B는 한국어 특화 모델이라 비교용으로 넣었습니다.

## 기능

- 모델 선택(2종), 창의성(temperature)·답변 길이 조절
- 답변이 한 토큰씩 실시간으로 나오는 스트리밍 출력
- 답변마다 토큰 수, 첫 글자까지 걸린 시간, 초당 토큰 수 표시
- 수업 내용과 콘크리트 예측 프로그램 정보를 담은 시스템 프롬프트로 "데이터처리개론 AI 조교" 역할 부여
- 대화 기록 내려받기(.md), 서버 사양 표시, "LLM은 어떻게 대답할까?" 설명 탭

## 로컬 실행

```bash
pip install -r llm_chat/requirements.txt
streamlit run llm_chat/app.py
```

처음 실행하면 Hugging Face에서 모델 파일을 내려받습니다(1~2분).
