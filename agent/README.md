# LLM 기반 자동 데이터분석 시스템 (마지막 과제)

> 과제: 오픈소스 LLM을 이용해 지금까지 본인이 만든 프로그램들을 대화를 통해 운용할 수 있는 프로그램을 만들어 제출하시오.

오픈소스 LLM이 대화를 듣고 **어떤 프로그램 기능을 실행할지 스스로 고른 뒤(도구 호출)**, 과제 1·2·3에서 만든 프로그램을 실제로 실행하고, **실행 결과를 다시 받아** 설명하는 웹 프로그램입니다. 화면에는 채팅창만 있고 클릭할 버튼은 없습니다.

- 🌐 **웹에서 바로 쓰기: https://csw33400-art--2-agentapp-meutiy.streamlit.app/**
- 📘 [프로그램 설명서](docs/PROGRAM.md): 구조, 사용법, 실행 결과, 한계
- 🧩 [코드 설명서](docs/CODE.md): 파일별 핵심 코드
- 📝 [시험 대비: 내 코드로 보는 시험 범위](docs/EXAM.md)

## 운용하는 프로그램

| 프로그램 | 파일 | LLM이 부르는 도구 |
| --- | --- | --- |
| 과제 1·2 회귀·인공신경망 대시보드 | [`app.py`](../app.py) | `run_dashboard_models`: 대시보드의 13개 모델을 같은 설정으로 학습·5겹 교차검증 |
| 과제 3 채팅 분석 도구 | [`llm_chat/tools.py`](../llm_chat/tools.py) | 데이터 불러오기, 변수 설정, 선형·비선형 회귀, 인공신경망, 딥러닝, 예측, 변화 곡선, 보고서 등 |
| 강의 지식·내 코드 | [`knowledge_base.py`](knowledge_base.py) | `search_lecture`(강의 RAG), `search_my_code`(코드 검색) |
| 기억 | `llm_chat/tools.py` 메모 | `remember`, `recall_memory` |

## 사용하는 LLM (모두 오픈소스, 비용 0원)

| 순서 | LLM | 언제 |
| --- | --- | --- |
| 1 | OpenRouter 무료 오픈소스 모델 (Qwen3-Coder → gpt-oss-20b → Llama 3.3 70B 순, 모두 응답 없을 때만 `openrouter/free`) | 기본. 도구 호출 기능으로 프로그램 운용 |
| 2 | 같은 모델에게 글(`<tool>{...}</tool>`)로 도구를 고르게 함 | 무료 모델이 도구 호출 기능을 지원하지 않을 때 |
| 3 | 서버에 설치한 Qwen2.5-1.5B (llama.cpp) + 낱말 규칙 | 열쇠가 없거나 무료 사용 한도(하루 50회)를 다 썼을 때 |

## 배포 (Streamlit Community Cloud)

1. share.streamlit.io → Create app → 저장소 `csw33400-art/-2`, 브랜치 `main`, Main file path `agent/app.py`
2. Advanced settings → Python 3.12, Secrets 칸에 아래 한 줄을 넣고 Deploy

```toml
OPENROUTER_API_KEY = "sk-or-로 시작하는 본인 열쇠"
```

열쇠를 넣지 않아도 예비 모드(서버 설치 모델 + 낱말 규칙)로 동작합니다. 열쇠는 GitHub에 올리지 않습니다.

## 로컬 실행

```bash
pip install -r agent/requirements.txt
streamlit run agent/app.py
```
