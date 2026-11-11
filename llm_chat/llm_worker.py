"""언어모델 전용 작업 프로세스.

언어모델(llama.cpp)을 웹 화면과 다른 프로세스에서 실행합니다. 모델이 메모리 부족이나
내부 오류로 죽어도 웹 앱(채팅 화면과 회귀·신경망 분석)은 계속 살아 있게 하려는 구조입니다.

통신 규칙 (한 줄에 JSON 하나):
    부모 → 자식 (stdin) : {"messages": [...], "temperature": 0.5, "max_tokens": 384}
    자식 → 부모 (stdout): {"ready": true} → {"t": "글자"} … → {"done": true} 또는 {"error": "이유"}

app.py가 `python llm_worker.py <repo_id> <filename> <threads>` 형태로 실행합니다.
"""
import json
import os
import sys


def main():
    repo, filename, threads = sys.argv[1], sys.argv[2], int(sys.argv[3])
    out = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")   # 통신 전용 통로
    os.dup2(2, 1)                                                     # 라이브러리가 찍는 글자는 로그로 보냄

    def send(obj):
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        out.flush()

    try:
        from llama_cpp import Llama

        llm = Llama.from_pretrained(repo_id=repo, filename=filename, n_ctx=4096,
                                    n_threads=threads, n_batch=128, verbose=False)
    except Exception as e:  # 설치 실패, 내려받기 실패 등
        send({"error": f"{type(e).__name__}: {e}"})
        return
    send({"ready": True})

    stdin = open(0, encoding="utf-8")
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            for chunk in llm.create_chat_completion(
                    messages=req["messages"], stream=True, temperature=req.get("temperature", 0.5),
                    top_p=0.9, repeat_penalty=1.1, max_tokens=req.get("max_tokens", 256)):
                text = chunk["choices"][0].get("delta", {}).get("content")
                if text:
                    send({"t": text})
            send({"done": True})
        except Exception as e:
            send({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
