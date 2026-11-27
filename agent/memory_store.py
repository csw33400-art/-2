"""오래가는 기억: 서버가 다시 켜져도 메모가 남도록 GitHub에 저장한다.

Streamlit 무료 서버는 코드가 바뀌거나 오래 쉬었다 깨어나면 새 컴퓨터로 다시 켜져서,
임시 폴더의 메모 파일(llm_chat/tools.py의 MEMORY_FILE)이 사라진다.
그래서 메모를 이 저장소의 `memory` 브랜치에 있는 agent_memory.json에도 적어 두고,
서버가 켜질 때 다시 읽어 온다. (main 브랜치가 아니라서 저장해도 앱이 다시 배포되지 않는다.)

Secrets에 GITHUB_TOKEN(이 저장소 Contents 읽기·쓰기 권한)이 있을 때만 동작하고,
없으면 지금처럼 서버 임시 파일에만 적는다.
"""
import base64
import json
import threading
import urllib.error
import urllib.request

from programs import T

API = "https://api.github.com/repos/csw33400-art/-2"
BRANCH = "memory"
PATH = "agent_memory.json"


class MemoryStore:
    def __init__(self, token):
        self.token = token
        self.sha = None
        self.lock = threading.Lock()
        self.error = None
        self.loaded = False

    @property
    def enabled(self):
        return bool(self.token)

    def _req(self, method, url, body=None):
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None,
                                     headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                                              "User-Agent": "llm-agent-memory"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode() or "{}")

    def _ensure_branch(self):
        try:
            self._req("GET", f"{API}/branches/{BRANCH}")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            main = self._req("GET", f"{API}/git/ref/heads/main")
            self._req("POST", f"{API}/git/refs", {"ref": f"refs/heads/{BRANCH}", "sha": main["object"]["sha"]})

    def pull(self):
        """GitHub에 저장된 메모를 읽어 서버 임시 파일과 합친다."""
        if not self.enabled:
            return
        try:
            data = self._req("GET", f"{API}/contents/{PATH}?ref={BRANCH}")
            self.sha = data["sha"]
            saved = json.loads(base64.b64decode(data["content"]).decode("utf-8"))
            local = T.recall_all()
            seen = {(m["time"], m["summary"]) for m in saved}
            merged = saved + [m for m in local if (m["time"], m["summary"]) not in seen]
            T.MEMORY_FILE.write_text(json.dumps(merged[-50:], ensure_ascii=False), encoding="utf-8")
            self.error = None
        except urllib.error.HTTPError as e:
            self.error = None if e.code == 404 else f"불러오기 실패 HTTP {e.code}"   # 404: 아직 저장한 적 없음
        except Exception as e:
            self.error = f"불러오기 실패: {e}"
        self.loaded = True

    def push(self):
        """서버 임시 파일의 메모를 GitHub에 저장한다. 화면이 멈추지 않게 뒤에서 실행."""
        if not self.enabled:
            return
        threading.Thread(target=self._push, daemon=True).start()

    def _push(self):
        with self.lock:
            try:
                self._ensure_branch()
                body = {"message": "기억 메모 저장", "branch": BRANCH,
                        "content": base64.b64encode(json.dumps(T.recall_all(), ensure_ascii=False, indent=1).encode()).decode()}
                if self.sha:
                    body["sha"] = self.sha
                try:
                    self.sha = self._req("PUT", f"{API}/contents/{PATH}", body)["content"]["sha"]
                except urllib.error.HTTPError as e:
                    if e.code not in (409, 422):
                        raise
                    cur = self._req("GET", f"{API}/contents/{PATH}?ref={BRANCH}")     # sha가 어긋나면 새로 받아 다시 저장
                    body["sha"] = cur["sha"]
                    self.sha = self._req("PUT", f"{API}/contents/{PATH}", body)["content"]["sha"]
                self.error = None
            except Exception as e:
                self.error = f"저장 실패: {e}"
