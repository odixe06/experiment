"""Call one Kaggle MCP tool as one saved account, without touching the native MCP connection.

    python kaggle_mcp.py OWNER TOOL ['{"request": {...}}']     # read: status, info, quota, output listing
    python kaggle_mcp.py OWNER TOOL '{...}' --write            # write: kernel writes wait in the shared push guard
    python kaggle_mcp.py OWNER --tools                         # tool names (live tools/list)

Token: ~/.kaggle/accounts/OWNER.mcp-token, held in memory and never printed. Every tool takes {"request": {...}};
an empty argument means {"request": {}}. Prints the tool's text payload; exit 1 when the server flags isError.

Writes (WRITES below) need --write and run in env `nckh`: identity is confirmed by token introspection first, and
save_notebook / create_notebook_session run inside push_slot(OWNER) from the kaggle-push skill, the same lock and
120 s per-account cooldown as guarded CLI pushes. A write outside this script bypasses that lock.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://www.kaggle.com/mcp"
KERNEL_WRITES = {"save_notebook", "create_notebook_session"}
WRITES = KERNEL_WRITES | {"cancel_notebook_session", "upload_dataset_file", "update_dataset_metadata",
                          "create_model", "update_model", "update_model_variation", "submit_to_competition",
                          "create_code_competition_submission", "start_competition_submission_upload",
                          "create_benchmark_task_from_prompt", "create_benchmark_arena_settings",
                          "update_benchmark_arena_settings", "update_benchmark_arena_combinations"}


class Session:
    def __init__(self, owner):
        tok = Path.home() / ".kaggle/accounts-veremi" / f"{owner}.mcp-token"
        if not tok.is_file():
            tok = Path.home() / ".kaggle/accounts" / f"{owner}.mcp-token"
        self.token = tok.read_text().strip()
        self.sid, self.n = None, 0
        self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                "clientInfo": {"name": "kaggle_mcp.py", "version": "1"}})
        self.rpc("notifications/initialized", {}, notify=True)

    def rpc(self, method, params, notify=False):
        body = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notify:
            self.n += 1
            body["id"] = self.n
        head = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"}
        if self.sid:
            head["Mcp-Session-Id"] = self.sid
        try:
            reply = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), head), timeout=600)
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"kaggle-mcp: HTTP {exc.code} on {method}") from None
        self.sid = reply.headers.get("Mcp-Session-Id") or self.sid
        text = reply.read().decode()
        if notify or not text.strip():
            return None
        if not text.lstrip().startswith("{"):
            text = next(l[5:] for l in text.splitlines() if l.startswith("data:"))
        msg = json.loads(text)
        if "error" in msg:
            raise SystemExit(f"kaggle-mcp: {method} error {msg['error'].get('code')}: {msg['error'].get('message', '')[:200]}")
        return msg["result"]


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    owner, tool = argv[0], argv[1]
    if tool == "--tools":
        print("\n".join(t["name"] for t in Session(owner).rpc("tools/list", {})["tools"]))
        return 0
    write = "--write" in argv
    rest = [a for a in argv[2:] if a != "--write"]
    args = json.loads(rest[0]) if rest else {"request": {}}
    if tool in WRITES and not write:
        sys.exit(f"kaggle-mcp: {tool} is a write; rerun with --write under the owner's authorization")
    run = lambda: Session(owner).rpc("tools/call", {"name": tool, "arguments": args})
    if tool in WRITES:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kaggle-training-notebook/scripts"))
        sys.path.insert(0, str(Path.home() / ".claude/skills/kaggle-push/scripts"))
        import kaggle_account
        from push_guard import push_slot
        kaggle_account.saved_api_token(owner)              # raises unless the server says the token is OWNER's
        if tool in KERNEL_WRITES:
            with push_slot(owner):
                result = run()
        else:
            result = run()
    else:
        result = run()
    for part in result.get("content", []):
        print(part.get("text", ""))
    return 1 if result.get("isError") else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
