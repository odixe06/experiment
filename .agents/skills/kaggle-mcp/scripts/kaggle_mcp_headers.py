"""headersHelper for the native `kaggle` MCP server: print {"Authorization": "Bearer ..."} for the active account.

The active account is one username in ~/.kaggle/mcp-active (or KAGGLE_MCP_ACCOUNT); its token is
~/.kaggle/accounts/<username>.mcp-token. Claude Code runs this on each connection, so switching account =
edit the file, then reconnect the server (/mcp). Stdout carries the header only; errors name the account, never the token.
"""
import json
import os
import re
import sys
from pathlib import Path

home = Path.home() / ".kaggle"
user = os.environ.get("KAGGLE_MCP_ACCOUNT") or (home / "mcp-active").read_text().strip()
if not re.fullmatch(r"[a-z0-9_-]+", user):
    sys.exit("kaggle-mcp: ~/.kaggle/mcp-active must hold one Kaggle username")
tok = home / "accounts-veremi" / f"{user}.mcp-token"
if not tok.is_file():
    tok = home / "accounts" / f"{user}.mcp-token"
token = tok.read_text().strip()
if not token.startswith("KGAT_"):
    sys.exit(f"kaggle-mcp: {user}.mcp-token is not a KGAT API token")
print(json.dumps({"Authorization": f"Bearer {token}"}))
