---
name: kaggle-mcp
description: Use the Kaggle MCP server (https://www.kaggle.com/mcp) — connect it, pick its account, read kernel status/output/quota per account, and route any MCP write (save_notebook, create_notebook_session) through the shared push guard. Use when a task mentions Kaggle MCP or `mcp__kaggle__*` tools, when switching the MCP account, or when a push must go through MCP. Notebook building, collection and audits stay in kaggle-training-notebook.
---

# Kaggle MCP

Two channels, one token store: `~/.kaggle/accounts-veremi/<user>.mcp-token` (KGAT API tokens, 0600). Only the experiment
and experiment_10k_clients repos read that directory. Keep those tokens out of `~/.kaggle/accounts/` and any global config.
Older tokens in `~/.kaggle/accounts/` are still read as a fallback.

| channel | account | use for |
|---|---|---|
| native server `kaggle` (`mcp__kaggle__*`) | the one in `~/.kaggle/mcp-active` | interactive reads for that account |
| `scripts/kaggle_mcp.py OWNER TOOL '{"request": {...}}'` | any saved account, per call | multi-account reads; every MCP write |

A KGAT token also drives the CLI: `kaggle_as.py OWNER -- kaggle kernels push -p DIR` needs no browser login.
**The guarded CLI stays the default push path** (byte-exact from disk, ledger receipt). Reach for an MCP write only
when the CLI path is blocked.

## Push spacing — same rules on every channel

Every kernel write, whichever channel, obeys the kaggle-push skill:

- per account: 120 s after the previous push finished (`~/.kaggle/kernel-push-locks/<user>.state`);
- continuation: audit PASS of the exact parent version and 15 min after its recorded COMPLETE;
- one kernel per account per launch wave before any second slot; at most 2 batch GPU sessions per account.

`kaggle_mcp.py ... --write` enforces the first rule: it confirms token identity by introspection, then runs
`save_notebook` / `create_notebook_session` inside `push_slot(OWNER)`, the same lock as guarded CLI pushes.
The native `mcp__kaggle__save_notebook` and `mcp__kaggle__create_notebook_session` are denied in
`.claude/settings.json`, because a native call bypasses that lock. Run writes in env `nckh`, one owner at a time,
under an explicit launch authorization, and append the push to the project's ledger like a CLI push.

## Reads

- A private kernel reads only as its owner: status of `datle211206/x` through `ohkazu` returns
  `Permission 'kernels.get' was denied`. Pass the kernel's owner as OWNER.
- Every tool takes `{"request": {...}}`; field names are camelCase (`userName`, `kernelSlug`, `versionLabel` +
  `hasVersionLabel: true` to pin a version). Use the string `"v1"` for version 1.
  `python scripts/kaggle_mcp.py OWNER --tools` lists current schemas.
- `get_notebook_session_status` prints `{}` for QUEUED: the default enum value is omitted, so read an empty status as QUEUED.
- Common reads: `get_notebook_session_status`, `get_notebook_info`, `list_notebook_session_output`,
  `get_accelerator_quota` (seconds; `total_time_allowed` 108000 s = 30 h), `get_dataset_info`, `list_dataset_files`.
- Public datasets answer without auth, so only an account-scoped read (`get_accelerator_quota`) proves the token.
- Large outputs: pull with `pull_kernel_output.py` (kaggle-training-notebook), which pins the version, pages and checks every file.

## Native server setup

`.mcp.json` (git-ignored) holds no token; Claude Code runs the header helper on each connection:

```json
{"mcpServers": {"kaggle": {"type": "http", "url": "https://www.kaggle.com/mcp",
  "headersHelper": "python3 -I <repo>/.claude/skills/kaggle-mcp/scripts/kaggle_mcp_headers.py"}}}
```

Switch account: write the username to `~/.kaggle/mcp-active`, then the owner reconnects with `/mcp`
(a running session keeps its old connection and tool list). After reconnect, prove it with
`mcp__kaggle__get_accelerator_quota` and compare against `kaggle_mcp.py <user> get_accelerator_quota`.

## Save notebook through MCP

`save_notebook` re-sends the whole notebook in `textNullable` and runs it top to bottom. Build the request from the
files on disk: `slugNullable` = `kernel-metadata.json` `id`, text = the `.ipynb` bytes, `isPrivateNullable: true`,
`enableGpuNullable: true`, `machineShapeNullable: "NvidiaTeslaT4"` (any other value silently becomes one P100),
the metadata's dataset and kernel sources, and `enableInternetNullable` from the metadata. Read the schema with
`--tools`/`tools/list` before the first write; record the returned version as the exact ref.

## Tokens

The owner adds a token as `~/.kaggle/accounts-veremi/<user>.mcp-token` (one line `KGAT_…`, chmod 600). Verify with
`kaggle_account.py health` (introspection must name the same user). Keep token values out of Git, docs, command
lines and chat output; report account and error class only. Use an account only after a run on it has passed the 2x T4
assert (or a one-cell probe printed `2 ['Tesla T4', 'Tesla T4']`). An unverified account can run the kernel with 0 or 1 GPU.
