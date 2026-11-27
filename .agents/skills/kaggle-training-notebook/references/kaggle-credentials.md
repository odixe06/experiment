# Saved Kaggle credentials on this host

Credentials live outside Git in `~/.kaggle/` (directory 0700, credential files 0600).

- `accounts/<username>.mcp-token`: owner-supplied KGAT API token, also usable by the CLI/SDK.
- `accounts-veremi/<username>.mcp-token`: KGAT tokens scoped to the experiment and 10k repos (27/11/2026). `kaggle_as.py`,
  `kaggle_account.py` and the kaggle-mcp scripts read them; other projects' helpers do not.
- `accounts/<username>.credentials.json`: optional OAuth snapshot created after browser login.
- `credentials.json`: active CLI OAuth login; do not replace it with a KGAT token.
- `account-roster.json`: private username/email mapping; never copy email/token values into skills or docs.

Use env `nckh`. Run each account-scoped command through the repository's existing `kaggle_as.py`;
this preserves the active global login. `KAGGLE_API_TOKEN` may contain the saved token FILE PATH,
which the installed SDK reads. Do not export it globally or put the raw value on a command line.

A token-only account can run account-scoped CLI commands without browser login. `kaggle_quota.py` and `kaggle_account.py health/plan` support both token-only and OAuth accounts.
Global `use/ensure` still requires OAuth snapshots; prefer `kaggle_as.py` for process-scoped commands. Use the existing authenticated CLI login. `list` inventories saved API tokens and OAuth snapshots; a missing snapshot
is not a CLI failure when another supported credential is available.

For a new OAuth account, use `kaggle_account.py add-account <username>`: save the current snapshot
before browser login, then save the new snapshot. The owner completes browser authorization.
Never invoke forced browser login unattended or silently overwrite another account's refresh token.

To verify a token-only identity, read its file in memory, build a Kaggle client with the token,
and call `security.oauth_client.introspect_token` with an `IntrospectTokenRequest` whose
`token` field is populated. Require `active=True` and a username matching the filename.
Suppress credential values and exception bodies; report only account, state and error class.
Quota/public metadata alone is not unique identity proof.

CLI is the default channel. Kaggle MCP is configured through the `kaggle-mcp` skill: `.mcp.json` calls its
header helper (token from `~/.kaggle/mcp-active`), and `kaggle_mcp.py` reads per account. Verify MCP with an
account-scoped native read.

Read [multi-account.md](multi-account.md) for account-scoped CLI operation. Current project/account policy
and local setup take precedence over dated observations in that reference.
