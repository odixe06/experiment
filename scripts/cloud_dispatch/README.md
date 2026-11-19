# Kaggle FL dispatcher

Private hourly submission queue for the authorized SAE, MLP and CNN-BiGRU-Attention
single-backbone experiments: 14 methods × seeds 41, 42, 43. Existing submissions
are discovered across all nine authorized accounts and never resubmitted.
TabM was already fully submitted and is not part of the pending queue.

Runs at minute 17 every hour (UTC; GitHub scheduling may be delayed).
Training remains on Kaggle. The user's computer can be powered off after deployment.

- Uses GitHub-hosted standard Linux runner. Actions minutes are shared with other
  repositories under the owner's plan. Set an Actions $0 spending budget with
  **Stop usage when budget limit is reached** before enabling the schedule if paid
  overages are possible. No LLM/API/GPU is used by this dispatcher.
- Repository secret `KAGGLE_TOKENS_JSON` maps the nine account names to saved API
  tokens. No tokens are stored in source or notebook files.
- Fails closed if any account inventory/identity fails; all pages are checked.
- Counts queued/running/cancel-pending sessions, max two per account; requires
  8.25 free quota hours for each 8-hour session (7-hour notebook budget).
- Records intent in `queue.json` and pushes it BEFORE submitting to Kaggle.
  A lost reply is reconciled on the next tick. An invisible ambiguous submission
  requires manual review instead of risking a duplicate.
- CANPerFL uses FedAvg on the same account, backbone and seed. Requires COMPLETE,
  round-50 output manifest/status and state file; notebook asserts state/config.
- ERROR or partial training is not automatically restarted. `submitted` means
  Kaggle accepted the job, NOT that 50 rounds or artifact audits passed.
- Source frozen from the tested local checkout. `source_manifest.json` records
  hashes. Training code, data, scientific settings and tests are unchanged.
- `review` entries require inspection. Do not manually submit queue jobs while
  the scheduler is enabled; use the shared ledger to avoid racing submissions.

Actions → Hourly Kaggle dispatcher → Run workflow: default dry-run; select execute
to launch immediately. Disable the workflow to stop further submissions. Existing
Kaggle sessions continue. This dispatcher does not collect heavyweight artifacts.
The workflow disables itself when no pending submissions remain (including when
the only remaining entries need manual review). No repeated idle checks after that.
Initial hourly execution is gated by repository variable `DISPATCH_ENABLED=true`.
