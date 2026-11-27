# DVC upload throughput

Measure end-to-end time from local artifact to verified remote objects. Keep separate timings for DVC hashing and staging, transfer, and verification. A FUSE write can finish before cloud upload. On this host, serial `dvc add --to-remote --remote-jobs 1` through rclone VFS also consumed SSD cache space and blocked multi-GB bundles. More DVC workers may raise staging speed while the cloud transfer stays limited.

## Choose the path

- For a native DVC backend, group explicit pointers in one `dvc push -j N`. Benchmark a moderate worker count first, then raise it while throughput improves and rate limits, open files and memory stay healthy. Read the transfer log. Verify once with `dvc status -c`.
- For bulk DVC v3 artifacts already cached locally and a configured **rclone OneDrive mount**, prefer direct rclone transfer of selected objects to the **same** OneDrive directory. This avoids VFS upload staging. Resolve the live DVC remote URL and independently confirm its mount-to-rclone mapping, using the backup record and mount service. Translate the path relative to the mount to the configured rclone remote and append `/files/md5`. Keep the DVC URL and namespace as configured. Use the native path if mapping or cache layout is uncertain.
- Stage new targets with DVC on a volume with enough space. Same-filesystem hardlinks or reflinks can save copies where supported. Check the actual filesystem. ORICO exFAT needs room for copies if using normal `dvc add` there. For missing cache objects, repair the selected target or use native DVC for it. Do not pull unrelated history just to create a local cache.
- Finish or coordinate any writer for the same object namespace before switching paths. Keep rename and garbage collection out of an active transfer. Preserve the authorized destination and selected targets.

## Direct copy for selected pointers

Run from the repository whose DVC targets you selected. Use its DVC Python environment. Set `cloud_objects` to the confirmed rclone remote path ending in `/dvc-storage/files/md5` and pass explicit `.dvc` files. This example handles DVC v3 MD5 single-file and directory outputs. Use native DVC for other layouts and pipeline targets.

```bash
set -euo pipefail
cache_objects="$(dvc cache dir)/files/md5"
object_lists="$(mktemp -d)"
python /path/to/repo-backup/scripts/dvc-object-list.py \
  "$cache_objects" "$object_lists" path/to/run1.dvc path/to/run2.dvc

transfers=16
chunk=20Mi
for kind in blobs dirs; do
  [ -s "$object_lists/$kind.txt" ] || continue
  rclone copy "$cache_objects" "$cloud_objects" \
    --files-from-raw "$object_lists/$kind.txt" --no-traverse \
    --immutable --checksum --transfers "$transfers" --checkers 16 \
    --onedrive-chunk-size "$chunk" --stats 30s --stats-one-line
done
rclone check "$cache_objects" "$cloud_objects" \
  --files-from-raw "$object_lists/all.txt" --one-way --checkers 16
dvc status -c path/to/run1.dvc path/to/run2.dvc
```

The list includes directory manifests and deduplicates blobs by hash. Copy payload first and directory manifests last. `copy --immutable` preserves existing objects and can resume the same selection. A conflict requires an integrity check. Avoid `sync`, `--no-check-dest` and whole-cache uploads. Confirm the local and remote rclone backends share a checksum and that `rclone check` used it. Size-only comparison confirms presence and size. If hashes are unavailable and content verification is needed, use `rclone check --download` for the affected selection.

For a small delta, `--no-traverse` avoids a full remote listing. For a large batch with many existing objects, measure one listing against per-object probes and drop `--no-traverse` if listing is faster. Keep one final verification for the selected target set. A stale mount cache can lag the direct OneDrive result. Allow it to refresh, then rerun the selected DVC status check. Treat `deleted` as cache-only absence when the remote object is verified.

## Tune on actual new bytes

Start with 8 or 16 simultaneous files and 10–20 MiB OneDrive chunks. OneDrive chunks must be multiples of 320 KiB. Compare disjoint comparable selections of actual missing objects so deduplication does not inflate throughput. Try 16–32 transfers for many independent files and 20–40 MiB chunks for large shards if memory and transfer logs permit. A rough chunk buffer budget is `transfers × chunk size` plus rclone buffers and concurrent workload. Check backend flags on the installed rclone version before adding tuning options.

Use transferred bytes divided by elapsed wall time. Record retries, 429/503 responses, disk headroom and workload mix. Stop increasing concurrency when the uplink saturates, wall time stops improving or backoff grows. Reuse the fastest stable measured setting for similar artifacts. Use one background writer and one log. Avoid overlapping timing trials or repeated scans while the actual transfer runs. A local smoke test verifies DVC layout compatibility. A cloud throughput claim requires a real upload.

References: [DVC remote jobs](https://dvc.org/doc/command-reference/remote/modify), [rclone transfer options](https://rclone.org/docs/), [OneDrive backend](https://rclone.org/onedrive/).
