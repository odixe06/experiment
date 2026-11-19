"""Read exact-version mixed production status and persisted training health."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent/'cloud_dispatch'))
import dispatcher as d

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = []
    for job in json.loads((ROOT/'runs/mixed128-production-launch.json').read_text()):
        if job['state'] != 'submitted':
            continue
        row = {k: job[k] for k in ('case', 'method', 'seed', 'ref', 'version')}
        api = d.make_api((Path.home()/'.kaggle/accounts'/f"{job['owner']}.mcp-token").read_text().strip())
        try:
            assert d.kernel_metadata(api, job['ref']).current_version_number == job['version']
            row['status'] = d.read_call(lambda: api.kernels_status(job['ref'])).status.name
            if row['status'] in ('COMPLETE', 'ERROR'):
                dest = ROOT/'runs/mixed128-production'/job['slug']
                dest.mkdir(parents=True, exist_ok=True)
                api.kernels_output(job['ref']+'/'+str(job['version']), str(dest),
                    file_pattern=r'(?:RUN_STATUS\.json|rounds\.jsonl|experiment_manifest\.json|source_manifest\.json)$',
                    force=True, quiet=True)
                history = dest/job['slug']/'rounds.jsonl'
                rows = [json.loads(line) for line in history.read_text().splitlines()]
                assert [r['round'] for r in rows] == list(range(1, len(rows)+1))
                row['last_round'] = rows[-1]['round']
                row['max_skip'] = max(r['skipped_frac'] for r in rows)
                row['backbone_max_skip'] = {k: max(r[k] for r in rows)
                    for k in rows[-1] if k.startswith('health/') and k.endswith('/skipped_frac')}
                row['health_history_pass'] = all(r['applied_steps'] > 0 and r['skipped_frac'] < .01 for r in rows)
                row['health_history_pass'] &= all(v < .01 for v in row['backbone_max_skip'].values())
                row['last_metrics'] = {k:v for k,v in rows[-1].items()
                    if k.startswith(('client/', 'client_proto/')) and isinstance(v, (int,float))}
                marker = dest/job['slug']/'RUN_STATUS.json'
                row['completion_pass'] = False
                if marker.exists():
                    status = json.loads(marker.read_text())
                    assert all(status[k] == job[k] for k in ('case','method','seed'))
                    row['completion_pass'] = row['status'] == 'COMPLETE' and status['complete'] and len(rows) == 50
        except Exception as exc:
            row['check_error'] = type(exc).__name__+': '+str(exc)[:160]
        results.append(row)
        (ROOT/'runs/mixed128-production-live.json').write_text(json.dumps(
            dict(checked_at=datetime.now(timezone.utc).isoformat(), rows=results), indent=2))
        print(json.dumps({k:v for k,v in row.items() if k != 'last_metrics'}), flush=True)


if __name__ == '__main__':
    main()
