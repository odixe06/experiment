"""Refresh all saved accounts; preserve unknown states explicitly, never launch."""
import json
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.insert(0, str(Path(__file__).parent / 'cloud_dispatch'))
import dispatcher as d

def main():
    path = Path('runs/aligned-v2-inventory.json')
    previous = json.loads(path.read_text())
    accounts = sorted((Path.home()/'.kaggle/accounts').glob('*.mcp-token'))
    def one(p):
        owner = p.name.removesuffix('.mcp-token')
        try:
            s = d.checked_inventory(owner, p.read_text().strip(), previous.get(owner, {}).get('records', {}))
            return owner, {k:v for k,v in s.items() if k != 'api'}
        except Exception as exc:
            return owner, {'status':'UNKNOWN', 'error_type':type(exc).__name__}
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = dict(pool.map(one, accounts))
    result = {'checked_at':datetime.now(timezone.utc).isoformat(), 'accounts':rows}
    Path('runs/accounts-results-check.json').write_text(json.dumps(result,indent=2))
    for owner, row in rows.items():
        print(owner, row.get('active','UNKNOWN'), row.get('free_hours','UNKNOWN'), flush=True)

if __name__ == '__main__':
    main()
