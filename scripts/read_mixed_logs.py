"""Bounded live-log reads; daemon readers cannot hold the local task open."""
import argparse
import json
import threading
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent/'cloud_dispatch'))
import dispatcher as d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('refs', nargs='+')
    ap.add_argument('--seconds', type=int, default=20)
    args = ap.parse_args()
    rows = {ref:[] for ref in args.refs}
    def read(ref):
        owner = ref.split('/')[0]
        api = d.make_api((Path.home()/'.kaggle/accounts'/f'{owner}.mcp-token').read_text().strip())
        try:
            for event in api.kernels_logs_stream(ref): rows[ref].append(str(event.get('data','')))
        except Exception as exc: rows[ref].append(type(exc).__name__)
    threads = [threading.Thread(target=read,args=(ref,),daemon=True) for ref in args.refs]
    deadline = time.monotonic()+args.seconds
    for t in threads: t.start()
    for t in threads: t.join(max(0,deadline-time.monotonic()))
    dest = Path('runs/mixed128-live-logs'); dest.mkdir(exist_ok=True)
    for ref,lines in rows.items():
        text = '\n'.join(lines)
        (dest/(ref.replace('/','__')+'.log')).write_text(text,encoding='utf-8')
        print(json.dumps({'ref':ref,'events':len(lines),'tail':text[-2400:]}),flush=True)


if __name__ == '__main__': main()
