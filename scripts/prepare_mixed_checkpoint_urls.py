"""Prepare private expiring download links for exact-version diagnostic input."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'cloud_dispatch'))
import dispatcher as d
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
owner='voxuandatapply';slug='mix128-case2-v2x-htfl-s42-a1'
api=d.make_api((Path.home()/'.kaggle/accounts'/f'{owner}.mcp-token').read_text().strip())
names={'source_snapshot.json','source_manifest.json',slug+'/checkpoint.json',slug+'/rounds.jsonl'}
names.update(slug+f'/checkpoint-r015/state_rank{i}.pt' for i in (0,1))
urls={};token=None
with api.build_kaggle_client() as client:
    while True:
        req=ApiListKernelSessionOutputRequest()
        req.user_name,req.kernel_slug,req.version_label,req.page_size=owner,slug,'v1',200
        if token:req.page_token=token
        r=client.kernels.kernels_api_client.list_kernel_session_output(req)
        urls.update({f.file_name:f.url for f in r.files if f.file_name in names})
        token=r.next_page_token
        if set(urls)==names or not token:break
assert set(urls)==names,list(urls)
Path('runs/mixed128-checkpoint-urls.json').write_text(json.dumps(urls))
print('CHECKPOINT_URLS_READY',len(urls))
