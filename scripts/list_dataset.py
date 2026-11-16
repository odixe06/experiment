"""Print name,size of state files in a dataset (all pages)."""
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate(); tok = None
while True:
    r = api.dataset_list_files(sys.argv[1], page_token=tok, page_size=200)
    for f in r.dataset_files:
        if f.name.split("/")[-1] in ("state.pt", "state_rank1.pt"): print(f"{f.name.split('/')[-1]},{f.total_bytes}")
    tok = r.next_page_token
    if not tok: break
