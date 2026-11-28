# Bài học vận hành MobiHet

- 27/11: khi Kaggle hủy giữa phiên, mốc metric đã audit có thể mới hơn mốc checkpoint. CNN-BiGRU A4 s42 audit R23 nhưng cả hai shard full state ở R20. Đọc round và đối chiếu config trong từng shard, rồi dựng prefix lịch sử tới round checkpoint trước khi nối.

- 27/11 mất account Kaggle kéo theo mất checkpoint. Collector ablation cố ý bỏ `state*.pt` và `models/` khi pull, còn backup DVC chạy theo đợt. Vì vậy 25 ô MobiHet/ablation đã audit PASS mất checkpoint, nằm trên 15 account không còn truy cập. Trong đó 11 account bị gỡ ngày 25/11, `bestyorn` và `sampanh` không có credential, `nocknnock` và `shippertocdo` mất quyền ngày 27/11. Kết quả vẫn còn vì CM và metrics đã pull. Sau mỗi audit PASS, pull checkpoint và đẩy DVC ngay, đừng đợi đợt backup sau. Dùng `checkpoint-backup-20261125/inventory.py` rồi `direct-backup.py`.
- 27/11 trước khi tin `runs/round-metrics/index.json` để biết phủ checkpoint, đối chiếu với `experiments/*/results.csv`. Index vẫn ghi `mobihet_l4` (L4 có LA, đã loại) và trỏ checkpoint `runs/mixed-l4/` đã xóa. Pointer DVC còn có thể chỉ chứa metrics, không có `.pt` (`runs/pulls/fl2-{fedavg,fedprox,fedproto}.dvc`). Vì vậy phải liệt kê `.pt` trong manifest `.dir`, đừng chỉ dựa vào việc pointer tồn tại.

- 25/11 DVC OneDrive: upload rclone bị ngắt để lại object 0 byte dưới tên MD5 hợp lệ. Kiểm kích thước toàn bộ object theo manifest và index gốc, kể cả bundle đã có receipt. Xác minh MD5 bản gốc trước khi chuyển object lỗi vào quarantine. Đọc lại object đã phục hồi từ cloud và so MD5.
- 26/11 OneDrive trả `activityLimitReached` với `Retry-After` tới 55 giây. Tái dùng snapshot kích thước và kiểm trực tiếp các object vừa ghi. Giãn nhịp khi Graph giới hạn truy cập, tránh chạy lại listing đệ quy giữa các đợt ghi cùng kho.
- 26/11 kiểm checksum OneDrive: rclone 1.60 ở máy trả SHA1 trống. Bản 1.76 chạy riêng so QuickXor local với `lsjson --stat --hash` trên OneDrive. Dùng đường dẫn đã resolve khi `hashsum` gặp file staging qua symlink. [Rclone OneDrive](https://rclone.org/onedrive/)
- 25/11 backup checkpoint: size từ `kernels files` có thể là của mô tả file (~900 B), đồng thời mất đường dẫn run.
  Lấy đường dẫn qua output listing với exact version, đo byte từ `Content-Range` hoặc `Content-Length` của URL output trước khi chọn dung lượng.
  Không lưu signed URL vào biên bản. Giữ checkpoint dưới thư mục run có version riêng.
- 25/11: controller có thể báo job collector đã khởi động trước khi worker ghi PID. Kiểm PID collector sống trước khi mở waiter.
  Waiter thoát 3 với `WATCHER_NOT_RUNNING` ở bước này là lỗi thứ tự startup. Kiểm collector, receipt và số dòng ack trước khi mở lại waiter.
- 22/11 collector nền. Một bản duy nhất (`flock`), lọc trùng ref, khóa theo exact version: chống pull hai lần vào cùng thư mục. Chờ tiến trình con bằng `pgrep -P <pid>`; `pgrep -f "a|b"` trong `until` khớp chính dòng lệnh shell đó nên treo mãi. Phiên dừng dưới đích hoặc pilot → sự kiện `handoff.jsonl`, phiên agent chờ bằng background task đọc file local (không polling Kaggle/model). Ack theo số dòng đã xử lý, không theo độ dài file lúc ack.
- 20/11 Kaggle API có ConnectTimeout xen kẽ. Đường `www.kaggle.com/api/v1` theo từng tiến trình xác minh được roster, pull và push. Khi SaveKernel trả 400 với notebook nhúng 1,19 MB, kiểm slug chưa tồn tại rồi đóng gói source BERT gọn hơn. Phản hồi lần hai lên v1.

- 20/11 `pkill -f "<chuỗi>"` trong cùng lệnh bash có chuỗi đó ⇒ khớp chính shell, giết lệnh (exit 144), phần sau không chạy.
  Dùng `pgrep -f` + kill PID cụ thể, hoặc lệnh riêng. `kaggle kernels list -m` có thể trả rỗng dù account có kernel RUNNING:
  kiểm slot bằng status từng ref đã biết; push trả "Maximum batch GPU session count of 2" là bằng chứng chắc nhất.
- 20/11 builder đếm FLOP Kaggle clone GitHub tại SHA đã push; code chưa push ⇒ `build_kaggle_count.py --embed` (nhúng source
  worktree, sha256 kiểm trong notebook). Panel nhỏ: compile gate `G = group_rows/64` client > số xe nhóm ⇒ đệm client rỗng ⇒
  BN NaN ⇒ gate FAIL giả; gate dùng `G = min(4, K_nhóm)`.

- 19/11 terminal resume: `ERROR`/`CANCEL_ACKNOWLEDGED` không tự chứng minh có checkpoint dùng được.
  Chỉ resume khi tải được marker + checkpoint + history liên tục và fingerprint khớp; nếu child terminal
  thiếu artifact, quay về parent `COMPLETE` gần nhất có committed round, không suy round từ tên slug.

- 19/11 checkpoint transfer: multi-GB GET can fail midstream with ChunkedEncodingError. Resume notebook now uses existing fetch helper, .part + validated Content-Range and bounded retries; a server ignoring Range restarts the partial instead of appending. Tested both interrupted206 continuation and ignored-Range200 cases. Never print signed URLs on retry.

- 19/11 quota admission: a fixed1.1h minimum left usable short quotas idle. Match each continuation to persisted t_train+t_eval with startup/save margins; check remaining time after restore/cache. Running does not prove progress, especially cold compile on sub-hour sessions. Do not change model/round horizon to fit quota.
- 21/11. Dùng lại cache v4 NPY khi exact ref, manifest và mọi file qua SHA-256. Centralized resume so toàn bộ `config.data`, gồm hai metadata gốc ngoài 10 NPY. Listing API xác nhận quyền đọc output. Notebook kiểm mount thực và fingerprint trước train. Đo riêng thời gian mount, kiểm SHA và round đầu khi tính budget.

- 17/11 mixed diagnostic: Kaggle nhận notebook nhưng bỏ `kernel_sources` trỏ tới
  production ERROR (cả ref thường và ref/version); SDK trả invalid_kernel_sources
  dù notebook vẫn RUNNING/QUEUED. Phải kiểm phản hồi + metadata nguồn, không suy
  push thất bại nghĩa là không tạo session. Chuyển checkpoint bằng URL output
  tạm thời exact-version; notebook private, URL chỉ nằm trong runs/notebooks ignored.

- 15/11/2026 — Kaggle quota30h và push accepted không chứng minh account được cấp GPU. `iamabighotdog`
  nhận0 GPU ở4 notebook độc lập, chết đúng gate `torch.cuda.device_count()==2`. Sau owner phone-verify,
  probe private xác nhận đúng2 TeslaT4 capability7.5; mới mở khóa và retry version2 cùng scientific slug.
  Luôn giữ gate2×T4; không sửa model hoặc bỏ assert để tận dụng CPU.
- 14/11/2026 — Kaggle output > một trang phải kéo bằng `pull_kernel_output.py`; CLI một trang từng làm FD-IDS s42
  trông như chỉ có26 CM dù remote đủ50. Helper version-pinned kéo117/117 file,0 missing.
- 14/11/2026 — COMPLETE/ERROR có thể còn giữ batch-GPU slot; `kernels push` vẫn trả maximum2 sessions. Không coi
  status terminal là slot đã nhả; retry sau, không tạo slug né giới hạn.

> Kế thừa từ repo `mobihet` / `mobihet-features` (29/10/2026): mục lịch sử là bằng chứng của các run cũ, không phải trạng thái repo này.

Đọc cùng [LESSONS kỹ thuật](../../LESSONS.md) sau [CONTEXT](../../CONTEXT.md). Lỗi kernel cụ thể (triệu chứng → cách tránh) ở
[sổ lỗi kernel](notebook-errors.md); quy trình account/quota ở [kaggle](kaggle.md), Git/DVC ở [backup (repo mobihet-features)](https://github.com/odixe06/mobihet-features/blob/main/docs/operations/backup.md).
Bài học ở đây là quy tắc rút ra khi chạy thí nghiệm và hạ tầng; bài học về phương pháp/train/đánh giá ghi ở LESSONS.

## Thiết lập host

- 14/11 Windows: `kaggle_account.py` import `fcntl` ngay khi khởi động làm cả lệnh chỉ đọc `list/quota` chết trước API.
  Dùng `fcntl.flock` trên Unix, `msvcrt.locking` một byte trên Windows; không bỏ khóa hoặc dùng credential global song song.

- 26/10: nhận diện máy từ cả OS nền và desktop session: `/etc/os-release` báo Ubuntu nhưng
  session là Lubuntu/LXQt. RAM vật lý và swap phải đo riêng; không kế thừa hạn mức WSL cũ.
  Số đo và đường dẫn hiện tại chỉ giữ tại [local-env](../../knowledge/local-env.md).
- 25/10: lỗi DNS/NVIDIA-SMI trong sandbox không chứng minh mạng hoặc driver host hỏng. Đối chiếu terminal
  host với quyền phiên; sau khi owner bật quyền, Git clone và CUDA hoạt động.
- Đích OneDrive hiện là rclone mount. Kiểm mount trước push để tránh ghi nhầm thư mục local bên dưới;
  kiểm cloud bằng rclone độc lập với VFS cache. [Remote và bằng chứng (repo mobihet-features)](https://github.com/odixe06/mobihet-features/blob/main/docs/operations/backup.md).

## Kaggle, notebook và đường pipeline

- 14/11 Kaggle image có package top-level `tests` sẵn; thư mục `tests/` local không có `__init__.py` có thể bị che khi gọi
  `python -m tests.x`. Chạy file `python tests/x.py` với `PYTHONPATH` trỏ source root, hoặc biến local thành package rõ ràng.
  Probe TabM v1 chết trước train vì lỗi import này; v2 chỉ sửa launcher test.

- 31/10 RelKD val v1: `state_rank1.pt` cố ý chỉ giữ round/method_state/RNG; metadata chung `method/cfg/hp/gen` chỉ ở
  rank0 `state.pt`. Smoke checker không được assert `state["method"]` trên mọi rank. V1 đã train+resume smoke r1–2 khỏe rồi ERROR
  ở checker này; sửa assertion rank-aware, không đổi model/gate/data và repush cùng slug.

- 31/10 P0 eval-only: cùng checkpoint nhưng eager và compiled eval lệch vài dự đoán (accuracy ~1,2e-6, macro-F1 ~4,6e-6),
  làm percentile p10 client lệch ~5e-4. Assertion tái lập không dùng tolerance 1e-12 qua backend; khóa tolerance theo metric.
  Kernel ERROR sau khi đã ghi CM/JSON vẫn có thể audit artifact hợp lệ; không rerun chỉ để đổi nhãn terminal.

- 30/10: owner tắt W&B cho MobiHet mới. `gen_notebook.py --no-wandb` bỏ login cell và cờ `--wandb`;
  không cần nhúng key. CLI2.2.4 stream log có hậu tố `/version` trả404, cùng slug không version đọc được live log;
  vẫn giữ version cụ thể khi status/pull. Xác nhận version đang chạy trước khi diễn giải stream latest.

- **20/10 final startup crash:** test `--smoke --final` vẫn đọc pilot fixture, không chứng minh đường production final-only.
  Gaussian legacy còn hash cứng pilot manifest ⇒ r24 zc/zi crash trước round1. Test bắt buộc: không tồn tại pilot,
  Gaussian đọc final manifest; đối chiếu context pilot với legacy. Notebook phải lan truyền exit code subprocess,
  không biến `rc=1, epochs=0` thành Kaggle COMPLETE. Source đã push giữ nguyên; sửa ở tag mới.
- **20/10 upload source:** `datasets create` có thể exit1/JSON parse error dù server đã tạo dataset ready.
  Đọc status, verify manifest names/sizes trước retry; không upload/push trùng từ exit code đơn lẻ.
- **Sửa 20/10: CLI hiện tại đọc được log live bằng `kaggle kernels logs <owner/slug> --follow` (SSE).** Đã đọc W&B auth và
  `Prepared real pilot … old cut: 0` khi r24 vẫn RUNNING; lệnh không `--follow` có thể vẫn rỗng. Dùng account-scoped, đọc có timeout;
  exit 124 do hết thời gian đọc không phải kernel ERROR. Các ghi chú cũ "chỉ có W&B khi đang chạy" không còn đúng ở môi trường này.
  [Bằng chứng startup](https://github.com/odixe06/mobihet/blob/main/docs/archive/operations/final-r24-2026-10-20/final-startup-2026-10-20.json). Log/status chưa thay terminal artifact audit.
- Fixture được fingerprint phải chính là fixture driver đọc. `--final` trước 20/10 chọn dữ liệu final ở Run nhưng main hash pilot; chưa có run final production. Dùng chung `fixture_path`, và evaluator từ chối manifest khác training. Smoke dùng pilot không phát hiện được sai đường dẫn production.
- Exact-resume fixture phải cover state của cả local/control và FL arm. CrossClientRank pooled replay PASS
  đã bỏ sót CE empty-byte state, khiến production verifier dừng dù r3 hợp lệ;
  [repro và recovery](notebook-errors.md) giữ nguyên weights/source, không cần SGD lại.

- **29/10: `kaggle datasets files` trả tối đa 200 file/trang**, kể cả với `--page-size 1000`, và in dòng `Next Page Token = …` lẫn vào đầu ra
  `--csv`. Đối chiếu dataset nhiều file phải lặp `--page-token` tới khi hết token, rồi so tập tên với thư mục local (v3: 823/823).

## Collect, poll và máy local

- 15/11 GitHub Actions dispatcher: Kaggle SDK2.2.4/0.1.37 `kernels_list_with_response`
  có thể trả `enable_gpu=False`/`current_version_number=0` mặc định dù kernel GPU/version1 thật.
  Không dùng các field listing này để suy slot trống hoặc version parent; lấy metadata kernel cho version.
  `next_page_token` rỗng vẫn có trang2 khi trang1 đủ100: phải hỗ trợ `page=2` và chặn trang lặp.
  Status404 của draft phải đối chiếu metadata (version0), không coi mọi404 là không có job.
  Query status hàng loạt gây429: throttle và cache terminal theo ref/last_run_time, không retry push mù.
  GitHub runner sạch cần khai báo `six` cho stack Kaggle này; local cài sẵn che mất dependency thiếu.
  Phản hồi `save_kernel` không đủ để xác nhận identity/version: đã gặp job version1 QUEUED/RUNNING
  nhưng gate dựa ref/URL/version response báo review giả. Khi không có explicit error/invalid inputs,
  đọc lại kernel thật, so toàn bộ cell source, private/GPU và current_version_number; chỉ khi khớp mới ghi success.
  Draft status404 + metadata current_version0 không phải batch session; không để draft GPU cũ khóa account mãi.

- 14/11 Windows: đọc notebook UTF-8 bằng `Path.read_text()` mặc định có thể thành mojibake trong docstring, tạo source mismatch giả dù manifest/hash remote đúng. Dùng `encoding="utf-8"` khi đọc notebook/source để audit; không normalize hoặc sửa bytes artifact để ép khớp. Đã gặp khi audit CNN-BiGRU version1.

- 13/11/2026 Windows CPU test centralized gặp `WinError 5` khi rename thư mục epoch vừa ghi, kể cả ngoài sandbox. Retry có giới hạn 2 giây cho `PermissionError` trên Windows giữ thao tác commit nguyên tử và giúp 8 test hoàn tất, gồm resume. Lỗi cleanup memmap trong traceback là hậu quả của exception còn giữ tensor. Linux vẫn raise ngay. [Runner](../../scripts/run_central_backbones.py).

- Máy local ngủ (WSL) làm poll nền dừng, Kaggle vẫn chạy. Sau khi thức đừng để watcher mở collector cho mọi session cùng
  lúc (mỗi audit đỉnh ~4 GB): chia nhóm session rời nhau cho 2–4 tiến trình `collect_fd.py --only`, pull ~20 s/session.
- Collector có thể thoát 0 mà chưa pull nếu thiếu `PUSHED`. Kiểm `AUDITED.json` và completion cụ thể; chỉ dựng lại marker sau khi
  xác nhận đúng kernel từ status account-scoped, rồi vẫn đối chiếu executed source trước audit (ya-v2, 20/10).
- **Không tạm dừng (SIGSTOP) tiến trình pull; giới hạn audit song song theo RAM còn trống.** 22/10: dừng pull để ưu tiên kernel khác làm đứt
  stream (`ChunkedEncodingError`); cùng lúc audit lớn (~2,6–4 GB mỗi audit 100 client) với tác vụ khác của owner đẩy RAM trống còn ~600 MiB
  và bộ canh RAM dừng hai collector. Ưu tiên bằng thứ tự hàng đợi, tối đa 2 audit cùng lúc trên máy 10 GB; pull nối tiếp được từ `.part`.

## Git, DVC và lưu trữ cloud

- SHA của `.git/index` khác chưa có nghĩa nội dung staged đổi. Kiểm staged tree và HEAD riêng; ghi cả byte identity và content identity,
  không khôi phục index chỉ để ép SHA trùng snapshot cũ (20/10).
- 30/10: resume state của phương pháp per-client phải ghi **theo rank** (mỗi rank giữ bank client riêng); bản cũ chỉ rank 0 ghi ⇒
  resume sai lặng lẽ. Bank W = 1 ≈ 4,2 GB/rank, ghi atomic giữ 2 bản ⇒ vượt ~20 GB `/kaggle/working`: chỉ ghi khi dừng budget,
  xóa bản copy vào sau khi nạp. Quyết định dừng budget phải đồng thuận giữa các rank (all_reduce), không thì treo collective.
- 30/10: owner cấm trailer `Co-Authored-By:` trong commit (skill repo-backup); quy tắc này thắng reminder attribution của harness.
  Commit trước 30/10 ~02:00 còn trailer này — không sửa lịch sử (force-push đã bị từ chối).
- 30/10: `gen_notebook.py` chép `--set` thô vào literal Python `"..."` rồi shell, `run_fl.py` đọc bằng `json.loads` ⇒ giá trị chuỗi
  cần `--set "k='\\\"v\\\"'"`. Luôn `ast.parse` mọi cell + `shlex.split` lệnh trước khi push (bản đầu fl2-canperfl lỗi cú pháp).
- 30/10: method giữ tensor per-client trên host (APFed θ pinned ~4 GB/rank) không được tạo bản tạm cùng cỡ trong `state()`
  (`1 − θ` fp32): 2 rank × (θ + bank copy + tạm) vượt 30 GB RAM Kaggle ⇒ SIGKILL ở lần ghi state cuối, sau khi train xong. Tính theo khối.
- 30/10: mạng tới `api.kaggle.com` chập chờn (ConnectTimeout lúc listing) ⇒ status trả rỗng / pull fail ngẫu nhiên: thử lại, đừng đọc
  status rỗng là lỗi kernel. Pull nhiều kernel song song (mỗi kernel một tiến trình, `-j 16`) nhanh hơn chuỗi tuần tự.
- 30/10: resume notebook cũ dùng `shutil.copy` cho mọi mục nên lỗi khi có thư mục `models/`. NILM session 2 đã sửa **trong notebook**:
  unique parent, kiểm state r36 cả hai rank trước decode, `copytree(..., copy_function=copyfile)` cho thư mục và `copyfile` cho file;
  giữ source train `903ac42`. Generator resume chung chưa sửa; không tái sinh notebook nối mà bỏ mất preflight này.
- 30/10: chạy source local chưa push Git bằng `gen_notebook.py --overlay methods/<file>.py`: clone base SHA bất biến, decode byte nhúng,
  kiểm SHA256, ghi `provenance/source_manifest.json` và source vào output. Audit phải kiểm **base + overlay**, không chỉ Git HEAD.
- 30/10: Notion connector đã tạo page + upload PDF local được. `replace_content` cần `new_str`, không phải `content`;
  giữ `<page url="...">` của mọi page con. Đọc lại kiểm bảng liên tục + 10 metric + PDF, xem [quy trình](notion-results.md).
- 26/11: Kaggle CLI `kernels list --mine --csv` có thể exit 0 và in đúng `Not found` khi tài khoản chưa có kernel. Dispatcher chỉ coi đây là danh sách rỗng sau khi chạy dưới credential đã xác minh và ledger không có phiên đã nhận của owner. Lỗi API hoặc stdout rỗng vẫn là unknown.

# 16/11/2026 — ledger/push aligned-v2

- Parent private thiếu quota không có nghĩa phải đợi reset. Có thể chạy trên account khác của owner bằng
  cách tải output từ phiên parent cố định qua signed URLs trong notebook private; không nhúng API token.
  Kiểm GET (HEAD có thể404), byte count, ghi atomic, lưu SHA256 receipt, xác minh source hashes và cả2rank
  checkpoint đúng round/method/seed trước train. URLs có hạn: giữ trong runs/notebooks gitignored,
  không log URL hoặc raw exception chứa URL. Chỉ được gọi transfer PASS khi worker đã xác nhận restore.

- Không chạy hai launcher cùng ghi một JSON ledger song song: remote push đều thành công nhưng tiến trình ghi sau có thể
  làm mất entry của tiến trình kia. Chạy tuần tự hoặc dùng lock/append atomic; luôn đối chiếu biên nhận remote.
- Inventory cũ không được quyết định retry: entry `submitted` mới hơn mtime inventory phải chặn local để tránh tạo version
  trùng trước lần refresh tiếp theo.
- MLP + V2X-HtFL không qua smoke telemetry ở cả seed 41/42/43: `skipped_frac` r1 khoảng 39,7–40,8%,
  r2 khoảng 16,3–23,3%, vượt gate 1%. Đây là lỗi số học có lặp lại, không phải lỗi slot; không retry nguyên trạng.

## 20/11/2026 — gộp repo, centralized, compute

- Builder notebook nhúng **danh sách file cố định** (`build_central_backbones.py`): `core/central_backbones.py` đã chuyển sang
  `from core.tabm import TabM` nhưng danh sách không có `core/tabm.py` ⇒ 11 phiên version 1 ERROR ở cell test (mất vài phút quota mỗi
  account). Builder giờ dựng snapshot vào thư mục tạm và chạy test của nó trước khi ghi notebook; builder nhúng source khác cần guard tương tự.
- Khi chọn account cho phiên nối, kiểm quota và slot GPU riêng. Account có quota cao vẫn có thể hết slot vì job khác.
- Script Python đặt trong thư mục có `queue.py` (vd `runs/total-compute-20261119/`): thư mục script là `sys.path[0]` ⇒ `from queue import
  Queue` của stdlib (torch.fx) lấy nhầm file ⇒ ImportError. Bỏ thư mục đó khỏi `sys.path` trước khi import torch, nạp queue.py bằng importlib.
- Account Kaggle quota thấp: đặt `--max-hours` < quota live để runner dừng sạch ở biên epoch (nối tiếp được) thay vì bị Kaggle cắt.
- Test CPU một tiến trình không lộ lỗi chỉ có ở DDP 2 rank: `sae-ce` (decoder không vào loss CE) ⇒ DDP "parameters that were not used";
  `config.json` rank 0 đang ghi thì rank 1 đọc rỗng (JSONDecodeError, race có sẵn). Runner nhiều rank cần test `torchrun --nproc_per_node=2
  --cpu` (gloo) trên fixture trước khi push (`test_two_rank_ddp`).
- Dùng exact ref và version trong inventory để nối marker hoặc CM từ backup với results.csv. Một số lượt rerun dùng cấu hình khác dù
  trùng backbone, method và seed. Marker cuối có thể giữ đủ 10 metric. Ghi mức kiểm chứng marker hoặc CM vào cột `status`.
- Khi dời checkout sang ORICO exFAT, đo riêng kích thước logic của cache DVC và artifact. Hardlink trên SSD có thể làm
  `du` báo cache nhỏ khi đo chung hai cây. Giữ metadata và symlink trên SSD, kiểm SHA-256 bản dữ liệu trên ORICO trước khi bỏ bản trùng.
- 24/11: `setsid cmd & echo $! > pid` ghi PID của `setsid` cũ, không phải tiến trình sống khi setsid phải fork. Đọc PID thật bằng
  `ps -eo pid,args | grep "[s]lot_queue.py"` (mẫu `[x]` để không khớp chính lệnh grep; `pgrep -f "<chuỗi>"` khớp cả lệnh shell chứa chuỗi đó).
- 24/11: hàng đợi push gặp "Maximum batch GPU session count of 2 reached" dù batch chỉ có 1 kernel trên account (session khác không thấy
  qua listing). Lời từ chối rõ không tạo kernel: giữ notebook chờ, thử lại lượt sau (`runs/ablation-mh/slot_queue.py`). Chỉ dừng owner khi
  reply không rõ. Dừng hàng đợi chỉ lúc không có `push.sh`/`guarded_cli.py` con, và sửa state sau khi tiến trình đã dừng (nó ghi đè file).
- 24/11: repo cấu hình hook Git LFS nhưng máy không có `git-lfs` ⇒ `git push` bị hook pre-push chặn dù commit không có file LFS.

- **27/11/2026 — `.claude/skills` của repo này là symlink tới `.agents/skills`; ở repo 10k thì là hai thư mục riêng.** Một vòng sync
  kiểu `rm -rf <đích>; cp -r <nguồn> <đích>` đã xóa chính skill nguồn vì đích và nguồn trùng nhau. Trước khi sync, so `realpath` của nguồn
  và đích, và chỉ copy vào đích chưa tồn tại hoặc dùng `diff`/`cp` từng file, không `rm` đích.
- **27/11/2026 — account mới có quota 30 h vẫn có thể không có GPU.** 6/20 account nhận kernel với 0 hoặc 1 GPU, assert 2×T4 lỗi sau 20 s.
  Trước khi giao run dài, cần một kernel đã qua assert trên account đó, hoặc probe một cell in `torch.cuda.device_count()` (tốn khoảng 20 s).
- **27/11/2026 — MobiHet seed 42 (slug cũ `*-abl-a3-s42`) chạy 2 phiên: phiên 1 dừng ở R2 (`--stop-after 2`), phiên 2 là `-h2` cùng owner.** Khi chạy lại trên owner
  khác, phải đổi parent trong `kernel_sources` và trong manifest nhúng trong notebook. Audit hiện hành đã bỏ arm `a3`: dùng worktree
  `803a84f^` để audit.
- **28/11/2026 — `watch_collect.py` (ablation-mh) gọi thẳng `scripts/audit_ablation.py --record`, không qua `collect.sh`.**
  Muốn đổi cách ghi của collector đang chạy thì sửa `audit_ablation.py` (được nạp lại mỗi lần collect); sửa `collect.sh` không có tác dụng.
  Collector giữ state trong RAM và ghi đè `watch-collect.json` mỗi vòng: sửa state bằng tay chỉ khi collector đã dừng.
- **28/11/2026 — `rerun_dispatch.py` chỉ nối parent có `audit == PASS`; collector ablation-mh ghi run dừng trước R50 là `NEEDS_CONTINUATION`.**
  Phiên nối mixed phải build tay (`scripts/build_mixed_l4.py --session N+1 --parent … --npy-cache …`) và push bằng `push.sh` sau ≥ 15 phút.
- **28/11/2026 — `handoff.ack` là con trỏ số dòng của `handoff.jsonl`, không phải log.** Chỉ đổi bằng `wait_handoff.py --ack N`.
- **28/11/2026 — tiến trình chạy làm background task của phiên agent sẽ chết khi phiên dừng.** Collector/loop cần sống qua phiên phải chạy
  `setsid nohup … &`; kiểm `ps` trước khi bật để không chạy trùng (mỗi collector giữ flock riêng).
- **28/11/2026 — worker 10k (`kaggle_mixed_10k.py run`) giữ `worker.lock` độc quyền:** collect-loop và worker không chạy song song; một loop
  (`runs/mixed-10k/mobihet-20261128/loop.sh`) chạy tuần tự collect → auto-continue → dispatch. Builder 10k từ chối phiên nối khi `source_hashes`
  khác parent, nên không sửa `core/`, `methods/`, `scripts/run_mixed_fl.py`, `scripts/compute/` khi còn run cần nối.
