# Bài học vận hành MobiHet

> Repo fl_veremi_bert: bài học dưới đây kế thừa từ v2 / mobihet. Bài học mới của repo này ghi tiếp ở cuối file.

> Kế thừa từ repo `mobihet` / `mobihet-features` (29/10/2026): mục lịch sử là bằng chứng của các run cũ, không phải trạng thái repo này.

Đọc cùng [LESSONS kỹ thuật](../../../../../LESSONS.md) sau [CONTEXT](../../../../../CONTEXT.md). Lỗi kernel cụ thể (triệu chứng → cách tránh) ở
[sổ lỗi kernel](../../../../../docs/operations/notebook-errors.md); quy trình account/quota ở [kaggle](../../../../../docs/operations/kaggle.md), Git/DVC ở [backup (repo mobihet-features)](https://github.com/odixe06/mobihet-features/blob/main/docs/operations/backup.md).
Bài học ở đây là quy tắc rút ra khi chạy thí nghiệm và hạ tầng; bài học về phương pháp/train/đánh giá ghi ở LESSONS.

## Thiết lập host

- 26/10: nhận diện máy từ cả OS nền và desktop session: `/etc/os-release` báo Ubuntu nhưng
  session là Lubuntu/LXQt. RAM vật lý và swap phải đo riêng; không kế thừa hạn mức WSL cũ.
  Số đo và đường dẫn hiện tại chỉ giữ tại [local-env](../../../../../knowledge/local-env.md).
- 25/10: lỗi DNS/NVIDIA-SMI trong sandbox không chứng minh mạng hoặc driver host hỏng. Đối chiếu terminal
  host với quyền phiên; sau khi owner bật quyền, Git clone và CUDA hoạt động.
- Đích OneDrive hiện là rclone mount. Kiểm mount trước push để tránh ghi nhầm thư mục local bên dưới;
  kiểm cloud bằng rclone độc lập với VFS cache. [Remote và bằng chứng (repo mobihet-features)](https://github.com/odixe06/mobihet-features/blob/main/docs/operations/backup.md).

## Kaggle, notebook và đường pipeline

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
  [repro và recovery](../../../../../docs/operations/notebook-errors.md) giữ nguyên weights/source, không cần SGD lại.

- **29/10: `kaggle datasets files` trả tối đa 200 file/trang**, kể cả với `--page-size 1000`, và in dòng `Next Page Token = …` lẫn vào đầu ra
  `--csv`. Đối chiếu dataset nhiều file phải lặp `--page-token` tới khi hết token, rồi so tập tên với thư mục local (v3: 823/823).

## Collect, poll và máy local

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
  giữ `<page url="...">` của mọi page con. Đọc lại kiểm bảng liên tục + 10 metric + PDF, xem [quy trình](https://github.com/odixe06/fl_dagsnet_veremi_v2/blob/main/docs/operations/notion-results.md).

## FeatureBERT batch (13/11/2026)

- Kaggle `list_kernels` 2.2.5 trả `ref` + `last_run_time`, nhưng `slug`, `enable_gpu`, `current_version_number`
  có thể giữ mặc định rỗng/false/0. Không suy account rảnh từ `enable_gpu=False` của listing.
  `scripts/kaggle_inventory.py` đọc status mọi ref, đọc metadata của phiên chưa terminal để xác định GPU;
  query lỗi loại account khỏi phân bổ. Quota và credential đều account-scoped, không đổi login global.
- Batch mới chỉ nhận account không có GPU job khác; mỗi push đọc lại quota/slot, giữ ngân sách cho session cùng account.
  Reply push phải có version trước marker `.pushed`; status/pull dùng owner/slug/version. Intent thiếu marker cần điều tra trước retry.
- Resume notebook kiểm unique parent + state hai rank cùng round trước decode; copyfile giữ output có quyền ghi.
  Test notebook kiểm cả JSON string CANPerFL và parent thiếu/lệch rank. Audit completion dùng target truyền vào (CANPerFL 10).
- Watcher poll 1.200 s; lần đầu đọc được r1 sẽ kết thúc có chủ ý để phiên ghi timing rồi nối watcher.
  Collect/audit lỗi trả nonzero để supervisor đánh thức phiên; không để lỗi vận hành ẩn dưới exit0.
- Inventory 13/11: listing còn có placeholder không `ref` (ngày mặc định 2010-04-01). Gửi status tới slug rỗng gây 403,
  loại nhầm 9 account rồi plan không đủ slot. Bỏ placeholder không định danh trước query; status lỗi trên ref thật vẫn UNKNOWN.
  Regression 2 case (placeholder + ref thật; query timeout không biến thành account rảnh) PASS.
- Retry r2 13/11: ref thật cũng gặp HTTP429/404; quét 4 account × 4 status đồng thời gây rate limit.
  Chuyển status sang tuần tự, 2 account survey song song, retry read HTTP429 có giới hạn (10/20 s), hết retry vẫn UNKNOWN.
  Reprobe minhtriethihi: rate-limited một read, retry thành công, inventory đầy đủ/busy0.
- Plan không được hủy toàn batch vì chưa đủ account đọc được: khóa phần đã xác minh, giữ method còn lại trong `deferred`.
  Collector thử cấp account mới cho deferred mỗi tick 1.200 s, kiểm lại quota/slot trước push; source/scientific config giữ nguyên.
  Test allocation một account/quota thiếu, deferred chỉ launch khi inventory đọc được: PASS.

- Resume13/11: notebook v1 chỉ lưu state/model, cache NPY/Inductor ở temp không vào output. Không thể khôi phục cache chưa được lưu. Generator mới export Inductor tar, reuse parent NPY khi có; giữ pinned runtime, compile gate và scientific source. Thời gian warmup T4 mới cần đo, không coi cache config là bằng chứng tăng tốc.

- 15/11: worker gọi Python bằng absolute path vẫn cần `PATH` chứa env `nckh/bin` nếu subprocess gọi `kaggle` bằng tên. Thiếu PATH gây status UNKNOWN rỗng; không suy kernel lỗi. Worker collect mới export PATH trước chạy.

- 15/11: nbformat tự sinh ID cell ngẫu nhiên; tái sinh notebook cùng source/HP vẫn đổi SHA và phá batch khóa. T3 launcher cố định `t3-cell-NNN`; repro sinh hai lần: trước FAIL chỉ `/cells/*/id`, sau PASS không diff. Chỉ cập nhật notebook hash trước push khi không có intent/marker, giữ receipt batch cũ; không nới kiểm hash source/HP.

- 15/11: quota có thể giảm giữa plan và live pre-push; assertion `quota changed; inspect budget` xảy ra trước intent. Collector phải ghi WAIT cho run này rồi xử lý run khác, không dừng toàn batch. Regression collect2run: run đầu thiếu quota, run sau vẫn push; RED→GREEN, suite batch4/4 PASS. Không bỏ gate quota hoặc sửa scientific settings.

- 15/11 T3 resume: worker gọi Python tuyệt đối nhưng subprocess `kaggle` vẫn cần PATH conda. Thiếu PATH làm FileNotFoundError trước gửi push; giữ intent/reply lỗi, prepend thư mục sys.executable rồi retry. Xác minh CLI2.2.5 trong env nckh.

- T3 resume15/11: gen_notebook --args chỉ đổi command train; ô import vẫn lấy --method mặc định fedavg nếu không truyền riêng. Luôn truyền method và kiểm guard notebook bằng checkpoint thật trước push.

- Watcher batch hash gắn thư mục state với manifest bất biến; đổi slug/version phải dùng thư mục watch mới. Hash mismatch sau push chỉ cần phục hồi watcher, không push lại training.

## Backup và kiểm tra16/11

- DVCadd/status/push cùngrepo giữlock; chạy nối tiếp, đa luồng bằngpush-j16 trongmộtprocess. Lockcontention không phải artifactcorruption; không xóalock của jobkhỏe.
- Cachev4npy .part còn từpullfull không làmCM/state đãaudit thànhpartial; backup bỏcacheset táitạođược, giữmodel/source/CM vàgiữfile.part local, khôngtựxóa.
- Test dùngrootdataset phải kiểm actualparquet train/test, không chỉ thưmục tồn tại. ToyAFPHA cầnmetadatafixture riêng, không dựa dữliệulocal đangonline.
- Shutdown sauDVCsyncmứcmount cóthểcòncloudVFSdirty; ghi riênghai trạngthái, giữcachevàlocalartifact đểrclonetiếpupload saubậtmáy.

- Kaggle importzip dataset tựgiải nén: readylisting trả memberpaths, archive-name download404 không chứngminh datafail. Verify memberSHA vớisource, khôngreupload. SDK dataset_download_file giữrelativepath; truyền path=downloadroot, không path=parentlầnnữa. Kiểm mộtfilethật trướcfanout/fullstate.

- Recoverylauncher phải preflight bằng đúng cách chạy scripttrực tiếp; import package cóthể khác moduleimport trong REPL. Verify emittedimport trên checkpointthật vàAST toànnotebook trước retry/push; localcheckerfail không làtrainingfail.

- OneDrive sau reboot: rclone phục hồiVFSdirty trước READY cóthể lâu hơn TimeoutStartSec; systemdkill tạo vòngrestart vàmount mất. Kiểm journalboot hiện tại (lọc signedURL), tăngstarttimeout qua drop-in tải sau cấu hình cũ, daemon-reload vàgiữPID; kiểm `findmnt`/ismount/path thật trước DVC. Cache dirty chưaupload khôngxóa.
