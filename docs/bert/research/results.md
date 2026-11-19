# FeatureBERT — kết quả audit16/11/2026

MobiHet/T3 đủr50 cả3seed41/42/43. Baseline42/42run đủendpoint, gồm pFedES41r50 đã audit; pFedES đã bổ sung lại mọi bảng. Cácbaseline khác50round;CANPerFL10fine-tune sauFedAvg50.

[8bảng so sánh hai model](comparison.md) · [số liệu từngseed/nguồn](comparison.json).

## MobiHet/T3:10metric

| Seed |accuracy|precision_macro|precision_micro|precision_weighted|recall_macro|recall_micro|recall_weighted|f1_macro|f1_micro|f1_weighted|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|41|0.927666857|0.928771758|0.927666857|0.920298886|0.910389141|0.927666857|0.927666857|0.914839434|0.927666857|0.919526756|
|42|0.927559986|0.928671519|0.927559986|0.920346528|0.909959726|0.927559986|0.927559986|0.914532369|0.927559986|0.919467423|
|43|0.927877819|0.928840229|0.927877819|0.920659307|0.910668851|0.927877819|0.927877819|0.914938911|0.927877819|0.919714605|

Macro-F1 **.914770238 ± .000211920**,mean±sampleSD, hạng1 trên bảng hiện hành. DAGSNetT3 **.907956319 ± .000331909**,hạng3. Không suy significance hoặc hiệu quả mixedmodel từ3seed/các thí nghiệm modelriêng.

## Bằng chứng và giới hạn

- Baseline105sessionCOMPLETE trước phiên cuối:executedsource41file/session khớpplan từngseed:44session base `cb193e6fd56fc2f0c6810745cc57d8242c77c366`,61session base `e8246054d476f0d3fe96a3171a4609b8dafd9db4`, 0overlay; hai commit chỉkhácdoc,41filecode cùngbyte; receipt `runs/maintenance-20261116/baseline-provenance.json`. BaERROR cũ được giữ riêng, không tínhPASS.
- T3BERT:CM/support/10metric PASS_BOUNDED; state2rank hữu hạn, strictrebuild77904tham số, HP/source/driver RNG hai rankPASS;2672+2671xe. ParentCM prefix byte-identical36/35/32round tương ứngseed41/42/43.
- Exacthead41 `minhtran0601/flb-t3-s41-p3/1`;42 `annhtnt/flb-t3-s42-p3/1`;43 `trietbackup/flb-t3-s43-p4/1`, cả3COMPLETE/r50.
- Chưa storedprediction→CM replay toàn mạng; baselinepull/snapshot nhỏ không chứng nhậnfullprivatebank hoặc exactresume; dataset feature-content fingerprint lịch sử chưa đủ.
- CM/10metric từngbaseline: `runs/batch{,-s41,-s43}/audit/*.json`; T3audit/state ở `runs/t3/`. Không dùng partialr45 thayr50.

Trạng thái job/backup/việc kế chỉ tại [continue](../operations/continue.md).

## pFedES: phiên cuối đã audit

Exact `outlookcheck/flb-pfedes-s41-p4/2`, nối parent `emtrangtri/flb-pfedes-s41-p3/1` r45→r50.
Pull158file,0thiếu; CM/support/10metric r1–50 PASS_BOUNDED. Parent45rows/CM giữ nguyên,41file executedsource khớpbaseGit/parent,HP/model giữ nguyên.
Statehai rankr50 đủ2672+2671xe, toànbank hữu hạn; strictrebuild xe đầu/cuối từngrank và proxy; proxyhai rankkhớp.
Baseline cũ thiếu torchCPU/CUDA RNG riêng; không gọi exact RNGresume. Chưa prediction→CMreplay toànmạng.

| Seed |accuracy|precision_macro|precision_micro|precision_weighted|recall_macro|recall_micro|recall_weighted|f1_macro|f1_micro|f1_weighted|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|41|0.902458888|0.888414311|0.902458888|0.895735890|0.867783699|0.902458888|0.902458888|0.873799410|0.902458888|0.895359746|
|42|0.902277988|0.884685097|0.902277988|0.895594092|0.868568561|0.902277988|0.902277988|0.872820494|0.902277988|0.895369379|
|43|0.903260478|0.889138766|0.903260478|0.896908712|0.868810404|0.903260478|0.903260478|0.875247284|0.903260478|0.896676049|

Mean±sampleSD Macro-F1 **.873955730 ± .001220923**; tổng50round **.1404MB/xe**, gồm1.404B lên+1.404B xuống/round.
[Audit receipt](pfedes-final-audit.json).
