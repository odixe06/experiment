# MambaCNN — replacement candidate

Feng, Sun and Wang, *Mambacnn: a lightweight intrusion detection system based on mamba for the internet of vehicles*, Cybersecurity 9, 64, published 5 May 2026. [Publisher](https://link.springer.com/article/10.1186/s42400-025-00473-3), [open PDF](https://link.springer.com/content/pdf/10.1186/s42400-025-00473-3.pdf).

Screened 13 November 2026 as a replacement candidate for SR-LiCNN. Subsequently read all 23 pages of the owner's [local PDF](../../docs/paper/MambaCNN-KD_2026.pdf), including visual inspection of Figures 3 and 5. PDF SHA256 `0c5f2462d81180c6177e6ff32126fe0e1fbdb457e54589bbd85eb9797d01cb46`.

- Table 19 reports **128.11 K parameters for MambaCNN-KD on CIC-IDS2017**, after selecting 20 features. The uncompressed models have 187.82 K with selection and 342.27 K without selection. Do not attribute 128.11 K to the base architecture or to a future 77-feature VeReMi adaptation.
- The paper addresses in-vehicle CAN and external networks; datasets are Car-Hacking, IVN, Survival and CIC-IDS2017. The 128.11 K external variant is not a VeReMi experiment.
- Architecture combines Mamba branches, CNN and attention. Student is compressed through distillation from a stacking teacher involving MambaCNN, Decision Tree, Random Forest, XGBoost and LightGBM. Full pipeline has additional training cost beyond student parameter count.
- Full 23-page PDF opened successfully. No author code repository identified in the search; exact student reconstruction still needs architecture inspection and parameter verification.
- Journal **SCImago SJR 2025 Q1, SJR 1.056**, verified at [SCImago](https://www.scimagojr.com/journalsearch.php?q=21101019779&tip=sid). This is not a claim of JCR Q1 or a 2026 quartile.

## Reconstruction findings

- Pages 8–9, Figure 3: scalar feature embedding L×1 → L×D; three independent residual Mamba branches, each with three blocks in the base model; two branches feed cross-attention and the third self-attention. Three CNN blocks process raw input. Attention outputs form a 2D vector, concatenated with CNN features before a two-linear-layer classifier.
- Page 11, Figure 5: student has fewer layers; no numerical student depth or embedding dimension is specified. State dimension, expansion, convolution channels/kernels/strides, number of attention heads, pooling, normalization, activation choice and classifier width are not fully specified. Table 4 lists IEHO/loss settings, not layer dimensions. Parameter totals cannot uniquely determine these missing values.
- Original external pipeline: cleaning, min–max scaling, ADASYN, IEHO selection, stacked teacher, then student CE + feature matching + decoupled KD. Teacher has MambaCNN, DT, RF, XGBoost, LightGBM and a LightGBM meta-learner. Table 4 gives feature/KD weights 0.1/0.1 and target/non-target weights 0.2/0.8; no numeric temperature found.
- Equation 15 prints CE without the usual minus sign; equations 19/21 print KL(student || teacher). These equations are not used in the approved supervised-only adaptation (standard positive PyTorch CE).
- Table 10's Car-Hacking student count/time differ from adjacent prose (18.97K/2.82 ms vs 20.25K/2.49 ms). Table 16 lists 0.022 under Parameter(K), inconsistent with the scale in Table 21. Do not use these entries to infer an exact architecture. The external Table 19 128.11K matches its prose.
- Mamba background writes a general zero-order-hold SSM; implementation uses the standard Mamba-1 selective recurrence from [official Mamba module](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py) and [reference scan](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py), with input-dependent delta/B/C and delta*B*u input discretization. Our code is a restricted real-valued PyTorch implementation, not a copy of the paper authors' unavailable model code.

Owner approved a supervised adaptation retaining the three branches, CNN and both attention types, with chosen dimensions in 100k–150k parameters. [Implemented specification, verification and current run status](rebuild.md). Kaggle launch subsequently requested14November.
