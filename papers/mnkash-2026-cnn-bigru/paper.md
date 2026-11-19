# CNN-BiGRU-Attention

Mnkash, Alawy and Ali, 2026. [Architecture paper](https://doi.org/10.67440/ahj.v21i4s.1182), [complexity analysis](https://doi.org/10.59543/jidmis.v3.1708). Local [PDF](../../docs/paper/CNN_BiGRU_Attention_2026.pdf).

Table 2 specifies 15 scalar features, valid Conv1D 64/kernel 3, BatchNorm, max-pool 2, valid Conv1D 128/kernel 3, bidirectional GRU 64 per direction, additive attention 128, dropout 0.3 and one sigmoid output. Sequence lengths are 15, 13, 6 and 4. Attention scores use vᵀ tanh(W h + b), softmax over positions, then a weighted sum of states. The table represents positions within one feature vector.

The text also mentions normalization/pooling after each convolution, conflicting with Table 2. The abstract mentions two independent BiGRUs, while Table 2 specifies one bidirectional layer. Equation 4 applies reset before the recurrent matrix and uses the update gate to weight the candidate. Standard PyTorch GRU uses reset after its recurrent projection and the opposite update-gate convention.

The companion analysis reports 116,097 parameters. Its BatchNorm row counts 256 although 64 channels have 128 trainable affine values. Its GRU uses one bias per gate. These conventions need explicit reconciliation in a PyTorch rebuild.

The source evaluates binary CIC-IoV 2024 detection with AWI federated aggregation. Its training uses AdamW, learning rate 0.001, weight decay 0.0001 and clipping 1, batch 128, early stopping and a plateau scheduler.
