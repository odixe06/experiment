"""Paired screening of Gaussian class allocation with fixed total weight."""
import build_mobihet_stats_realbn_val as builder


if __name__ == "__main__":
    builder.template.OWNER = "dangthiet"
    builder.SLUG = "mobihet-v4-stats-massnorm-val-r10"
    builder.CANDIDATE = "mobihet_stats_massnorm"
    builder.DESCRIPTION = (
        "Adaptive class weights batch/(batch+n_kc) normalized to mean 1 within each client. "
        "This isolates class allocation from the previous arm's reduced total synthetic loss mass. "
        "Evidence: adaptive val R10 class13 +0.03754, class11 -0.03301, class15 -0.01316. "
        "Hypothesis, not proven cause. Train-only counts; no test-driven coefficient fitting. "
        "Control is uniform-weight mobihet_stats; same BN, Gaussian batch, optimizer and schedule."
    )
    builder.main()
