"""Train/val diagnostic: uniform Gaussian control vs real-only CE."""
import build_mobihet_stats_realbn_val as builder


if __name__ == "__main__":
    builder.template.OWNER = "dangthiet"
    builder.SLUG = "mobihet-v4-stats-nogauss-val-r10"
    builder.CANDIDATE = "mobihet_stats_nogauss"
    builder.DESCRIPTION = (
        "Diagnostic ablation: remove Gaussian generation, synthetic forward rows and synthetic CE. "
        "Keep private initialization, real logit-adjusted CE, optimizer, train split and schedule. "
        "Bootstrap moments/prior still computed and counted identically for compatibility; "
        "this is not a communication-minimized implementation. "
        "BN now sees real rows only and dropout RNG consumption changes with the forward shape, "
        "so effects are of removing the whole augmentation pipeline, not CE alone. "
        "Control is uniform mobihet_stats. Report all per-class changes, particularly 11/13/15, "
        "regardless of the standard improvement gate outcome. No automatic R50 launch."
    )
    builder.main()
