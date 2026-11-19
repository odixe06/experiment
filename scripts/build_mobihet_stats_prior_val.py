"""Paired train/val screening for consistent real/synthetic target priors."""
import build_mobihet_stats_realbn_val as builder


if __name__ == "__main__":
    builder.template.OWNER = "dangthiet"
    builder.SLUG = "mobihet-v4-stats-prior-val-r10"
    builder.CANDIDATE = "mobihet_stats_prior"
    builder.DESCRIPTION = (
        "Real CE uses z + log(local prior) - log(global prior), whereas control Gaussian CE uses raw z "
        "with uniform class sampling. Candidate uses Gaussian CE(z - log(global prior), y), "
        "aligning its target prior with real CE under the class-conditional model assumption. "
        "Keep synthetic samples, weight, BN, initialization, optimizer and schedule unchanged. "
        "Prior comes only from existing train bootstrap; no extra payload or proxy. "
        "This tests objective consistency, not an established explanation of prior failures. "
        "Stop at R10 and audit locked gate; no automatic R50 launch."
    )
    builder.main()
