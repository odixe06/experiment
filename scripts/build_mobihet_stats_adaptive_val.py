"""Build train-only class-support adaptive Gaussian CE screening arm."""
import build_mobihet_stats_realbn_val as builder


if __name__ == "__main__":
    builder.template.OWNER = "dangthiet"
    builder.SLUG = "mobihet-v4-stats-adaptive-val-r10"
    builder.CANDIDATE = "mobihet_stats_adaptive"
    builder.DESCRIPTION = (
        "Gaussian CE class weight for client k is batch/(batch+n_kc), with n_kc from train only. "
        "Missing classes keep weight 1; well-supported classes receive less synthetic loss. "
        "Real CE, BN, batch size, LR and model unchanged. No proxy or per-round communication. "
        "Val-size stratification showed larger gains for smallest clients, but did not prove harm to large clients. "
        "This arm tests that hypothesis; no validation-derived weight fitting."
    )
    builder.main()
