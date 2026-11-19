"""Matched train/validation test of histogram-marginal synthetic sampling."""
import build_mobihet_stats_realbn_val as builder


if __name__ == '__main__':
    builder.template.OWNER = 'dangthiet'
    builder.SLUG = 'mobihet-v4-stats-hist-val-r10'
    builder.CANDIDATE = 'mobihet_stats_hist'
    builder.DESCRIPTION = (
        'Candidate maps correlated Gaussian coordinates through train-only class/feature histograms: '
        '256 fixed bins on [-4,4], occupied-bin means, no interpolation across atoms. '
        'The latent correlation comes from the existing Gaussian; not a fitted Gaussian copula. '
        'Preserves binary flags and approximates other marginal supports, but does not enforce joint constraints. '
        'Keep original real/synthetic CE, BN, uniform class sampling, optimizer and schedule. '
        'Additional one-time payload/client: 5,046,272 upload bytes and 2,523,136 download bytes; '
        'zero per-round payload. Bootstrap is recomputed from identical train data on resume. '
        'Fixed R10 screening gate; no automatic R50 launch. Official test excluded.'
    )
    builder.main()
