"""Validation and coordinate masks for unobserved gaps between CpGs."""
from numbers import Integral


def validate_max_cpg_gap_bp(value):
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
    ):
        raise ValueError("max_cpg_gap_bp must be a positive integer or None")
    return None if value is None else int(value)


def missing_gap_masks(probe_data, max_cpg_gap_bp):
    """Return half-open unobserved intervals using start-to-start distances."""
    threshold = validate_max_cpg_gap_bp(max_cpg_gap_bp)
    if threshold is None or probe_data is None or probe_data.empty:
        return {}
    masks = {}
    for chrom, frame in probe_data.groupby('CpG_chrm', sort=False):
        frame = frame.sort_values(['CpG_beg', 'CpG_end'])
        starts = frame['CpG_beg'].to_numpy()
        ends = frame['CpG_end'].to_numpy()
        breaks = ((starts[1:] - starts[:-1] > threshold)
                  & (ends[:-1] < starts[1:])).nonzero()[0] + 1
        intervals = [(int(ends[i - 1]), int(starts[i])) for i in breaks]
        if intervals:
            masks[str(chrom)] = intervals
    return masks
