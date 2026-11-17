"""Build-specific genomic masks used by cleaning, never by raw segmentation.

All intervals are zero-based, half-open. ``exclusion_block`` labels the connected
pieces of each chromosome outside the mask; cleaners must never merge different
blocks, even when the configured merge distance exceeds the excluded interval.
"""
from bisect import bisect_right
from importlib.resources import files

import pandas as pd


def _merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        start, end = int(start), int(end)
        if start < 0 or end <= start:
            raise ValueError(f"Invalid exclusion interval: {start}, {end}")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def validate_genome_build(genome_build):
    """Require an explicit supported assembly; never guess from coordinates."""
    if genome_build is None or not str(genome_build).strip():
        raise ValueError("genome_build is required; explicitly specify hg19/GRCh37 or hg38/GRCh38")
    aliases = {"hg19": "hg19", "grch37": "hg19", "hg38": "hg38", "grch38": "hg38"}
    build = aliases.get(str(genome_build).strip().lower())
    if build is None:
        raise ValueError(f"Unsupported genome_build {genome_build!r}; use hg19 or hg38")
    return build


def load_exclusions(genome_build, exclude_centromeres=True,
                    exclude_telomeres=False):
    """Load packaged masks. Unknown genome builds fail rather than silently mask.

    Accepted aliases are GRCh37/hg19 and GRCh38/hg38. Telomeres are optional;
    centromeres are excluded by default in MethylSeg v2.
    """
    build = validate_genome_build(genome_build)
    selected = set()
    if exclude_centromeres:
        selected.add("centromere")
    if exclude_telomeres:
        selected.add("telomere")
    result = {}
    resource = files(__package__).joinpath("data", "genomic_exclusions", f"{build}.bed")
    for line in resource.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        chrom, start, end, kind = line.split("\t")
        if kind in selected:
            result.setdefault(chrom, []).append((int(start), int(end)))
    return {chrom: _merge_intervals(intervals) for chrom, intervals in result.items()}


def split_regions(regions_df, exclusions, probe_data=None):
    """Subtract masks and label hard merge barriers, without length filtering.

    ``regions_df`` needs CpG_chrm/start/end. Unclipped rows retain their native
    statistics. Clipped rows require probe_data with CpG_chrm/CpG_beg/CpG_end/beta;
    probe_count and avg_beta are recalculated from fully contained probes, keeping
    duplicate probe coordinates. Fragments with no measured probes are omitted.
    Clipped fragments span the first retained CpG_beg through the last retained
    CpG_end, so mask subtraction cannot leave unmeasured sequence at their ends.
    """
    columns = list(regions_df.columns)
    if "exclusion_block" not in columns:
        columns.append("exclusion_block")
    if regions_df.empty:
        return pd.DataFrame(columns=columns)
    masks = {str(chrom): _merge_intervals(intervals)
             for chrom, intervals in exclusions.items()}
    probes_by_chrom = None
    if probe_data is not None:
        required = {"CpG_chrm", "CpG_beg", "CpG_end", "beta"}
        missing = required.difference(probe_data.columns)
        if missing:
            raise ValueError(f"probe_data missing columns: {sorted(missing)}")
        probes_by_chrom = {str(chrom): frame for chrom, frame in
                           probe_data.groupby("CpG_chrm", sort=False)}
    output = []
    for record in regions_df.to_dict("records"):
        chrom = str(record["CpG_chrm"])
        start, end = int(record["start"]), int(record["end"])
        intervals = masks.get(chrom, [])
        fragments = [(start, end)]
        for mask_start, mask_end in intervals:
            if mask_start >= end:
                break
            if mask_end <= start:
                continue
            fragments = [(left, right) for a, b in fragments
                         for left, right in ((a, min(b, mask_start)),
                                             (max(a, mask_end), b)) if left < right]
        clipped = fragments != [(start, end)]
        if clipped and fragments and probes_by_chrom is None:
            raise ValueError("Clipped regions require probe_data to recompute probe_count and avg_beta")
        ends = [b for _, b in intervals]
        for left, right in fragments:
            item = dict(record, start=left, end=right,
                        exclusion_block=bisect_right(ends, left))
            if clipped:
                probes = probes_by_chrom.get(chrom)
                if probes is None:
                    continue
                selected = probes.loc[(probes["CpG_beg"] >= left) &
                                      (probes["CpG_end"] <= right) &
                                      probes["beta"].notna()]
                if selected.empty:
                    continue
                item["start"] = int(selected["CpG_beg"].min())
                item["end"] = int(selected["CpG_end"].max())
                item["probe_count"] = int(len(selected))
                item["avg_beta"] = float(selected["beta"].mean())
            output.append(item)
    return pd.DataFrame(output, columns=columns)
