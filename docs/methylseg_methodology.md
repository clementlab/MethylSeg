# MethylSeg Methodology

MethylSeg identifies large-scale DNA methylation domains by converting site-level methylation measurements into context-aware features, assigning each CpG to a biologically interpretable methylation state, and smoothing those assignments into genomic regions with a hidden Markov model (HMM). The workflow consists of five main stages:

1. methylation-data preprocessing;
2. context-aware emission generation;
3. biological state assignment;
4. HMM-based segmentation; and
5. region cleaning.

## 1. Data preprocessing

MethylSeg filters unreliable methylation measurements before generating model features. For WGBS data, two filtering strategies are available.

### Coverage filtering

CpGs with insufficient read coverage are removed. By default, MethylSeg retains loci with coverage greater than 10, although this threshold can be changed by the user.

### Low-coverage-like value filtering

When explicit coverage information is unavailable or an additional safeguard is desired, MethylSeg can remove CpGs whose beta values are characteristic of estimates derived from very few reads. The default set is:

$$
\left\{0,\ 0.25,\ 0.33,\ 0.50,\ 0.66,\ 0.75,\ 1.0\right\}.
$$

These discrete values can arise when only a small number of methylated and unmethylated reads contribute to an estimate. This filter is therefore a heuristic for identifying potentially low-depth measurements; it is not a substitute for direct coverage filtering when read counts are available.

## 2. Context-aware emission generation

A single CpG beta value contains little information about the larger methylation domain in which the locus occurs. MethylSeg therefore represents each CpG using both its observed beta value and summary features calculated from multiple genomic windows centered on that locus. Using several window sizes allows the model to capture methylation patterns at local, distal, and broad genomic scales.

Within each window, MethylSeg calculates:

- the mean beta value;
- the standard deviation of beta values;
- the proportion of lowly methylated CpGs;
- the proportion of intermediately methylated CpGs; and
- the proportion of highly methylated CpGs.

The observed beta value and window-level summaries are combined into a multiscale emission profile for each CpG. These profiles form the input to the biological state-assignment step.

## 3. Biological state assignment

Before HMM smoothing, MethylSeg assigns every CpG to one of four coarse biological states:

- **Low**: strongly hypomethylated loci;
- **PMD-associated**: intermediately methylated loci with a surrounding methylation profile characteristic of a partially methylated domain (PMD);
- **Intermediate**: loci with intermediate or heterogeneous methylation that do not exhibit the full PMD-associated profile; and
- **High**: strongly methylated loci.

MethylSeg supports two state-assignment methods: a data-driven K-means method and an interpretable rule-based method. Both methods operate on the same context-aware emission features and pass their per-CpG labels to the same downstream segmentation procedure.

### 3.1 K-means-based assignment

K-means is the default state-assignment method. It groups CpGs with similar multiscale emission profiles into four clusters. Because K-means cluster identifiers are arbitrary, MethylSeg maps the four clusters to the four biological states after clustering.

For each cluster, MethylSeg summarizes the mean beta value ($\bar{\beta}$), beta-value standard deviation ($\sigma_\beta$), and proportions of lowly ($P_{low}$), intermediately ($P_{int}$), and highly ($P_{high}$) methylated CpGs across the contextual features. It also calculates how closely the cluster's mean beta value falls within the expected intermediate-methylation range.

Let $C_{int}^{low}$ and $C_{int}^{high}$ denote the lower and upper intermediate-methylation cutoffs. The midpoint and half-width of this range are:

$$
M_{int} = \frac{C_{int}^{high} + C_{int}^{low}}{2}
$$

$$
H_{int} = \frac{C_{int}^{high} - C_{int}^{low}}{2}.
$$

The intermediate-beta score is then:

$$
S_{\beta,int} = \max\left(0,\ 1 - \frac{\left|\bar{\beta} - M_{int}\right|}{H_{int}}\right).
$$

MethylSeg calculates a score for assigning the cluster to each biological state:

$$
S_{low} = 2P_{low} + (1-\bar{\beta}) - 0.5P_{int} - 0.75P_{high} - 0.5\sigma_\beta
$$

$$
S_{high} = 2P_{high} + \bar{\beta} - 0.5P_{int} - 0.75P_{low} - 0.5\sigma_\beta
$$

$$
S_{PMD} = 3P_{int} + P_{low} - 1.5P_{high} + S_{\beta,int} - \sigma_\beta
$$

$$
S_{int} = 2P_{int} + 2P_{high} - 2P_{low} + S_{\beta,int} + \sigma_\beta.
$$

The Low and High scores emphasize enrichment for lowly and highly methylated CpGs, respectively, while penalizing characteristics associated with other states. Both the PMD-associated and Intermediate scores favor intermediate methylation, but they capture different surrounding patterns. The PMD-associated score favors intermediate and low methylation while penalizing high methylation and variability, reflecting the broad hypomethylation expected within PMDs. The Intermediate score permits a greater contribution from highly methylated CpGs and local variability, representing transitional or heterogeneous methylation patterns.

MethylSeg evaluates every possible one-to-one mapping between the four clusters and four biological states. It selects the mapping with the highest total score across all four cluster-state assignments. This global optimization prevents multiple clusters from receiving the same state label and makes the biological interpretation of K-means clusters consistent across model fits.

K-means clustering is performed with scikit-learn using 10 initializations and a fixed random seed of 42 for reproducibility.

#### When to use K-means assignment

K-means assignment is most appropriate when:

- state definitions should be learned from the training data;
- the fitted model will be reused across similarly prepared samples; or
- state assignment should incorporate the complete multifeature emission profile rather than explicit feature cutoffs.

#### Example

```python
from pathlib import Path

from methylseg import MethylSegPathway, MethylStateAssignmentMethod

sample_info, removed_df = MethylSegPathway.prepare_sample_info(
    sample_name="sample_1",
    sample_file="sample_1.tsv.gz",
    resolution="wgbs",
    min_coverage=10,
)

pathway = MethylSegPathway(
    train_sample_info=sample_info,
    state_assignment_method=MethylStateAssignmentMethod.KMEANS,
    out_dir=Path("out") / sample_info.sample_id,
)

pathway.fit_pathway()
regions = pathway.generate_regions(sample_info=sample_info, chrom="chr1")

pathway.plot_labels(
    label_source="kmeans",
    sample_info=sample_info,
    sample_info_removed=removed_df,
    chrom="chr1",
)
```

For a complete runnable example, see `examples/03_kmeans_based_model.ipynb`.

### 3.2 Rule-based assignment

The rule-based method assigns biological states directly from explicit emission-feature thresholds. During pathway fitting, MethylSeg optimizes the rule parameters and then applies the resulting cutoffs to each CpG emission profile.

State separation is based on:

- the beta value at the locus;
- the proportion of intermediately methylated CpGs in the surrounding windows;
- the within-window standard deviation of beta values;
- the proportion of highly methylated CpGs in each window; and
- the proportion of lowly methylated CpGs in each window.

In the default implementation, a CpG is labeled:

- **Low** when its beta value is below `beta_low_max` and the PMD rule does not apply;
- **High** when its beta value is above `beta_high_min` and the PMD rule does not apply;
- **PMD-associated** when its beta value is within the intermediate range and at least one contextual window satisfies the PMD cutoffs; or
- **Intermediate** when none of the preceding conditions is met.

Because the rules are explicit, this method can help determine whether a locus was assigned to a state because of its own beta value, local heterogeneity, or the methylation profile of the surrounding region.

#### When to use rule-based assignment

Rule-based assignment is most appropriate when:

- the interpretation of individual state assignments is a priority;
- a single sample is being examined closely; or
- biological states need to be described using explicit feature thresholds.

#### Example

```python
from pathlib import Path

from methylseg import MethylSegPathway, MethylStateAssignmentMethod

sample_info, removed_df = MethylSegPathway.prepare_sample_info(
    sample_name="sample_1",
    sample_file="sample_1.tsv.gz",
    resolution="wgbs",
    min_coverage=10,
)

pathway = MethylSegPathway(
    train_sample_info=sample_info,
    state_assignment_method=MethylStateAssignmentMethod.DEFINITION,
    out_dir=Path("out") / sample_info.sample_id,
)

pathway.fit_pathway()
regions = pathway.generate_regions(sample_info=sample_info, chrom="chr1")

pathway.plot_labels(
    label_source="rule_based",
    sample_info=sample_info,
    sample_info_removed=removed_df,
    chrom="chr1",
)
```

For a complete runnable example, see `examples/04_rule_based_model.ipynb`.

### 3.3 How assignment fits into the pipeline

`MethylSegPathway.fit_pathway()` trains the K-means components through `MethylStateAssigner.train_kmeans_for_sample()`. When rule-based assignment is selected, the pathway also optimizes the rule cutoffs through `MethylStateAnalyzer.optimize_rule_params_random()`.

During `generate_regions()` or `run_pathway()`, `MethylSegmentor.assign_states()` follows the selected branch:

- `MethylStateAssignmentMethod.KMEANS` applies the trained K-means model to each emission profile; or
- `MethylStateAssignmentMethod.DEFINITION` applies the optimized rule cutoffs directly to the emission features.

After this branch, both methods use the same HMM smoothing and region-generation workflow. Initial and smoothed labels can be compared with:

```python
pathway.plot_labels(label_source="kmeans")
pathway.plot_labels(label_source="rule_based")
pathway.plot_labels(label_source="hmm")
```

## 4. HMM-based segmentation

Per-CpG state assignments can change rapidly because of biological variability, measurement noise, and uneven spacing between observed CpGs. MethylSeg applies an HMM to smooth these initial labels and infer a coherent sequence of hidden methylation states. Consecutive loci assigned to the same smoothed state are then converted into genomic regions.

MethylSeg provides two HMM implementations:

- a continuous-time HMM implemented with `ctHMM` (v0.0.3); and
- a conventional categorical HMM implemented with `hmmlearn` (v0.3.3).

### Continuous-time HMM

The continuous-time HMM can account for the genomic distance between consecutive CpGs, which is useful because CpG spacing is irregular. By default, MethylSeg uses the forward-backward algorithm to calculate the posterior probability of each hidden state at every locus and assigns the state supported by those posterior probabilities.

### Categorical HMM

The categorical HMM treats the initial biological state labels as a sequence of discrete observations. MethylSeg uses the Viterbi algorithm to identify the most probable sequence of hidden states given the observed labels and fitted model parameters.

The HMM stage serves the same purpose for both assignment methods: it reduces isolated state changes and converts noisy, site-level labels into spatially coherent methylation domains.

## 5. Region cleaning

MethylSeg exposes region cleaning as a separate step so that the transformation from raw HMM calls to final regions remains transparent and configurable. Cleaning proceeds in three stages:

1. **Incorporate transitional regions.** Raw regions may optionally be merged with adjacent Intermediate regions, allowing transitional sequence to be retained in downstream analyses.
2. **Merge nearby regions.** Adjacent regions assigned to the same state are merged when the gap between them does not exceed 100 kb by default.
3. **Apply size thresholds.** Regions that do not meet the minimum CpG-count or genomic-length requirements are removed. The default thresholds are six CpGs and 5 kb per region.

The resulting cleaned regions are suitable for downstream analyses, while the raw calls remain available for users who want to inspect or customize the filtering process.

## Implementation reference

The main classes used by the workflow are:

| Class | Role |
|---|---|
| `methylseg.MethylSegPathway` | High-level interface for preparing data, fitting model components, running segmentation, and writing outputs. |
| `methylseg.MethylStateAssigner` | Generates emission features and trains or applies the K-means model. |
| `methylseg.MethylStateAnalyzer` | Optimizes and applies rule-based state definitions. |
| `methylseg.MethylSegmentor` | Assigns per-CpG states and converts HMM-smoothed labels into genomic regions. |
