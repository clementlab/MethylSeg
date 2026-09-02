# MethylSeg Methodology

MethylSeg is a methylome segmentation framework that uses context-aware methylation features to assign CpGs to methylation states and construct coherent genomic domains. The workflow consists of five main stages:

1. Data preprocessing
2. Context-aware emission generation
3. CpG state assignment
4. HMM-based smoothing
5. Region cleaning

## Workflow at a glance

MethylSeg converts CpG-level methylation measurements into coherent genomic domains through the following steps:

1. **Data preprocessing** - remove potentially unreliable methylation measurements.
2. **Context-aware emission generation** - summarize methylation across multiple genomic scales.
3. **CpG state assignment** - classify CpGs into Low, Intermediate, PMD-associated, or High states.
4. **HMM-based smoothing** - reduce isolated state changes and infer spatially coherent state sequences.
5. **Region cleaning** - merge and filter raw domains to produce final calls.

## 1. Data preprocessing

MethylSeg provides two approaches for filtering potentially unreliable methylation measurements: coverage filtering and low-coverage-like filtering.

### Coverage filtering

When sequencing coverage information is available, MethylSeg can remove CpG loci with insufficient read coverage. By default, loci with coverage greater than 10 are retained.

### Low-coverage-like filtering

When coverage information is unavailable, MethylSeg can identify beta values that are consistent with low read counts and remove them from the analysis. By default, these values are:

```text
0, 0.25, 0.33, 0.50, 0.66, 0.75, 1.0
```

This filtering can also be used alongside explicit coverage filtering.

## 2. Context-aware emission generation

MethylSeg represents each CpG using its observed beta value together with summary features calculated across multiple genomic windows centered on the CpG. Using multiple window sizes allows MethylSeg to capture methylation patterns at different genomic scales, from local variation to broader regional patterns.

Within each window, MethylSeg calculates:

- the mean beta value;
- the standard deviation of beta values;
- the proportion of lowly methylated CpGs;
- the proportion of intermediately methylated CpGs; and
- the proportion of highly methylated CpGs.

The observed beta value and these window-level summaries are combined to form a multiscale emission profile for each CpG. These emission profiles provide the input for the state-assignment step.

Methylation-state proportions are calculated using the following beta-value thresholds:

- **Low methylation:** beta value < 0.2
- **Intermediate methylation:** beta value >= 0.2 and <= 0.7
- **High methylation:** beta value > 0.7

The genomic window sizes are configurable, allowing users to adjust the genomic scales represented by the contextual features and identify methylation patterns at different scales.

## 3. CpG state assignment

Following emission generation, MethylSeg groups CpGs with similar context-aware methylation profiles into four states:

- **Low**
- **Intermediate**
- **PMD-associated**
- **High**

MethylSeg supports two approaches for state assignment:

1. K-means clustering
2. Rule-based assignment

The K-means approach provides a data-driven method for identifying methylation states, whereas the rule-based approach provides more explicit and interpretable state definitions.

### 3.1 K-means state assignment

MethylSeg first applies K-means clustering to the context-aware emission profiles. Because K-means cluster labels are arbitrary, the resulting clusters must subsequently be mapped to the four biologically interpretable methylation states.

For each cluster, MethylSeg calculates the mean beta value and the mean proportions of low-, intermediate-, and highly methylated CpGs across each contextual window. These summaries are used to calculate a state-specific score for every possible cluster-to-state assignment. The scoring functions are shown below.

$$
INT_{mid} = \frac{IntCutoff_{f_{high}} + IntCutoff_{f_{low}}}{2}
$$

$$
INT_{span\_half} = \frac{IntCutoff_{f_{high}} - IntCutoff_{f_{low}}}{2}
$$

$$
Score_{\beta\_Int} =
\max\left(
0,
1 -
\left|
\frac{\beta_{avg} - INT_{mid}}{INT_{span\_half}}
\right|
\right)
$$

$$
Score_{Low}
=
2 \times \%Low
+ (1 - \beta_{avg})
- 0.5 \times \%Int
- 0.75 \times \%High
- 0.5 \times \beta_{std}
$$

$$
Score_{High}
=
2 \times \%High
+ \beta_{avg}
- 0.5 \times \%Int
- 0.75 \times \%Low
- 0.5 \times \beta_{std}
$$

$$
Score_{PMD}
=
3 \times \%Int
+ \%Low
- 1.5 \times \%High
+ Score_{\beta\_Int}
- 1 \times \beta_{std}
$$

$$
Score_{Int}
=
2 \times \%Int
+ 2 \times \%High
- 2 \times \%Low
+ Score_{\beta\_Int}
+ 1 \times \beta_{std}
$$

The variables used in these scores are summarized below:

| Variable | Description |
|---|---|
| `beta_avg` | Mean beta value for the cluster across the contextual window. |
| `beta_std` | Standard deviation of beta values for the cluster across the contextual window. |
| `%Low` | Proportion of lowly methylated CpGs in the contextual window. |
| `%Int` | Proportion of intermediately methylated CpGs in the contextual window. |
| `%High` | Proportion of highly methylated CpGs in the contextual window. |
| `IntCutoff_f_high` | Upper intermediate-methylation cutoff used for the feature being scored. |
| `IntCutoff_f_low` | Lower intermediate-methylation cutoff used for the feature being scored. |

The **Low** and **High** scores emphasize enrichment of lowly and highly methylated CpGs, respectively, while penalizing characteristics associated with alternative states. Both the **PMD-associated** and **Intermediate** scores emphasize intermediate methylation but distinguish the states using their broader methylation characteristics.

The PMD-associated score favors:

- increased intermediate methylation;
- increased low methylation;
- reduced high methylation; and
- penalizes high within-window methylation variability.

The Intermediate score also favors intermediate methylation but allows a greater contribution from highly methylated CpGs, distinguishing it from the PMD-associated state.

MethylSeg evaluates all possible one-to-one mappings between the four K-means clusters and the four biological states. The mapping that maximizes the combined state score across all four clusters is selected as the final cluster-to-state assignment.

K-means clustering is performed using scikit-learn with 10 initializations and a fixed random seed of 42 for reproducibility.

For a complete runnable example, see the {doc}`K-means-based model tutorial </tutorials/generated/03_kmeans_based_model>`.

### 3.2 Rule-based assignment

The rule-based method provides greater interpretability because states are assigned using explicit thresholds rather than cluster labels. However, this comes at the cost of some flexibility compared with K-means. Because each CpG must satisfy predefined cutoff values, small changes in methylation or contextual features can cause adjacent CpGs within an otherwise coherent domain to receive different state assignments. As a result, regions identified as continuous domains by K-means may be fragmented or missed by the rule-based approach.

State assignment considers:

- the beta value of the CpG;
- the proportion of intermediately methylated CpGs in surrounding windows;
- the within-window standard deviation of beta values;
- the proportion of highly methylated CpGs in each window; and
- the proportion of lowly methylated CpGs in each window.

In the default implementation, CpGs are assigned as follows:

- **Low:** beta value is below `beta_low_max` and the PMD rule does not apply.
- **High:** beta value is above `beta_high_min` and the PMD rule does not apply.
- **PMD-associated:** beta value falls within the intermediate range and at least one contextual window satisfies the PMD cutoffs.
- **Intermediate:** none of the preceding conditions is met.

Because the rules are explicit, this approach allows users to determine which characteristics contributed to a state assignment, including the CpG's beta value, local methylation variability, and methylation patterns in the surrounding region.

#### When to use rule-based assignment

Rule-based assignment is most useful when interpretability of individual state assignments is a priority, such as when:

- examining a single sample in detail;
- investigating why individual CpGs received a particular state; or
- defining states using explicit feature thresholds.

For a complete runnable example, see the {doc}`rule-based model tutorial </tutorials/generated/04_rule_based_model>`.

## 4. HMM-based smoothing

Initial CpG-level state assignments can contain isolated state changes caused by biological variability, measurement noise, or uneven spacing between observed CpGs. MethylSeg applies a hidden Markov model (HMM) to smooth these assignments and infer a spatially coherent sequence of methylation states.

After smoothing, consecutive CpGs assigned to the same state are merged into genomic regions.

MethylSeg provides two HMM implementations:

- a continuous-time HMM implemented with `ctHMM` (v0.0.3); and
- a sticky categorical HMM implemented with `hmmlearn` (v0.3.3).

The two HMM implementations serve the same general purpose: reducing isolated state changes and converting CpG-level assignments into spatially coherent methylation domains.

### Continuous-time HMM

The continuous-time HMM explicitly accounts for the genomic distance between consecutive CpGs. This makes it well suited to sparse methylation datasets with irregular CpG spacing, such as HM450K data.

By default, MethylSeg uses the forward-backward algorithm to calculate the posterior probability of each hidden state at every locus and assigns the state with the highest posterior support.

### Sticky categorical HMM

The sticky categorical HMM is a conventional categorical HMM with a strong self-transition prior. This prior encourages neighboring CpGs to remain in the same state and reduces spurious state switching.

MethylSeg uses the Viterbi algorithm to identify the most probable sequence of hidden states given the observed state assignments and fitted model parameters.

### HMM recommendations

For WGBS data, we recommend the sticky categorical HMM because WGBS provides dense, relatively uniform CpG coverage, making explicit modeling of genomic distance less critical while avoiding the additional computational cost of the ctHMM. For sparse microarray datasets, we recommend the continuous-time HMM because it accounts for the irregular genomic distances between neighboring CpGs.

## 5. Region cleaning

MethylSeg separates region cleaning from the initial HMM-derived calls so that the transition from raw segmentation results to final regions remains transparent and configurable.

Cleaning proceeds in three stages:

### 5.1 Incorporate transitional regions

Raw regions may optionally be merged with adjacent Intermediate regions. This allows transitional sequence between methylation states to be retained in downstream analyses.

### 5.2 Merge nearby regions

Regions assigned to the same state can be merged when the genomic gap between them is no more than 100 kb by default.

### 5.3 Apply size thresholds

Regions that do not meet the minimum size requirements are removed. By default, regions must contain at least six CpGs and span at least 5 kb.

The cleaned regions are intended for downstream analyses, while the raw HMM-derived regions remain available for users who want to inspect or customize post-processing.

## Implementation reference

The primary classes used by the MethylSeg workflow are:

| Class | Role |
|---|---|
| `methylseg.MethylSegPathway` | High-level interface for preparing data, fitting model components, running segmentation, and writing outputs. |
| `methylseg.MethylStateAssigner` | Generates context-aware emission features and performs K-means-based state assignment. |
| `methylseg.MethylStateAnalyzer` | Defines, optimizes, and applies rule-based methylation-state assignments. |
| `methylseg.MethylSegmentor` | Performs HMM-based state smoothing and converts the resulting state sequence into genomic regions. |
