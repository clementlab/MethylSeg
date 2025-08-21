from enum import Enum
import logging
import sys
from .segmentation_steps.data_prep import DataPrep
from .segmentation_steps.meth_seg import MethSegMethod
from .segmentation_steps.region_identifier import GenerateMethylationRegions
from .segmentation_steps.preprocess_window_data import WindowPreprocessor


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class MethylSegSteps(Enum):
    """
    Enumeration of the steps involved in the methylation pathway process.
    """

    PREPROCESS = "preprocess"
    DATA_PREP = "data_prep"
    GENERATE_METHYLATION_REGIONS = "generate_methylation_regions"


class MethylSegConfig:

    def __init__(
        self,
        window_size=100000,
        step_size=50000,
        filter_chroms=[],
        filter_regions=[],
        filter_samples=[],
        out_dir="",
        tmp_folder="/tmp",
        disable_tqdm=False,
        disable_loading_from_cache=False,
        non_null_cutoff=0.9,
        int_count_cutoff=3,
        avg_cpg_cutoff=8,
        n_jobs=1,
        window_summary_metric="pct",
        show_plots=False,
    ):
        self.window_summary_metric = window_summary_metric
        self.window_size = window_size
        self.step_size = step_size
        self.filter_chroms = filter_chroms
        self.filter_regions = filter_regions
        self.filter_samples = filter_samples
        if self.filter_samples and len(self.filter_samples) < 10:
            logger.warning(
                "Filter samples list is less than 10. This may lead to insufficient data for analysis and downstream errors."
            )
        self.out_dir = out_dir
        self.tmp_folder = tmp_folder
        self.disable_tqdm = disable_tqdm
        self.disable_loading_from_cache = disable_loading_from_cache
        self.non_null_cutoff = non_null_cutoff
        self.int_count_cutoff = int_count_cutoff
        self.avg_cpg_cutoff = avg_cpg_cutoff
        self.n_jobs = n_jobs
        self.show_plots = show_plots


class MethylSeg:

    def __init__(
        self,
        meth_ref_path,
        samples_info_path,
        selected_samples_path,
        genome_file,
        config: MethylSegConfig,
    ):
        window_size = config.window_size
        step_size = config.step_size
        tmp_folder = config.tmp_folder
        disable_tqdm = config.disable_tqdm
        disable_loading_from_cache = config.disable_loading_from_cache
        self.show_plots = config.show_plots
        self.out_dir = config.out_dir

        self.data_preprocessor = WindowPreprocessor(
            meth_ref_file=meth_ref_path,
            samples_file=samples_info_path,
            selected_samples_file=selected_samples_path,
            window_size=window_size,
            step_size=step_size,
            genome_file=genome_file,
            tmp_dir=tmp_folder,
            out_dir=f"{self.out_dir}/{MethylSegSteps.PREPROCESS.value}",
            disable_loading_from_cache=disable_loading_from_cache,
            n_jobs=config.n_jobs,
        )
        self.data_prep = DataPrep(
            preprocessor=self.data_preprocessor,
            use_window_averaging=True,
            out_dir=f"{self.out_dir}/{MethylSegSteps.DATA_PREP.value}",
            disable_loading_from_cache=disable_loading_from_cache,
        )
        self.generate_methylation_regions = GenerateMethylationRegions(
            data_prep=self.data_prep,
            seg_method=MethSegMethod.GAUSSIAN_HMM,
            selected_samples_path=selected_samples_path,
            num_filter=20,
            n_jobs=50,
            out_dir=f"{self.out_dir}/{MethylSegSteps.GENERATE_METHYLATION_REGIONS.value}",
            disable_loading_from_cache=disable_loading_from_cache,
        )
        self.data_processed = False
        self.processed_samples = {}
        self.aggregated_regions = {}

    def preprocess_data(self):
        self.data_preprocessor.run(self.show_plots)
        self.data_prep.run(self.show_plots)
        self.generate_methylation_regions.init()
        self.data_processed = True

    def process_single_sample(self, sample_id, region_type="intermediate"):
        if not self.data_processed:
            self.preprocess_data()

        if region_type == "all":
            return {
                "intermediate": self.process_single_sample(sample_id, "intermediate"),
                "high": self.process_single_sample(sample_id, "high"),
                "low": self.process_single_sample(sample_id, "low"),
            }
        if not self.disable_loading_from_cache and sample_id in self.processed_samples:
            regions = self.processed_samples[sample_id]
            if region_type in regions:
                return regions[region_type]
        regions = self.generate_methylation_regions.segmentor.generate_genomic_regions(
            sample=sample_id, region_type=region_type
        )
        if sample_id not in self.processed_samples:
            self.processed_samples[sample_id] = {}
        self.processed_samples[sample_id][region_type] = regions
        return regions

    def generate_common_regions(self, region_type="intermediate"):
        if not self.data_processed:
            self.preprocess_data()
        if (
            not self.disable_loading_from_cache
            and region_type in self.aggregated_regions
        ):
            return self.aggregated_regions[region_type]
        if region_type == "all":
            return {
                "intermediate": self.generate_common_regions("intermediate"),
                "high": self.generate_common_regions("high"),
                "low": self.generate_common_regions("low"),
            }
        self.aggregated_regions[region_type] = (
            self.generate_methylation_regions.segmentor.aggregate_genomic_regions(
                n_jobs=self.generate_methylation_regions.n_jobs,
                fit_args=self.generate_methylation_regions.segmentor_args,
                region_type=region_type,
            )
        )
        return self.aggregated_regions[region_type]
