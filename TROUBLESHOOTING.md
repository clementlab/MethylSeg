# Troubleshooting

## LLVM crash on ARM64 systems

On some ARM64 systems, certain versions of Numba/LLVM may terminate with an `incomplete machine model` error. This is an [upstream Numba/LLVM issue](https://github.com/numba/numba/issues/10388), rather than a MethylSeg error.

### Conda environments

As a workaround, configure Numba to use a generic CPU target in the active Conda environment:

```bash
conda env config vars set NUMBA_CPU_NAME=generic
conda deactivate
conda activate <environment-name>
```

MethylSeg commands and Python scripts launched from the activated environment should then run normally.

### Python scripts

Alternatively, set the CPU target directly in Python. This must occur before importing MethylSeg, Numba, or any package that imports Numba:

```python
import os

os.environ["NUMBA_CPU_NAME"] = "generic"

import methylseg
```

### VS Code Jupyter notebooks

If you are using vscode jypyter nodebooks, we reccommend creating a dedicated kernel:

```bash
conda activate <environment-name>
python -m pip install ipykernel

python -m ipykernel install \
    --prefix "$CONDA_PREFIX" \
    --name methylseg-arm64 \
    --display-name "Python (MethylSeg ARM64)"

jupyter kernelspec list
```

Open the `kernel.json` file listed for `methylseg-arm64` and add the following top-level entry:

```json
"env": {
  "NUMBA_CPU_NAME": "generic"
}
```

For example:

```json
{
  "argv": [
    "/path/to/environment/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python (MethylSeg ARM64)",
  "language": "python",
  "env": {
    "NUMBA_CPU_NAME": "generic"
  }
}
```

Reload VS Code and select **Jupyter Kernel → Python (MethylSeg ARM64)**. The custom kernel appears under Jupyter kernels rather than under Python environments.

### CT-HMM smoothing warnings with microarray data

Occasionally (around 10% of tested samples) microarray samples may emit a nonfatal warning such as:

```text
WARNING: failure to converge. delta=nan
```

This is a CT-HMM fitting/smoothing warning, not a segmentation warning. It occurs while MethylSeg estimates the continuous-time transition behavior between CpGs. Segmentation continues afterward and can still write region outputs.

To address these warnings, users can adjust CT-HMM fitting parameters such as the initial holding time, convergence tolerance, or iteration limit.

```python
from methylseg import HMMType, MethylSegPathway

pathway = MethylSegPathway(
    train_sample_info=sample_info,
    out_dir="methylseg_output",
    hmm_type=HMMType.CT,
    hmm_params={
        "n_emissions": 4,
        "holding_time_guess": 1_500_000,
        "algorithm": "forward-backward",
        "max_iter": 100,
        "tol": 1e-3,
    },
)

pathway.run_pathway()
```


