# FURAX-CS Documentation

**GPU-accelerated CMB parametric component separation** built on [JAX](https://github.com/google/jax) and [FURAX](https://github.com/CMBSciPol/furax).

FURAX-CS implements adaptive K-means and multi-resolution clustering strategies for spatially-varying foreground modeling of Cosmic Microwave Background (CMB) data, with benchmarks against FGBuster.

## Installation

```bash
# Install JAX (CPU or CUDA)
pip install -U "jax[cpu]"       # CPU-only
pip install -U "jax[cuda]"      # GPU (recommended for production)

# Install furax-cs
pip install -e ".[all]"
```

```{toctree}
:maxdepth: 2
:caption: User Guide

quick_start
cli_reference
minimization
```

```{toctree}
:maxdepth: 2
:caption: Analysis

r_analysis/index
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

notebooks/01_FGBuster_vs_FURAX_Comparison
notebooks/02_KMeans_Adaptive_Component_Separation
notebooks/03_PTEP_Multi_Resolution_Component_Separation
notebooks/04_Scripts_and_Analysis_Workflow
Estimating r <notebooks/05_Estimating_r>
notebooks/06_Optimizer_Benchmark
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```
