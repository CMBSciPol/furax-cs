# FURAX Component Separation

[![PyPI version](https://badge.fury.io/py/furax-cs.svg)](https://badge.fury.io/py/furax-cs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue?logo=readthedocs)](https://furax-cs.readthedocs.io/en/latest/)
[![Results Explorer](https://img.shields.io/badge/%F0%9F%A4%97%20Results-Explorer-yellow?)](https://askabalan-furax-cs-results.hf.space/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b?logo=arxiv)](https://arxiv.org/abs/XXXX.XXXXX)

**FURAX-CS** (FURAX Component Separation) is a Python package designed to benchmark and implement advanced component separation techniques for Cosmic Microwave Background (CMB) analysis. It leverages **JAX** for high-performance computing on GPUs and implements novel adaptive clustering methods.

This project specifically focuses on comparing:
*   **FGBuster**: parametric component separation (standard).
*   **FURAX**: Adaptive, gradient-based separation with spatially varying spectral parameters.

### Furax ADABK

Furax CS is a comprehensive software package designed for Component Separation for the Cosmic Microwave Background (CMB) data analysis.
The main tool is the minimizer provided under the name of Furax ADABK which is an adaptive gradient based optimizer specifically designed to handle extremely noise dominated data such as CMB observations and physical bound constraints.
The minimizer is orders of magnitude faster than traditional minimizers such as Scipy-TNC and is able to reach lower minima in fewer iterations.

<p align="center">
  <img src="docs/images/runtime_comparison.png" alt="Runtime Comparison" width="400"/>
</p>

This provides a much easier and faster way to explore the spatial variability of foregrounds and their impact on the CMB recovery.

This has an impact on the estimated r tensor-to-scalar ratio as shown in the figure below where we compare the likelihood profiles obtained with using the KMeans spatial clustering gridding runs and compared with [LiteBIRD PTEP-like](https://arxiv.org/abs/2202.02773) run obtained using FGBuster using multiresolution spatial clustering.

<p align="center">
  <img src="docs/images/r_likelihood_comparison.png" alt="r Likelihood Comparison" width="450"/>
</p>

---


> **Interactive Results:** Explore the results from our paper ([PLACEHOLDER](https://arxiv.org/abs/XXXX.XXXXX)) using the [Results Explorer](https://askabalan-furax-cs-results.hf.space/) Streamlit app.

## Installation

### 1. Prerequisites (JAX)
This package depends on JAX. To enable GPU acceleration (highly recommended), you must install the CUDA version of JAX **before** installing this package.

**For NVIDIA GPUs:**
```bash
pip install -U "jax[cuda]"
```

**For CPU only:**
```bash
pip install jax
```

### 2. Install Package
First, install the package from PyPi

```bash
pip install furax-cs
```

Some packages are not up to date on PyPi, to install the latest development version, install the requirement files after installing furax-cs:

```bash
pip install -r https://raw.githubusercontent.com/CMBSciPol/furax-cs/main/requirements.txt
```

---

## Documentation

Full documentation is available at **[furax-cs.readthedocs.io](https://furax-cs.readthedocs.io/en/latest/)**.

*   **[Quick Start (Python API)](https://furax-cs.readthedocs.io/en/latest/quick_start.html)**: Learn how to use the Python API for data loading and running component separation.
*   **[CLI Reference & Workflow](https://furax-cs.readthedocs.io/en/latest/cli_reference.html)**: Comprehensive guide on using the command-line interface for the full analysis pipeline.
*   **[Minimization Solvers](https://furax-cs.readthedocs.io/en/latest/minimization.html)**: Guide to available optimization algorithms (Active Set, L-BFGS, etc.) and programmatic usage.
*   **[Analysis Tools (r_analysis)](https://furax-cs.readthedocs.io/en/latest/r_analysis/index.html)**: Detailed documentation for the result analysis and plotting suite.

---

## Development

### Running Tests
```bash
pytest
```

### Pre-commit Hooks
Ensure code quality before committing:
```bash
pre-commit install
pre-commit run --all-files
```
