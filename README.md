# NetCP: Network Modelling of Asynchronous Change-Points in Multivariate Time Series

This repository contains the code to reproduce the results from the paper 'Network Modelling of Asynchronous Change-Points in Multivariate Time Series' (McKee & Kalli, 2025) which is available on arxiv [here](https://arxiv.org/abs/2506.15801).

A description of each directory/file is given below:  
* `plots_notebook.ipynb` Provides step by step guidance to reporduce all plots and tables in the paper. This can be done by using the pre-computed results available in this repository, or by re-running the MCMC from scratch.  
* `sim_study` Contains the results from the simulation study. This was performed on a high performance computing cluster (approx 200 nodes) and so only the cluster results + plotting code is given here.  
* `eeg` Contains all data and scripts to reproduce the eeg results from scratch. Also contains the pre-computed results which are used in the papers plots.  
* `seismic` Contains all data and scripts to reproduce the seismology results from scratch. Also contains the pre-computed results which are used in the papers plots.  
* `sampler_comparison` Contains code to reproduce the comparison of different MCMC samplers given in the appendix of the paper.  
* `src` Contains the raw C++ source code to perform MCMC on the NetCP model model along with the models of Yao (1984) and Quinlan et al. (2024) using both particle MCMC and single site gibbs samplers.  

## 🛠️ Compiler Requirement

If you wish to re-run the MCMC from scratch, a **g++ compiler** is required to build and run the C++ code. 

### Installation Instructions

- **macOS**:  
  Install via [Homebrew](https://brew.sh/):
  ```bash
  brew install gcc

- **Windows**  
  Install [MinGW-w64](https://www.mingw-w64.org/) or use [MSYS2](https://www.msys2.org/).
  After installation, ensure g++ is added to your system PATH.
  You can verify it by running:
  ```bash
  g++ --version
