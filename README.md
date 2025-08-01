# NetCP: Network Modelling of Asynchronous Change-Points in Multivariate Time Series

This repository contains the code to reproduce the results from the paper 'Network Modelling of Asynchronous Change-Points in Multivariate Time Series (McKee & Kalli, 2025)' which is available on arxiv [here](https://arxiv.org/abs/2506.15801).

A description of each directory is given below:
1. plots_notebook.ipynb: Provides step by step guidance to reporduce all plots and tables in the paper. This can be done by using the pre-computed results available in this repository, or by re-running the MCMC from scratch.
2. sim_study: Contains the results from the simulation study. This was performed on a high performance computing cluster (approx 200 nodes) and so only the cluster results + plotting code is given here.
3. eeg: Contains all data and scripts to reproduce the eeg results from scratch. Also contains the pre-computed results which are used in the papers plots.
4. seismic: Contains all data and scripts to reproduce the seismology results from scratch. Also contains the pre-computed results which are used in the papers plots.
5. sampler_comparison: Contains code to reproduce the comparison of different MCMC samplers given in the appendix of the paper.
6. src: Contains the raw C++ source code to implement the NetCP model along with the models of Yao (1984) and Quinland et al. (2024) using both particle MCMC and single site gibbs samplers. 
