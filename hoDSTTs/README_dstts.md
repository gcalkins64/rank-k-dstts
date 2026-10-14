# sttgmm



## Overview

This code will be used for a journal paper (technical note) on using DSTTs for aerocapture, focusing on why we need a time-varying direction and need to consider the higher order directions for this problem for deterministic perturbation propagation and uncertainty propagation for a single Gaussian

##  Monte Carlo Workflow
The workflow to generate Monte Carlo samples is as follows:

- `./inputs/` contains the input cases for all simulations
- `get_monte_carlo_results` in Python runs the Monte Carlo simulation
- `plot_monte_carlo_results` in Python plots the Monte Carlo results

##  STT / DSTT Workflow
The workflow to generate the STTs / DSTTs / HOCGTs and generate Gaussian Mixture Model propagations are as follows:

- `inputs` contains the input cases for all simulations
- `get_regime_ALL_cgts` in Python computes the nonsymmetric HOCGTs for the 4 regimes and all CGT  types (HOCGTs, qCGTs, SCGTs)
- `get_sliding_window_sym_cgts.m` in MATLAB symmetrizes the HOCGTs and computes the tensor eigenpairs using the SS-HOPM method and saves them off for comparison and use in DSTT computation
- `get_Xdstts_regime` in Python computes the STTs from tk to tkp1 for the parameters and computes  all types of DSTTs (DSTTs, pDSTTs, hoDSTTs, sDSTTs, qDSTTs)
- `get_gmmuqs` in Python computes GMM propagation for all propagation methods
- `plot_compare_gmm_uqs` in Python plots all the GMM results and compares them