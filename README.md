Black hole spin in X-ray binaries: giving uncertainties an f
===

This repository contains all of the data and source code used in Salvesen & Miller (2020). This paper is an exercise in uncertainty quantification applied to black hole spin measurements in X-ray binaries. Please see the end of this README file for copyright and licensing information.

# Workflow

We consider eight different black hole X-ray binary sources, whose nicknames are: A0620, H1743, J1550, J1655, LMCX1, LMCX3, M33X7, U1543. Python scripts should be run from within the `scripts/` directory because there are some hardcoded paths to data input/output files (alternatively, the user can modify these paths). The pipeline for re-producing the data and results in Salvesen & Miller (2020) for any one of these eight sources is as follows:

## Generate the data (optional)

Running the analysis script `scripts/[source].py` produces the corresponding pickle file `data/CF/CF_results_[source].p`, which contains data and embedded code to sample from published probability distributions specific to that source (e.g., `[source] = J1655`).

Running the analysis script `scripts/gGR_gNT_[source].py` produces the corresponding data file `data/GR/gGR_gNT_[source].h5`, which stores general relativistic correction factors needed in the subsequent analysis. This step requires a local installation of [XSPEC](https://heasarc.gsfc.nasa.gov/xanadu/xspec/).

## Generate the results

Use the data file `data/CF/CF_results_[source].p` as input to the analysis script `scripts/convert_fr2fK.py` to produce the data file `data/fK/fK_[source].h5`, which contains the reverse-engineered probability distribution for the "disk flux normalization" f_K. This step involves manipulating probability densities and inverting an integral transform equation by solving a constrained optimization problem. (See `scripts/fK_commands.txt` for the command to run for each source.)

Use the data file `data/fK/fK_[source].h5` as input to the analysis script `scripts/convert_fK2fr.py` to produce the data file `data/fr/fr_[source].h5`, which contains the updated probability distribution for the "inner disk radius" f_r (equivalent to the black hole spin), taking into account all sources of uncertainty. This step involves manipulating probability densities.

## Plot the results

All Python files in the `scripts/` directory pre-pended with `plot_` are plotting scripts used to produce the figures in Salvesen & Miller (2020).
