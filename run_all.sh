#!/bin/bash
python sphere_pp_vec.py 
python sphere_est_avg.py
python sphere_np_boot.py
python sphere_np_percentile.py
Rscript plot_est_intensity.R
Rscript plot_np_lower.R
Rscript plot_np_upper.R
