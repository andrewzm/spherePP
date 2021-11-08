## Reproducible data and scripts for the results in Section 4.2 of the paper Non-Homogeneous Poisson Process Intensity Modeling and Estimation using Measure Transport by James Ng and Andrew Zammit-Mangion

**Authors:**  Tin Lok James Ng and Andrew Zammit-Mangion

**Date:** 07 November 2021

**Description:** Data and code for reproducing the results of the experiment in Section 4.2 of the paper

**Software requirements:** R, Python (>=2.7), PyTorch, Numpy

**Hardware requirements:** Experiments in this section can only be run on a CPU, but some of this code may improve in performance if modified to use a top-end GPU

**Contents:** 
    - pacific.csv              Data containing the end locations of cyclones ni the North Pacific Ocean
    - sphere_pp_vec.py         Fits the model to the cyclone locations data
    - sphere_est_avg.py        Obtains the ensemble average from the model fits
    - sphere_np_boot.py        Obtain nonparametric bootstrap estimates
    - sphere_np_percentile.py  Obtain the 10th and 90th percentiles from nonparametric bootstrap estimates
    - plot_est_intensity.R     Plots the estimated intensity function
    - plot_np_lower.R          Plots the empirical  10  percentile  of the  bootstrap  distribution
    - plot_np_upper.R          Plots the empirical  90  percentile  of the  bootstrap  distribution
    - run_all.sh		     A bash script that runs all of the above

**Notes:**	Please run the bash script or the scripts in the following order:
	- sphere_pp_vec.py 
	- sphere_est_avg.py 
	- sphere_np_boot.py
	- sphere_np_percentile.py
	- plot_est_intensity.R 
	- plot_np_lower.R
	- plot_np_upper.R 
