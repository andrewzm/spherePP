# Non-Homogeneous Poisson Process Intensity Modeling and Estimation using Measure Transport

This GitHub page provides code for reproducing the results in Section 4.2 of the manuscript titled *Non-Homogeneous Poisson Process Intensity Modeling and Estimation using Measure Transport* by Tin Lok James Ng and Andrew Zammit-Mangion. The manuscript describes the use of normalizing flows, specifically radial flows, to model the intensity function of point process on the sphere.

The figure below depicts results from modeling the end locations of cyclone data in the North Pacific Ocean using the normalizing flows. The center panel shows the intensity-function estimate, while the left and right panels depict the empirical 10 and 90 percentiles of the bootstrap distribution of the intensity function, respectively.

<div class="row">
  <div class="column">
    <img src="pacific_end_dens1_est_radial_v20000_all.png" alt="Predictions" style="width:100%">
  </div>
</div>

## Instructions

Please note the following when trying to reproduce the results in Section 4.2.

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
- run_all.sh		       A bash script that runs all of the above

**Notes:**	Please run the bash script `run_all.sh` or the scripts in the following order:
- sphere_pp_vec.py 
- sphere_est_avg.py 
- sphere_np_boot.py
- sphere_np_percentile.py
- plot_est_intensity.R 
- plot_np_lower.R
- plot_np_upper.R 
