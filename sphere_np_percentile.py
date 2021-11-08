import numpy as np
import json
import math


samp_all = []
nsim = 50
for i in range(nsim):
    file = 'pacific_end_dens_est_radial_np_v' + str(i+20000) + '.json'
    with open(file) as data_file:
        samp = json.load(data_file)
    samp_all.append(samp)

n = len(samp_all[0])
lower_list = []
upper_list = []
for i in range(n):
    lamda_list = []
    for j in range(nsim):
        if not(math.isnan(samp_all[j][i][2])): lamda_list.append(samp_all[j][i][2])
    l = len(lamda_list)
    lamda_list.sort()
    uq = round(l*0.9) - 1
    lq = round(l*0.1) - 1
    lamda = lamda_list[uq]
    upper_list.append([samp_all[0][i][0], samp_all[0][i][1], float(lamda)])
    lamda = lamda_list[lq]
    lower_list.append([samp_all[0][i][0], samp_all[0][i][1], float(lamda)])

with open('pacific_end_dens_est_radial_upper_v20000.json', 'w') as f:
    json.dump(upper_list, f)
with open('pacific_end_dens_est_radial_lower_v20000.json', 'w') as f:
    json.dump(lower_list, f)
