import numpy as np
import json
import math

print("### Averaging over committee members ###")

samp_all = []
nsim = 50
for i in range(nsim):
    file = 'pacific_end_dens_est_radial_v' + str(i+20000) + '.json'
    with open(file) as data_file:
        samp = json.load(data_file)
    samp_all.append(samp)

n = len(samp_all[0])
avg_list = []
for i in range(n):
    lamda_list = []
    for j in range(nsim):
        if not(math.isnan(samp_all[j][i][2])): lamda_list.append(samp_all[j][i][2])
    lamda_list.sort()
    lamda = np.mean(lamda_list)
    avg_list.append([samp_all[0][i][0], samp_all[0][i][1], float(lamda)])

with open('pacific_end_dens_est_radial_est_avg.json', 'w') as f:
    json.dump(avg_list, f)
