"""
takes sphere-average contact map generated in 11_makeSphereContactMaps.py, performs iterative correction on this 10chain by 10chain contact map,  then average its cis contacts to generate a 1chain by 1chain average, then computes and returns an observed over expected map
"""
import os
import pickle
import numpy as np
from cooltools.lib import numutils
import traceback
import sys

cutoff_rad = 5
binSize = 1

# Extrusion params
lambda_ = 100
d_X = 50
dAB_list = [50, 100, 200, 400, 1000]

with open('/net/levsha/share/emily/notebooks/sims/bombyx/toy_models/polymer_info.pkl','rb') as monInfo_file:
    monInfo = pickle.load(monInfo_file)
n_per_chain = monInfo['L']
n_chains = 30
n_chains_per_sphere = 10

n_per_sphere = n_per_chain*n_chains_per_sphere

AB = XA = XB = 0.00
AB_self_attr = [0, 0.025, 0.05, 0.1, 0.15, 0.25]
XX = 0

base_path = '/net/levsha/share/emily/notebooks/sims/bombyx/toy_models/extrusion_and_compartments/extrusion_density_sims'
sim_dir_list = ['Xboundaries']

interrupt = False
for sim_dir in sim_dir_list:
    for d_AB in dAB_list:

        d_dir = f'lambda-{lambda_}_dX-{d_X}_dAB-{d_AB}'
        for AB in AB_self_attr:
            BB = AA = AB
            comp_dir = f'AA{AA:.2f}_BB{BB:.2f}_XX{XX:.2f}'
            sim_dir_path = f'{base_path}/{d_dir}/{comp_dir}/{sim_dir}'

            if os.path.isdir(sim_dir_path) is False:
                print(f'{sim_dir_path} does not exist')
                continue    
            elif os.path.exists(f'{sim_dir_path}/_in_progress_flag'):
                print(f'{comp_dir}/{sim_dir}:\t{d_dir}\t\tSimulation is not finished')
                continue

            hmap_path = f'{sim_dir_path}/results/heatmaps'

            sim = f'{comp_dir}__{sim_dir}'
            hmap_fh = f'{sim}__cutoff-{cutoff_rad:04.1f}_binSize-{binSize}_sphereMap_smartCutoff.npy'
            save_fh = f'{sim}__cutoff-{cutoff_rad:04.1f}_binSize-{binSize}_IC_chainMap.npy'

            if os.path.exists(f'{hmap_path}/{save_fh}'):
                print(f'{d_dir}   {sim}:\tHeatmap already processed')
                continue
            if not os.path.exists(f'{hmap_path}/{hmap_fh}'):
                print(f'{d_dir} {sim}:\tCannot find heatmap {hmap_fh}')
                continue                
            try:
                print(f'{d_dir}   {sim}:\tProcessing heatmap')
                with open(f'{hmap_path}/{hmap_fh}', 'rb') as f:
                    hmap = np.load(f)

                hmap_ice = numutils.iterative_correction_symmetric(hmap*1.)[0]

                hmap_sum = np.zeros([n_per_chain, n_per_chain])
                for start in range(0, n_per_sphere, n_per_chain):
                    end = start+n_per_chain
                    hmap_sum += hmap_ice[start:end, start:end]
                hmap_avg = hmap_sum/n_chains_per_sphere

                with open(f'{hmap_path}/{save_fh}', 'wb') as f:
                    np.save(f, hmap_avg)
            except KeyboardInterrupt:
                print(traceback.format_exc())
                interrupt = True
                break

            if interrupt:
                break   
        if interrupt:
            break  
    if interrupt:
        break   

