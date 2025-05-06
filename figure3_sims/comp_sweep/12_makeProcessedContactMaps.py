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

with open('/net/levsha/share/emily/notebooks/sims/bombyx/toy_models/polymer_info.pkl','rb') as monInfo_file:
    monInfo = pickle.load(monInfo_file)
n_per_chain = monInfo['L']
n_chains = 30
n_chains_per_sphere = 10

n_per_sphere = n_per_chain*n_chains_per_sphere

A_self_attr = [0.00, 0.025, 0.0375, 0.05, 0.075, 0.1]
B_self_attr = A_self_attr
X_self_attr = [0.00, 0.025, 0.05]#, 0.1]

base_path = '/net/levsha/share/emily/notebooks/sims/bombyx/toy_models/compartments_only/sweep_output'
sim_dir = 'SC-10pol'
d_dir = f'lambda-{0}_dX-{0}_dAB-{0}'

interrupt = False 
for XX in X_self_attr:
    for BB in B_self_attr:
        for AA in A_self_attr:
            comp_dir = f'AA{AA:.2f}_BB{BB:.2f}_XX{XX:.2f}'
            sim_dir_path = f'{base_path}/{comp_dir}/{sim_dir}'
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
            save_ooe_fh = f'{sim}__cutoff-{cutoff_rad:04.1f}_binSize-{binSize}_IC_OOE_chainMap.npy'
            
            if os.path.exists(f'{hmap_path}/{save_fh}'):
                #print(f'{sim}:\tHeatmap already processed')
                continue
            if not os.path.exists(f'{hmap_path}/{hmap_fh}'):
                #print(f'{sim}:\tCannot find heatmap {hmap_fh}')
                continue                
            try:
                print(f'{d_dir}   {sim}:\tProcessing heatmap')
                with open(f'{hmap_path}/{hmap_fh}', 'rb') as f:
                    hmap = np.load(f)
                    
                if np.sum(hmap) == 0:
                    os.remove(f'{hmap_path}/{hmap_fh}')
                else:
                    hmap_ice = numutils.iterative_correction_symmetric(hmap*1.)[0]

                    hmap_sum = np.zeros([n_per_chain, n_per_chain])
                    for start in range(0, n_per_sphere, n_per_chain):
                        end = start+n_per_chain
                        hmap_sum += hmap_ice[start:end, start:end]
                    hmap_avg = hmap_sum/n_chains_per_sphere

                    hmap_ooe = numutils.observed_over_expected(hmap_avg)[0]

                    with open(f'{hmap_path}/{save_fh}', 'wb') as f:
                        np.save(f, hmap_avg)
                    with open(f'{hmap_path}/{save_ooe_fh}', 'wb') as f:
                        np.save(f, hmap_ooe)
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
