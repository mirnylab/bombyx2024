"""
takes sphere-average contact map generated in 11_makeSphereContactMaps.py, performs iterative correction on this 3chain by 3chain contact map, generate an observed over expected map, then average its cis signal to generate a 1chain by 1chain observed over expected and ICed contact map
"""
import os
import pickle
import numpy as np
from cooltools.lib import numutils
import traceback
import sys
import pandas as pd

binned = True
binSizes = [10,25]
ignoreDiags_dict = {
    10: 1,
    25: 1,
}

cutoff_rad = 7.5

with open(f'/net/levsha/scratch2/emily/flagship/monomers/mon1d.pkl','rb') as monInfo_file:
    monInfo = pickle.load(monInfo_file)

XX = 0

base_path = f'/net/levsha/share/emily/notebooks/sims/bombyx/flagship/compartment_sweep/sims_mon1d'
sim_dir = 'SC-1JoinedPol_errTol-0.01_coll-0.01'

comp_param_sets = pd.read_csv(f"/net/levsha/share/emily/notebooks/sims/bombyx/flagship/compartment_sweep/AB_param_sets_15Sep23.csv",\
                              sep="\t", index_col=0)

lambda_list = [55, 110]
dX_list     = [19, 37, 55, 110] #28,
AB_fac  = np.array([5,10,np.inf])#np.concatenate([np.arange(2.5,15,2.5), np.array([np.inf])])
max_lodAB = 0.3

interrupt = False 
for binSize in binSizes:
    n_per_copy = int(monInfo['N']/binSize)
    n_copies_per_chain = 3
    n_per_chain = n_copies_per_chain*n_per_copy
    n_chains = int(15/n_copies_per_chain)
    
    for lambda_ in lambda_list:
        for d_X in dX_list:
            dAB_list = d_X*AB_fac
            dAB_list = dAB_list[np.where(lambda_/dAB_list < max_lodAB)[0]]
            dAB_list = np.round(np.where(dAB_list == np.inf, 0, dAB_list)).astype(int)
            for d_AB in dAB_list:
                d_dir = f'lambda-{lambda_}_dX-{d_X}_dAB-{d_AB}'
                for idx in comp_param_sets.index:
                    vals = comp_param_sets.iloc[idx]
                    AA = vals.AA_attr
                    BB = vals.BB_attr

                    comp_dir = f'AA{AA:.2f}_BB{BB:.2f}_XX{XX:.2f}'
                    sim_dir_path = f'{base_path}/{d_dir}/{comp_dir}/{sim_dir}'

                    if os.path.isdir(sim_dir_path) is False:
                        print(f'{sim_dir_path} does not exist')
                        continue    
                    elif os.path.exists(f'{sim_dir_path}/_in_progress_flag'):
                        print(f'{d_dir}/{comp_dir}/{sim_dir}:\t\t\tSimulation is not finished')
                        continue

                    hmap_path = f'{sim_dir_path}/results/heatmaps'

                    sim = f'{comp_dir}__{sim_dir}'
                    hmap_fh = f'{sim}__cutoff-{cutoff_rad:04.1f}_binSize-{binSize}_sphereMap.npy'
                    save_fh = f'{sim}__cutoff-{cutoff_rad:04.1f}_binSize-{binSize}_IC_chainMap.npy'
                    save_ooe_fh = f'{sim}__cutoff-{cutoff_rad:04.1f}_binSize-{binSize}_IC_OOE_chainMap.npy'
                    if os.path.exists(f'{hmap_path}/{save_fh}'):
                        print(f'{sim}:\tHeatmap already processed')
                        continue
                    if not os.path.exists(f'{hmap_path}/{hmap_fh}'):
                        print(f'{hmap_fh} does not exist')
                        continue
                    try:
                        print(f'{d_dir}   {sim}:\tProcessing heatmap')
                        with open(f'{hmap_path}/{hmap_fh}', 'rb') as f:
                            sphere_map = np.load(f)

                        sphere_map_ice = numutils.iterative_correction_symmetric(sphere_map/np.nanmean(sphere_map), \
                                                   ignore_diags=ignoreDiags_dict[binSize])[0]

                        sphere_map_ooe = numutils.observed_over_expected(sphere_map_ice)[0]
                        subchains_hmap = np.array([sphere_map_ice[i*n_per_copy:(i+1)*n_per_copy, i*n_per_copy:(i+1)*n_per_copy] \
                                                   for i in range(0, int(n_per_chain/n_per_copy))])
                        subchains_map_ooe = np.array([sphere_map_ooe[i*n_per_copy:(i+1)*n_per_copy, i*n_per_copy:(i+1)*n_per_copy] \
                                                   for i in range(0, int(n_per_chain/n_per_copy))])
                        hmap_avg = np.nanmean(subchains_hmap, axis=0)
                        map_ooe_avg = np.nanmean(subchains_map_ooe, axis=0)

                        with open(f'{hmap_path}/{save_fh}', 'wb') as f:
                            np.save(f, hmap_avg)
                        with open(f'{hmap_path}/{save_ooe_fh}', 'wb') as f:
                            np.save(f, map_ooe_avg)
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
        if interrupt:
            break   
    if interrupt:
        break
