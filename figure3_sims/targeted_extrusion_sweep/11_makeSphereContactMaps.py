import polychrom
import numpy as np 
import h5py 
import glob
import os 
import shutil
import pickle
from copy import deepcopy as dcopy
import traceback
import polychrom.polymerutils
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file, save_hdf5_file
from polychrom import contactmaps
from polychrom.polymer_analyses import smart_contacts
import argparse

parser = argparse.ArgumentParser(description='Makes heatmaps')
parser.add_argument('-c', '--CPU', help='Number of CPUs to use', default=15, type=int)
args = parser.parse_args()

block_eq = 30_000
cutoff_rad = 5 
binSize = 1

# Extrusion params
n_lefIters = 500_000
lambda_ = 100
d_X = 50
dAB_list = [50, 100, 200, 400, 1000]

n_ints_per_lefIter = 10
T_blocks = int(n_lefIters/n_ints_per_lefIter)
max_data_length = 100

base_path = '/net/levsha/share/emily/notebooks/sims/bombyx/toy_models/extrusion_and_compartments/extrusion_density_sims'
sim_dir_list = ['Xboundaries']

interrupt = False

with open('/net/levsha/share/emily/notebooks/sims/bombyx/toy_models/polymer_info.pkl','rb') as monInfo_file:
    monInfo = pickle.load(monInfo_file)
n_per_chain = monInfo['L']
n_chains = 30
n_chains_per_sphere = 10
n_per_sphere = n_per_chain*n_chains_per_sphere
chains = np.asarray([(i*n_per_chain, (i+1)*n_per_chain) for i in range(0, n_chains)])
sphere_starts = [chains[i][0] for i in range(0,n_chains,n_chains_per_sphere)]

A_self_attr = [0, 0.025, 0.05, 0.1, 0.15, 0.25]
XX = 0.00

interrupt = False
for sim_dir in sim_dir_list:
    for d_AB in dAB_list:

        d_dir = f'lambda-{lambda_}_dX-{d_X}_dAB-{d_AB}'

        for AA in A_self_attr:
            BB = AA
            comp_dir = f'AA{AA:.2f}_BB{BB:.2f}_XX{XX:.2f}'
            sim_dir_path = f'{base_path}/{d_dir}/{comp_dir}/{sim_dir}'

            if os.path.exists(f'{sim_dir_path}/_in_progress_flag'):
                print(f'{d_dir}/{comp_dir}:\t\t\tSimulation is not finished')
                continue

            final_conf_fh = f'blocks_{T_blocks-max_data_length}'+\
                        f'-{T_blocks-1}.h5'
            save_path = f'{sim_dir_path}/results/heatmaps'
            sim = f'{comp_dir}__{sim_dir}'
            save_fh = f'{sim}__cutoff-{cutoff_rad:04.1f}_binSize-{binSize}_sphereMap_smartCutoff.npy'
            
            if os.path.exists(f'{sim_dir_path}') is False:
                print(f'{d_dir}/{comp_dir}:\t\tsimulation not started yet')
                continue
            elif os.path.exists(f'{sim_dir_path}/{final_conf_fh}') is False:
                print(f'cannot find {sim_dir_path}/{final_conf_fh}')
                continue

            elif os.path.exists(f'{save_path}/{save_fh}'):
                print(f'{d_dir}/{comp_dir}:\t Heatmap already extracted for contact radius: {cutoff_rad:04.1f}')
                continue

            os.makedirs(save_path, exist_ok=True)    
            try:
                with open(f'{save_path}/_in_progress_flag', 'x') as fp:
                    print("")  
            except FileExistsError:
                print(f'{d_dir}/{comp_dir}:\tHeatmap is being calculated on another machine')
                continue

            try:
                print(f'{sim}  {d_dir}:\t Extracting Heatmap for contact radius: {cutoff_rad:04.1f}, binsize: {binSize}')
                uris = list(list_URIs(f'{sim_dir_path}/', return_dict=True).values())

                mat = contactmaps.monomerResolutionContactMapSubchains(
                                                        method=lambda x, cutoff: \
                                                            smart_contacts(x,cutoff,percent_func=lambda z:1.5/z),
                                                        filenames=uris[block_eq:],  
                                                        mapStarts=sphere_starts, 
                                                        mapN=n_per_sphere, 
                                                        cutoff=cutoff_rad, 
                                                        n=int(args.CPU), 
                                                        loadFunction=lambda x:load_URI(x)["pos"]
                                                      )

                with open(f'{save_path}/{save_fh}', 'wb') as f:
                    np.save(f, mat)

            except KeyboardInterrupt:
                print(traceback.format_exc())
                interrupt = True
                break    
            finally:
                os.remove(f'{save_path}/_in_progress_flag')
            if interrupt:
                break    
        if interrupt:
            break   
    if interrupt:
        break 



