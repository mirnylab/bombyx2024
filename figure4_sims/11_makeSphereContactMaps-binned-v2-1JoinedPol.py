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
from cooltools.lib import numutils
import argparse
import sys

parser = argparse.ArgumentParser(description='Makes heatmaps')
parser.add_argument('-c', '--CPU', help='Number of CPUs to use', default=15, type=int)
args = parser.parse_args()


n_chains = 15
block_eq = 1_500
cutoff_rad = 7.5
binSize = 10

n_ints_per_lefIter = 10
n_lefIters = 100_000
max_data_length=100

base_path = '/net/levsha/scratch2/emily/flagship/compartment_sweep/sims_chr15:0-6p5Mb'

interrupt = False

with open(f'chr15:0-6p5Mb.pkl','rb') as monInfo_file:
    monInfo = pickle.load(monInfo_file)
n_per_chain = 3*monInfo['N']
n_chains = int(15/3)
chains = np.array([(i*n_per_chain, (i+1)*n_per_chain) for i in range(0, n_chains)])
L = n_per_chain*n_chains

n_chains_per_sphere = 1
n_per_sphere = n_per_chain*n_chains_per_sphere
sphere_starts = [chains[i][0] for i in range(0,n_chains,n_chains_per_sphere)]

lam_list = [55, 110]
dX_list = [19, 37, 55, 110]
AB_fac  = np.array([5,10,np.inf])
max_lodAB = 0.3

# extrusion_dir = {
#     0: {
#         "dX" : np.array([0]), 
#         "dAB": np.array([0])
#     }
# }

# for lambda_ in lam_list:
#     dAB_list = d_X*AB_fac
#     dAB_list = dAB_list[np.where(lambda_/dAB_list < max_lodAB)[0]]
    
#     extrusion_dir[lambda_] = {}
#     extrusion_dir[lambda_]["dX"] = np.round(lambda_/lodX_list).astype(int)
#     extrusion_dir[lambda_]["dAB"] = np.round(np.where(dAB_list == np.inf, 0, dAB_list)).astype(int)

sim_dir = 'SC-1JoinedPol_errTol-0.01_coll-0.01'
for lambda_ in lam_list:
    for d_X in dX_list:
        dAB_list = d_X*AB_fac
        dAB_list = dAB_list[np.where(lambda_/dAB_list < max_lodAB)[0]]
        dAB_list = np.round(np.where(dAB_list == np.inf, 0, dAB_list)).astype(int)
        for d_AB in dAB_list:
            d_dir = f'lambda-{lambda_}_dX-{d_X}_dAB-{d_AB}'
            if os.path.isdir(f'{base_path}/{d_dir}') is False:
                continue   
            comp_dirs = os.listdir(f'{base_path}/{d_dir}')
            for comp_dir in comp_dirs:
                if comp_dir == '.ipynb_checkpoints':
                    continue
                if os.path.isdir(f'{base_path}/{d_dir}/{comp_dir}') is False:
                        continue    
                sim_dirs = os.listdir(f'{base_path}/{d_dir}/{comp_dir}')

                sim_dir_path = f'{base_path}/{d_dir}/{comp_dir}/{sim_dir}'

                if os.path.isdir(sim_dir_path) is False:
                    continue    
                elif os.path.exists(f'{sim_dir_path}/_in_progress_flag'):
                    print(f'{comp_dir}/{sim_dir}:\t\t\tSimulation is not finished')
                    continue


                final_conf_fh = f'blocks_{int((n_lefIters/n_ints_per_lefIter)-max_data_length)}'+\
                            f'-{int(n_lefIters/n_ints_per_lefIter)-1}.h5'
                save_path = f'{sim_dir_path}/results/heatmaps'
                sim = f'{comp_dir}__{sim_dir}'
                save_fh = f'{sim}__cutoff-{cutoff_rad:04.1f}_binSize-{binSize}_sphereMap.npy'

                if os.path.exists(f'{sim_dir_path}/{final_conf_fh}') is False:
                    print(f'cannot find {sim_dir_path}/{final_conf_fh}')
                    continue

                if os.path.exists(f'{save_path}/{save_fh}'):
                    print(f'{sim}:\t Heatmap already extracted for contact radius: {cutoff_rad:04.1f}')
                    continue

                os.makedirs(f'{save_path}', exist_ok=True)

                try:
                    with open(f'{save_path}/_in_progress_flag', 'x') as fp:
                        print("")  
                except FileExistsError:
                    print(f'{sim}:\tHeatmap is being calculated on another machine')
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

                    mat = numutils.zoom_array(mat, (n_per_sphere/binSize, n_per_sphere/binSize))

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
    if interrupt:
        break   