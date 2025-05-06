import os
import pickle 
import h5py
from copy import deepcopy as dcopy
import numpy as np
import numpy.ma as ma
import pandas as pd
import traceback
import polychrom
from polychrom import polymer_analyses, polymerutils
from polychrom.hdf5_format import list_URIs, load_URI
from cooltools.lib import numutils
import multiprocess as mp
import argparse

# parser = argparse.ArgumentParser(description='Makes heatmaps')
# parser.add_argument('-c', '--CPU', help='Number of CPUs to use', default=15, type=int)
# args = parser.parse_args()


def Ps_sorter(blocks, bin_edges, chains, cutoff=1.1):

    def process(uri):
        idx = int(uri.split('::')[-1])
        try:
            data = load_URI(uri)['pos']
        except:
            print(uri)

        ser = {}
        chunk = np.searchsorted(blocks, idx, side='right')
        ser['chunk'] = [chunk]

        bins = None
        contacts = np.zeros(len(bin_edges)-1)
        for st, end in zip(chains[0:-1],chains[1:]):
            conf = data[st:end,:]
            x,y = polymer_analyses.contact_scaling(conf, bins0=bin_edges, cutoff=cutoff)
            
            if bins is None:
                bins = x

            contacts = contacts + y

        ser['Ps'] = [(bins, contacts)]
        return pd.DataFrame(ser)

    return process

block_eq = 30_000
cutoff_rad = 5 

T_steps = 500_000
steps_per_save = 10
T_blocks = T_steps//steps_per_save

# load compartment labels
with open('../polymer_info.pkl', 'rb') as f:
    monInfo = pickle.load(f)    
n_per_chain = monInfo['L']
n_chains = 30
chains = [i*n_per_chain for i in range(n_chains+1)]

mon_id = monInfo['compartment_ID']

AA = BB = XX = 0.00

base_path = '/net/levsha/share/emily/notebooks/sims/bombyx/toy_models/compartments_only/sweep_output'
sim_dir = 'SC-10pol'
comp_dir = f'AA{AA:.2f}_BB{BB:.2f}_XX{XX:.2f}'
sim_dir_path = f'{base_path}/{comp_dir}/{sim_dir}'
save_path = f'{sim_dir_path}/results/Ps_scaling'
save_fh = f'{comp_dir}__{sim_dir}__cutoff{cutoff_rad:04.1f}.txt'

os.makedirs(save_path, exist_ok=True)

bin_edges = numutils._logbins_numba(1, n_per_chain, ratio=1.25, prepend_zero=True)

uris = list(list_URIs(f'{sim_dir_path}/', return_dict=True).values())
if len(uris) == T_blocks:
    start = block_eq
elif len(uris) == T_blocks - block_eq:
    start = 0
    
end = T_blocks
blocks = np.array([start, end]).astype(int)

f = Ps_sorter(blocks, bin_edges=bin_edges, chains=chains, cutoff=cutoff_rad)
with mp.Pool(20) as p:
    results = p.imap_unordered(f, uris, chunksize=5)
    df = polymer_analyses.streaming_ndarray_agg(results, 
                                                chunksize=5000,
                                                ndarray_cols=['Ps'], 
                                                aggregate_cols=['chunk'], 
                                                add_count_col=True, divide_by_count=True
                                                )

Ps = df.iloc[0]['Ps']
df = pd.DataFrame({'dist':Ps[0,:], 'Ps':Ps[1,:]})
df.to_csv(f'{save_path}/{save_fh}', sep='\t', header=True, index=False)
