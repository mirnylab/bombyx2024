import os, sys
import pickle, h5py
import glob
import time
import shutil
import warnings
import traceback
import argparse
from copy import deepcopy as dcopy
import manySphericalSpheres
import numpy as np

import openmm
import polychrom
from polychrom import simulation, starting_conformations, forces, forcekits
from polychrom.simulation import Simulation
from polychrom.starting_conformations import grow_cubic
from polychrom.hdf5_format import HDF5Reporter

parser = argparse.ArgumentParser(description='X Compartment Simulator')
parser.add_argument('-g', '--GPU', help='GPU number to use')
parser.add_argument('-o', '--out_path', default='SC-10pol', help='location where simulation save folder is created')
args = parser.parse_args()

def initiate_sim(polymer, r, cell_size, chains, GPU): 
        
    sim = simulation.Simulation(
        platform="CUDA",
        GPU=GPU,
        integrator="variableLangevin",
        error_tol=0.01,
        collision_rate=0.01,
        N=len(polymer),
        save_decimals=2,
        reporters=[reporter],
    )
    sim.set_data(polymer)
    
    sim.add_force(manySphericalSpheres.spherical_confinement_many(sim, r=r, cell_size=cell_size))
        
    
    sim.add_force(
        forcekits.polymer_chains(
            sim,
            
            # Bond forces
            chains=chains, 
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                "bondLength": 1.0, # avg dist or dist at rest
                "bondWiggleDistance": 0.05,  # Bond distance will fluctuate +- 0.05 on average
            },

            # Angle forces
            angle_force_func=forces.angle_force,
            angle_force_kwargs={
                "k": 1.5, # persistence length; re-evaluate this for your monomer size
            },
            
            # Non bonded forces
            nonbonded_force_func=forces.heteropolymer_SSW,
            nonbonded_force_kwargs={
                'attractionEnergy': 0,
                'attractionRadius': 1.5,
                'interactionMatrix': ixn_mtx,
                'monomerTypes': mon_types,
                'extraHardParticlesIdxs': []
            },

            except_bonds=True,
        )
    )
    
    return(sim)

# Defining chains
with open('/net/levsha/share/emily/notebooks/sims/bombyx/toy_models/polymer_info.pkl','rb') as monInfo_file:
    monInfo = pickle.load(monInfo_file)
n_per_chain = monInfo['L']
n_chains = 30
chain_list = n_chains*[n_per_chain]
N = sum(chain_list)

if n_chains == 1: # unclear if this actually is needed
    chains = [(0, None, False)]
else:
    chains = [(i*n_per_chain, (i+1)*n_per_chain, False) for i in range(n_chains)]

T_steps = 500_000
steps_per_save = 10
T_blocks = T_steps//steps_per_save

Nmd = 750

# Spherical Confinement
# Parameters for initializing conformations and positioning each 5 chains into its own sphere
density = 0.2
n_chains_per_sphere = 10
r = (3 * n_per_chain * n_chains_per_sphere / (4 * 3.141592 * density)) ** (1/3)
cell_size = 5*r
offsets = [[cell_size*i,0,0] for i in range(int(n_chains/n_chains_per_sphere))]
offsets_multi = np.repeat(np.array(offsets), int(n_per_chain*n_chains_per_sphere), axis=0)

# Block copol params
mon_types = np.tile(monInfo['compartment_ID'], n_chains)
AB = XA = XB = 0.00
A_self_attr = [0.00, 0.025, 0.05, 0.075, 0.1]
B_self_attr = dcopy(A_self_attr)
X_self_attr = [0.00, 0.05]

max_data_length=100
interrupt = False

base_path = '/net/levsha/share/emily/notebooks/sims/bombyx/toy_models/compartments_only/sweep_output'

for XX in X_self_attr:
    for BB in B_self_attr:
        for AA in A_self_attr:

            extrusion_conds = f'lambda-{0}_dX-{0}_dAB-{0}'
            compartment_conds = f'AA{AA:.2f}_BB{BB:.2f}_XX{XX:.2f}'
            
            save_path = f'{base_path}/AA{AA:.2f}_BB{BB:.2f}_XX{XX:.2f}/{args.out_path}'
            os.makedirs(save_path, exist_ok=True)    
            print(f'\n\nAA: {AA:.2f}\tBB: {BB:.2f}\tXX: {XX:.2f}')
            try:
                with open(f'{save_path}/_in_progress_flag', 'x') as fp:
                    pass       
            except FileExistsError:
                print("\tSimulation is currently running on another GPU\n\n")
                continue

            reporter = HDF5Reporter(folder=save_path, max_data_length=100, 
                                    overwrite=False, check_exists=False, blocks_only=False
                                   )

            try:
                if os.path.exists(f'{save_path}/blocks_{T_blocks-max_data_length}-{T_blocks-1}.h5'):
                    print("\tSimulation is complete\n\n")
                    continue

                try:
                    current_block, data = reporter.continue_trajectory()
                    polymer = data['pos']
                    current_block = current_block + 1
                    print(f'\tFound existing simulation with {current_block} saved blocks...')
                    print(f'\tSimulating and saving another {T_blocks - current_block} blocks.\n\n\n\n')

                except ValueError:
                    current_block = 0
                    print(f'\tStarting new simulation...')
                    print(f'\tSimulating {T_blocks} blocks.\n\n\n\n')
                    confs_starting = np.concatenate([np.array(starting_conformations.grow_cubic(n_per_chain*n_chains_per_sphere, 
                                                                            int((n_per_chain*n_chains_per_sphere)**(1/3)+2))) 
                                 for i in range(int(n_chains/n_chains_per_sphere))])
                    polymer = confs_starting+offsets_multi

                ixn_mtx = np.array([[AA, AB, XA],
                                   [AB, BB, XB],
                                   [XA, XB, XX],
                                  ])


                sim = initiate_sim(polymer, r=r, cell_size=cell_size, chains=chains, GPU=args.GPU) 
                if not current_block:
                    sim.local_energy_minimization() 
                else:
                    sim._apply_forces()

                for i in range(T_blocks - current_block):  
                    if i % 25 == 0:
                        print(f'\n\n\t\tAA: {AA:.2f}\tBB: {BB:.2f}\tXX: {XX:.2f}\nNo Extrusion\n\n')
                    sim.do_block(steps=Nmd*steps_per_save) 

            except KeyboardInterrupt:
                print(traceback.format_exc())
                interrupt = True
                break
            except:
                print(traceback.format_exc())
            finally:
                os.remove(f'{save_path}/_in_progress_flag')
            reporter.dump_data()
            if interrupt:
                break
        if interrupt:
            break
    if interrupt:
        break
