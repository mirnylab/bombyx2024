import pickle
import os
import sys
import time
import numpy as np
import polychrom
from polychrom import simulation, starting_conformations, forces, forcekits
from polychrom.hdf5_format import HDF5Reporter
import warnings
import h5py
import argparse
import traceback

sys.path.append("../")
import manySphericalSpheres


parser = argparse.ArgumentParser(description='Loop extrusion and Compartment Simulator')
parser.add_argument('-g', '--GPU', help='GPU number to use', type=str)
parser.add_argument('-o', '--out_path', default='SC-1JoinedPol_errTol-0.01_coll-0.01', help='location where simulation save folder is created')
args = parser.parse_args()

def initiate_sim(polymer, r, cell_size, chains, GPU, ixn_mtx, mon_types): #dims
        
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
                "k": 1.5, # persistence length
            },

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
with open(f'chr15:0-6p5Mb.pkl','rb') as monInfo_file:
    monInfo = pickle.load(monInfo_file) # mon info for each copy of the polymer
mon_conds = args.monomer_filename.split(".")[0]
base_path =  f'/net/levsha/scratch2/emily/flagship'
n_per_chain = 3*monInfo['N']
n_chains = int(15/3)
chain_list = n_chains*[n_per_chain]
N = sum(chain_list)

if n_chains == 1:
    chains = [(0, None, False)]
else:
    chains = [(i*n_per_chain, (i+1)*n_per_chain, False) for i in range(n_chains)]

# Block copol params
mon_types = np.tile(monInfo['mon'], 3*n_chains)
AB = XA = XB = 0.00
XX = 0.00

# Extrusion params
traj_fh = 'LEFpositions_steps-100000_nChains-15.h5'
lambda_ = 0
d_X = 0
d_AB = 0

# Other things
max_data_length = 100
T_steps = 100_000
steps_per_save = 10
Nmd = 750
T_blocks = T_steps//steps_per_save

# Variables for initializing conformations to later position each chain into its own sphere
density = 0.2
n_chains_per_sphere = 1
r = (3 * n_per_chain * n_chains_per_sphere / (4 * 3.141592 * density)) ** (1/3)
cell_size = 5*r
offsets = [[cell_size*i,0,0] for i in range(int(n_chains/n_chains_per_sphere))]
offsets_multi = np.repeat(np.array(offsets), int(n_per_chain*n_chains_per_sphere), axis=0)
A_self_attr = np.round(np.arange(0.00, 0.30, 0.02), 2)
B_self_attr = np.round(np.arange(0.00, 0.20, 0.02), 2)

interrupt = False
for AA in A_self_attr:
    for BB in B_self_attr:
 
        extrusion_conds = f'lambda-{0}_dX-{0}_dAB-{0}'
        compartment_conds = f'AA{AA:.2f}_BB{BB:.2f}_XX{XX:.2f}'

        save_path = f'{base_path}/compartment_sweep/sims_{mon_conds}/{extrusion_conds}/{compartment_conds}/{args.out_path}'
        print(save_path)
        sys.exit()
        os.makedirs(save_path, exist_ok=True)

        print(f'\n\nAA: {AA:.2f}\tBB: {BB:.2f}\tXX: {XX:.2f}')
        print(extrusion_conds)
        try:
            with open(f'{save_path}/_in_progress_flag', 'x') as fp:
                pass       
        except FileExistsError:
            print("\tSimulation is currently running on another GPU\n\n")
            continue

        reporter = HDF5Reporter(folder=save_path, max_data_length=max_data_length, 
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



            sim = initiate_sim(polymer, r=r, cell_size=cell_size, chains=chains, GPU=args.GPU, ixn_mtx=ixn_mtx, mon_types=mon_types) 
            if not current_block:
                sim.local_energy_minimization()
            else:
                sim._apply_forces()
            print('\n')

            for i in range(T_blocks - current_block):
                if i % 25 == 0:
                    print(f'\n\n\t\tAA: {AA:.2f}\tBB: {BB:.2f}\tXX: {XX:.2f}\nNo Extrusion\n\n')
                sim.do_block(Nmd*steps_per_save)

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


